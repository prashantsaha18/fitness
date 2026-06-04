"""
scripts/train_kaggle.py
————————————————————————
End-to-end DeepFM training on the four Kaggle fitness datasets.

Pipeline:
  1. Generate / load Kaggle datasets (synthetic or real)
  2. ETL  ->  [N, 409] float32 feature matrix + binary labels
  3. Train DeepFM with MTL loss + early stopping
  4. Evaluate: AUC-ROC, log-loss
  5. Export trained model to ONNX with graph optimisations
  6. Save JSON training report

Usage:
  python scripts/train_kaggle.py                   # defaults (15 epochs)
  python scripts/train_kaggle.py --epochs 5        # quick smoke-test
  python scripts/train_kaggle.py --use-real-data   # Kaggle API download
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ranking.model import DeepFMRanker   # single canonical definition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_ROOT    = _ROOT / "data" / "kaggle_raw"
ARTIFACT_DIR = _ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# == CLI =====================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepFM on Kaggle fitness datasets")
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch-size",    type=int,   default=2048)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=3)
    p.add_argument("--use-real-data", action="store_true")
    return p.parse_args()


# == Dataset =================================================================

class KaggleDS(Dataset):
    def __init__(self, X: np.ndarray, yc: np.ndarray, yk: np.ndarray) -> None:
        self.X  = torch.from_numpy(X).half()          # float16 to halve RAM
        self.yc = torch.from_numpy(yc.astype(np.float32))
        self.yk = torch.from_numpy(yk.astype(np.float32))

    def __len__(self) -> int:
        return len(self.yc)

    def __getitem__(self, i: int) -> tuple:
        return self.X[i].float(), self.yc[i], self.yk[i]


# == Evaluation ==============================================================

def _evaluate(model: DeepFMRanker, loader: DataLoader) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score
    model.eval()
    sc, sl, kc, kl = [], [], [], []
    with torch.no_grad():
        for Xb, yc, yk in loader:
            c, k = model(Xb)
            sc.extend(torch.sigmoid(c).squeeze(-1).tolist())
            sl.extend(yc.tolist())
            kc.extend(torch.sigmoid(k).squeeze(-1).tolist())
            kl.extend(yk.tolist())

    def _safe(yt, ys):
        if len(set(yt)) < 2:
            return 0.5, 1.0
        return round(roc_auc_score(yt, ys), 4), round(log_loss(yt, ys), 4)

    ac, lc = _safe(sl, sc)
    ak, lk = _safe(kl, kc)
    return {"auc_click": ac, "ll_click": lc,
            "auc_complete": ak, "ll_complete": lk,
            "auc_combined": round(0.4*ac + 0.6*ak, 4)}


# == ONNX export =============================================================

def _export_onnx(model: DeepFMRanker, path: str) -> bool:
    try:
        import onnx, onnxruntime as ort

        class _W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m.predict_proba(x)

        w = _W(model); w.eval()
        torch.onnx.export(w, torch.randn(64, 409), path,
            opset_version=17, do_constant_folding=True,
            input_names=["features"], output_names=["engagement_score"],
            dynamic_axes={"features":{0:"batch"},"engagement_score":{0:"batch"}})
        onnx.checker.check_model(onnx.load(path))
        out = ort.InferenceSession(path).run(
            ["engagement_score"],
            {"features": np.random.randn(10,409).astype(np.float32)}
        )[0]
        assert out.shape == (10, 1)
        log.info("ONNX verified  shape=%s  path=%s", out.shape, path)
        return True
    except Exception as exc:
        log.warning("ONNX export skipped: %s", exc)
        return False


# == Training loop ===========================================================

def train(X, y_click, y_complete, *, epochs=15, batch_size=2048, lr=1e-3, patience=3):
    N = len(X)
    nv, nt = int(N*.15), int(N*.15)
    nr     = N - nv - nt

    full = KaggleDS(X, y_click, y_complete)
    tr, va, te = random_split(full, [nr, nv, nt],
                               generator=torch.Generator().manual_seed(42))
    kw = dict(num_workers=0)
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True, **kw)
    val = DataLoader(va, batch_size=batch_size*2, **kw)
    tel = DataLoader(te, batch_size=batch_size*2, **kw)

    model    = DeepFMRanker()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Params: %s  |  train=%d val=%d test=%d", f"{n_params:,}", nr, nv, nt)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, eta_min=5e-6)

    best_auc, best_state, no_imp = 0.0, None, 0
    history = []

    log.info("  %4s  %8s  %7s  %7s  %9s  %6s", "Ep","Loss","AUC-C","AUC-K","COMBINED","ms")
    log.info("  " + "-"*50)

    for ep in range(1, epochs+1):
        model.train(); tot, nb = 0.0, 0; t0 = time.perf_counter()
        for Xb, yc, yk in trl:
            opt.zero_grad(set_to_none=True)
            l = model.compute_mtl_loss(Xb, yc, yk)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += l.item(); nb += 1
        sched.step(); ms = (time.perf_counter()-t0)*1000

        m = _evaluate(model, val); model.train()
        mark = " *" if m["auc_combined"] > best_auc else ""
        log.info("  %4d  %8.4f  %7.4f  %7.4f  %9.4f  %6.0f%s",
                 ep, tot/nb, m["auc_click"], m["auc_complete"], m["auc_combined"], ms, mark)
        history.append({"epoch":ep,"loss":round(tot/nb,4),**m})

        if m["auc_combined"] > best_auc:
            best_auc, best_state, no_imp = m["auc_combined"], {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                log.info("  Early stopping at epoch %d", ep); break

    model.load_state_dict(best_state)
    tm = _evaluate(model, tel)

    ckpt = ARTIFACT_DIR / "ranking_model_kaggle.pt"
    torch.save({"model_state_dict":model.state_dict(),"test_metrics":tm,
                "n_params":n_params,"history":history}, ckpt)

    onnx_path = str(ARTIFACT_DIR / "ranking_model.onnx")
    _export_onnx(model, onnx_path)

    return {"test_metrics":tm,"best_val_auc":round(best_auc,4),"history":history,
            "n_params":n_params,"train_size":nr,"val_size":nv,"test_size":nt,
            "checkpoint":str(ckpt),"onnx_model":onnx_path}


# == Data preparation ========================================================

def _prepare(use_real: bool):
    if use_real:
        try:
            from data_pipeline.kaggle.downloader import KaggleDownloader
            KaggleDownloader(DATA_ROOT).download_all()
        except Exception as e:
            log.warning("Kaggle download failed (%s) — using synthetic data.", e)

    if not (DATA_ROOT / "fitbit" / "dailyActivity_merged.csv").exists():
        log.info("Generating synthetic Kaggle-schema datasets...")
        from data_pipeline.kaggle.synthetic_datasets import generate_all
        generate_all(DATA_ROOT, verbose=True)

    log.info("Building TF-IDF embeddings (JL projection 384-dim)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.random_projection import SparseRandomProjection

    CORPUS = {
        "HIIT":     "high intensity interval training hiit anaerobic sprint burst cardio metabolic conditioning fat loss plyometrics explosive power",
        "Strength": "strength training resistance hypertrophy muscle building compound lifts progressive overload barbell squat deadlift bench press powerlifting",
        "Yoga":     "yoga mindfulness flexibility balance posture alignment breathing pranayama sun salutation warrior poses meditation stress relief recovery",
        "Cardio":   "cardio cardiovascular endurance aerobic running cycling rowing steady state zone two fat adaptation vo2 max mitochondria heart health",
        "Pilates":  "pilates core stability postural alignment spine health pelvic floor deep core mat reformer controlled movement low impact rehabilitation",
    }
    corpus_list = list(CORPUS.values()) * 40
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=2000, sublinear_tf=True)
    tfidf.fit(corpus_list)
    rp = SparseRandomProjection(n_components=384, random_state=42, density=0.1)
    rp.fit(tfidf.transform(corpus_list))

    def _emb(wt):
        v = rp.transform(tfidf.transform([CORPUS.get(wt, CORPUS["Cardio"])])).toarray().astype(np.float32)[0]
        return v / max(np.linalg.norm(v), 1e-8)

    log.info("Running ETL...")
    from data_pipeline.kaggle.etl import (
        engineer_fitbit_labels, engineer_fitbit_user_features,
        engineer_gym_features, engineer_mental_features, hr_to_zone,
        load_fitbit, load_fitness_2024, load_gym_members, load_mental_health,
    )
    fitbit = load_fitbit(DATA_ROOT); gym = load_gym_members(DATA_ROOT)
    f24    = load_fitness_2024(DATA_ROOT); mental = load_mental_health(DATA_ROOT)

    daily = fitbit["dailyActivity_merged"]
    labeled    = engineer_fitbit_labels(daily)
    user_feats = engineer_fitbit_user_features(labeled, fitbit.get("sleepDay_merged"), fitbit.get("weightLogInfo_merged"))
    gym_feats  = engineer_gym_features(gym)
    mh         = engineer_mental_features(mental)
    mean_stress = float(mh["stress_score"].mean())
    labeled    = labeled.merge(user_feats, on="Id", suffixes=("","_agg"))

    hr_df = fitbit.get("heartrate_seconds_merged")
    user_hr = hr_df.groupby("Id")["Value"].median().to_dict() if hr_df is not None else {}

    rng = np.random.default_rng(42)
    gi  = gym_feats["intensity_score"].values
    rows_X, rows_yc, rows_yk = [], [], []

    # FitBit x Gym cross-join
    for _, row in labeled.iterrows():
        act = float(row.get("activity_level", 0.5))
        w   = np.exp(-4*(gi - act)**2); w /= w.sum()
        idx = int(rng.choice(len(gym_feats), p=w))
        gr  = gym_feats.iloc[idx]
        vec = np.zeros(409, dtype=np.float32)
        vec[:384] = _emb(str(gr.get("Workout_Type","Cardio")))
        vec[384]  = float(gr.get("workout_type_enc",3))/6
        vec[385]  = 0.5; vec[386] = float(row.get("fitness_goal_enc",0))/4
        vec[387]  = float(gr.get("hr_zone_norm",0.5))
        vec[391]  = math.sin(2*math.pi*9/24); vec[392] = math.cos(2*math.pi*9/24); vec[393]=0.5
        se = float(row.get("sleep_efficiency",0.82))
        vec[394]=min(float(user_hr.get(row["Id"],70))/220,1); vec[395]=float(np.clip(act*(1-se)*0.7+mean_stress*0.3,0,1))
        vec[396]=se; vec[397]=float(gr.get("calories_norm",0.3)); vec[398]=float(gr.get("global_ctr",0.12))
        vec[399]=float(gr.get("global_completion_rate",0.5))
        vec[400]=math.log1p(float(gr.get("Workout_Frequency(days/week)",3))*100)/20
        vec[401]=float(gr.get("intensity_score",0.5)); vec[402]=float(gr.get("duration_norm",0.4))
        vec[406]=float(row.get("bmi",23.5))/40; vec[408]=float(row.get("adherence_rate",0.5))
        rows_X.append(vec); rows_yc.append(int(row["click_label"])); rows_yk.append(int(row["complete_label"]))

    # Fitness 2024 rows
    df24 = f24.copy(); df24.columns = [c.strip().replace(" ","_") for c in df24.columns]
    cal_col = next((c for c in df24.columns if "Calorie" in c), None)
    if cal_col:
        med = float(df24[cal_col].median())
        for _, row in df24.iterrows():
            wt=str(row.get("Workout_Type","Cardio")); cal=float(row.get(cal_col,300))
            avg=float(row.get("Avg_BPM",130)); mxb=float(row.get("Max_BPM",max(avg+20,185)))
            dur=float(row.get("Session_Duration_(hours)",1) or 1); freq=float(row.get("Workout_Frequency_(days/week)",3) or 3)
            age=int(row.get("Age",30)); bmi=float(row.get("BMI",23.5)); exp=int(row.get("Experience_Level",2))
            inten=min(avg/max(mxb,1),1); hrz=hr_to_zone(avg,age)
            vec=np.zeros(409,dtype=np.float32); vec[:384]=_emb(wt)
            wt_map={"HIIT":0,"Strength":1,"Yoga":2,"Cardio":3,"Pilates":4}
            vec[384]=wt_map.get(wt,3)/6; vec[385]=0.5; vec[387]=hrz/4
            vec[394]=avg/220; vec[395]=min(inten*0.5+freq/7*0.3,1); vec[396]=1-inten*0.4
            vec[397]=min(cal/1500,2); vec[398]=0.15-inten*0.05; vec[399]=max(0.3,0.85-inten*0.3)
            vec[400]=math.log1p(freq*100)/20; vec[401]=inten; vec[402]=min(dur/3,1)
            vec[404]=min(cal/1500,2); vec[406]=bmi/40; vec[407]=age/100; vec[408]=min((exp-1)/2,1)
            click=1 if cal>med*0.85 else 0; complete=1 if (cal>med and inten>0.55 and dur>=0.75) else 0
            rows_X.append(vec); rows_yc.append(click); rows_yk.append(complete)

    X=np.vstack(rows_X).astype(np.float32); yc=np.array(rows_yc,np.int8); yk=np.array(rows_yk,np.int8)
    assert np.isfinite(X).all(), "Non-finite values in feature matrix."
    assert len(set(yc.tolist()))==2 and len(set(yk.tolist()))==2, "Labels have no variance."
    log.info("Matrix %s  click=%.1f%%  complete=%.1f%%", X.shape, yc.mean()*100, yk.mean()*100)
    return X, yc, yk


# == Entry point =============================================================

def main() -> None:
    args = _parse_args()
    print()
    print("+-----------------------------------------------------------------+")
    print("|  FitAI -- DeepFM Training Pipeline                             |")
    print("+-----------------------------------------------------------------+")
    X, yc, yk = _prepare(args.use_real_data)
    res = train(X, yc, yk, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience)
    tm = res["test_metrics"]
    print()
    print(f"  AUC combined : {tm['auc_combined']:.4f}  (click {tm['auc_click']:.4f}  complete {tm['auc_complete']:.4f})")
    print(f"  Best val AUC : {res['best_val_auc']:.4f}")
    print(f"  Checkpoint   : {Path(res['checkpoint']).name}")
    print(f"  ONNX         : {Path(res['onnx_model']).name}")
    report = {"datasets":{"fitbit":"arashnic/fitbit","mental_health":"bhavikjikadara/mental-health-dataset",
              "fitness_2024":"sonialikhan/fitness-track-daily-activity-dataset-2024",
              "gym_members":"valakhorasani/gym-members-exercise-dataset"},
              "embedding":"TF-IDF+SparseRandomProjection(384d,JL)",
              **{k:v for k,v in res.items() if k!="history"}, "history":res["history"]}
    rp = ARTIFACT_DIR/"training_report_kaggle.json"
    rp.write_text(json.dumps(report, indent=2))
    log.info("Report -> %s", rp)


def main_sync() -> None:
    main()


if __name__ == "__main__":
    main()
