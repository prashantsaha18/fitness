"""
data_pipeline/kafka_simulator.py
─────────────────────────────────
Async real-time telemetry pipeline simulating wearable device streams.

Event taxonomy:
  ┌─────────────────────────────────────────────────────────────┐
  │  Topic: fitness.telemetry.realtime                          │
  │  Partition key: user_id (guarantees per-user ordering)      │
  │  Throughput target: 5,000 events/sec aggregate              │
  │  Retention: 24h (raw telemetry → Flink/Spark aggregation)   │
  └─────────────────────────────────────────────────────────────┘

The TelemetryProducer is a simulation engine; swap aiokafka for the
real Confluent Kafka SDK in production without changing the downstream
consumer contract (Avro schema remains identical).

Consumer usage pattern:
  The online feature store consumer (TelemetryConsumer) maintains a
  sliding-window aggregation in Redis, keyed by user_id, with TTL=300s.
  This feeds directly into Feast's online retrieval path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.errors import KafkaConnectionError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from config.settings import settings

logger = structlog.get_logger(__name__)


# ── Avro-compatible Telemetry Event Schema ────────────────────────────────────

@dataclass
class TelemetryEvent:
    """
    Immutable snapshot of a user's biometric state at a single timestamp.
    All numeric fields use SI units for cross-device normalisation.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = "biometric_snapshot"

    # ── Cardio ─────────────────────────────────────────────────────────────
    heart_rate_bpm: int = 0               # instantaneous HR
    heart_rate_zone: str = "resting"      # resting/fat_burn/cardio/peak/anaerobic
    spo2_pct: float = 98.0                # blood oxygen saturation

    # ── Activity ───────────────────────────────────────────────────────────
    active_calories_kcal: float = 0.0    # cumulative session calories
    steps_per_minute: int = 0
    cadence_rpm: float = 0.0             # cycling cadence
    power_watts: float = 0.0

    # ── Subjective / Computed ─────────────────────────────────────────────
    fatigue_level: float = 0.0           # 0.0 (fresh) – 1.0 (exhausted)
    perceived_exertion_rpe: float = 0.0  # Borg 6-20 scale, normalised 0–1
    recovery_score: float = 1.0          # HRV-derived, 0–1

    # ── Device Context ────────────────────────────────────────────────────
    device_type: str = "smartwatch"
    device_id: str = ""
    firmware_version: str = "1.0.0"

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "TelemetryEvent":
        return cls(**json.loads(data))


def _simulate_heart_rate_zone(hr: int) -> str:
    """Map instantaneous HR to training zone."""
    if hr < 100: return "resting"
    if hr < 120: return "fat_burn"
    if hr < 150: return "cardio"
    if hr < 170: return "peak"
    return "anaerobic"


def _simulate_fatigue(active_cal: float, hr: int, session_minutes: float) -> float:
    """Heuristic fatigue model: integrates caloric expenditure + HR stress."""
    cal_factor = min(active_cal / 600.0, 1.0)
    hr_factor = min((hr - 60) / 140.0, 1.0) if hr > 60 else 0.0
    time_factor = min(session_minutes / 90.0, 1.0)
    return round(0.4 * cal_factor + 0.35 * hr_factor + 0.25 * time_factor, 3)


class TelemetryGenerator:
    """
    Stateful per-user telemetry simulation.
    Models a realistic workout arc: warm-up → peak → cool-down.
    """

    def __init__(self, user_id: str, session_id: str | None = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.device_id = str(uuid.uuid4())
        self._session_start = time.monotonic()
        self._base_hr = random.randint(55, 75)
        self._peak_hr = random.randint(150, 185)
        self._cumulative_calories = 0.0

    def generate_event(self) -> TelemetryEvent:
        elapsed_min = (time.monotonic() - self._session_start) / 60.0

        # Arc: 0–5min ramp-up, 5–40min sustained, 40–50min cool-down
        if elapsed_min < 5:
            progress = elapsed_min / 5.0
        elif elapsed_min < 40:
            progress = 1.0
        else:
            progress = max(0.0, 1.0 - (elapsed_min - 40) / 10.0)

        hr = int(
            self._base_hr
            + (self._peak_hr - self._base_hr) * progress
            + random.gauss(0, 4)
        )
        hr = max(50, min(220, hr))

        spm = int(80 * progress + random.gauss(0, 5))
        cal_rate = (0.05 * hr) * (1 / 60)  # ~kcal per second
        self._cumulative_calories += cal_rate
        fatigue = _simulate_fatigue(self._cumulative_calories, hr, elapsed_min)

        return TelemetryEvent(
            user_id=self.user_id,
            session_id=self.session_id,
            heart_rate_bpm=hr,
            heart_rate_zone=_simulate_heart_rate_zone(hr),
            spo2_pct=round(random.gauss(98.2, 0.5), 1),
            active_calories_kcal=round(self._cumulative_calories, 2),
            steps_per_minute=spm,
            cadence_rpm=round(spm / 2.0 + random.gauss(0, 2), 1),
            power_watts=round(hr * 0.8 * progress + random.gauss(0, 10), 1),
            fatigue_level=fatigue,
            perceived_exertion_rpe=round(fatigue * 0.8 + random.gauss(0, 0.05), 3),
            recovery_score=round(1.0 - fatigue * 0.7 + random.gauss(0, 0.02), 3),
            device_id=self.device_id,
        )


# ── Kafka Producer ────────────────────────────────────────────────────────────

class TelemetryProducer:
    """
    Async Kafka producer for the telemetry stream.
    Implements idempotent delivery (enable_idempotence=True) to prevent
    duplicate events on producer retry after network partitions.
    """

    def __init__(self):
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        if not KAFKA_AVAILABLE:
            logger.warning("aiokafka not available — running in mock mode")
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: v,           # bytes passthrough
            key_serializer=lambda k: k.encode(),
            enable_idempotence=True,
            acks="all",
            compression_type="lz4",
            linger_ms=5,                            # micro-batch for throughput
            max_batch_size=65536,
            request_timeout_ms=10_000,
        )
        await self._producer.start()
        logger.info("Kafka producer started", broker=settings.KAFKA_BOOTSTRAP_SERVERS)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish_event(self, event: TelemetryEvent) -> None:
        if not self._producer:
            return  # silent no-op in mock mode
        await self._producer.send(
            topic=settings.KAFKA_TELEMETRY_TOPIC,
            key=event.user_id,
            value=event.to_json(),
        )


# ── Simulation Harness ────────────────────────────────────────────────────────

async def run_simulation(
    num_users: int = 500,
    events_per_second_per_user: float = 0.2,  # ~1 event per 5 seconds per user
    duration_seconds: float = 60.0,
) -> None:
    """
    Load-simulation driver targeting ~5,000 RPS aggregate.
    num_users=500, events/sec/user=10 → 5,000 events/sec.
    """
    producer = TelemetryProducer()
    await producer.start()

    generators = {
        str(uuid.uuid4()): TelemetryGenerator(user_id=str(uuid.uuid4()))
        for _ in range(num_users)
    }

    interval = 1.0 / events_per_second_per_user
    deadline = time.monotonic() + duration_seconds
    event_count = 0

    logger.info(
        "Telemetry simulation started",
        num_users=num_users,
        target_rps=num_users * events_per_second_per_user,
    )

    async def _emit_for_user(gen: TelemetryGenerator) -> None:
        nonlocal event_count
        evt = gen.generate_event()
        await producer.publish_event(evt)
        event_count += 1

    try:
        while time.monotonic() < deadline:
            batch_start = time.monotonic()
            tasks = [_emit_for_user(gen) for gen in generators.values()]
            await asyncio.gather(*tasks)
            elapsed = time.monotonic() - batch_start
            sleep_time = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_time)
    finally:
        await producer.stop()
        logger.info("Simulation complete", total_events=event_count)


# ── Kafka Consumer (Online Feature Updater) ───────────────────────────────────

class TelemetryConsumer:
    """
    Consumes telemetry events and writes sliding-window aggregates to Redis.

    Aggregation keys (Redis Hash):
      user:{user_id}:realtime_features
        hr_mean_5min     → exponential moving average of HR
        fatigue_latest   → most recent fatigue reading
        cal_total_session → cumulative active calories
        last_event_ts    → ISO-8601 timestamp of last event received
    """

    def __init__(self, redis_client):
        self._redis = redis_client
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        if not KAFKA_AVAILABLE:
            return
        self._consumer = AIOKafkaConsumer(
            settings.KAFKA_TELEMETRY_TOPIC,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=f"{settings.KAFKA_CONSUMER_GROUP}-feature-updater",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            fetch_max_wait_ms=100,
            max_poll_records=500,
        )
        await self._consumer.start()
        asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        EMA_ALPHA = 0.15  # smoothing factor for HR EMA
        try:
            async for msg in self._consumer:
                event = TelemetryEvent.from_json(msg.value)
                key = f"user:{event.user_id}:realtime_features"

                # Read existing EMA; default to current reading on first event
                existing = await self._redis.hget(key, "hr_mean_5min")
                prev_hr = float(existing) if existing else float(event.heart_rate_bpm)
                new_hr_ema = EMA_ALPHA * event.heart_rate_bpm + (1 - EMA_ALPHA) * prev_hr

                await self._redis.hset(
                    key,
                    mapping={
                        "hr_mean_5min": round(new_hr_ema, 2),
                        "fatigue_latest": event.fatigue_level,
                        "cal_total_session": event.active_calories_kcal,
                        "recovery_score": event.recovery_score,
                        "heart_rate_zone": event.heart_rate_zone,
                        "last_event_ts": event.timestamp_utc,
                    },
                )
                await self._redis.expire(key, settings.REDIS_FEATURE_TTL_SECONDS)
        except Exception as exc:
            logger.error("Consumer loop error", error=str(exc))
        finally:
            if self._consumer:
                await self._consumer.stop()


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fitness telemetry simulator")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            num_users=args.users,
            duration_seconds=args.duration,
        )
    )
