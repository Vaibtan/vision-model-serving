"""Redis-backed atomic prediction job state."""

from __future__ import annotations

from math import ceil
from time import time
from typing import Callable, Protocol

from .contracts import (
    GatewayObservations,
    GatewayUnavailable,
    IdempotencyConflict,
    PredictionFailure,
    PredictionHandle,
    PredictionId,
    PredictionJobState,
    PredictionNotFound,
    PredictionStatus,
    QueueSaturated,
)
from .state import Admission, JobStateRecord


class _RedisClient(Protocol):
    def eval(self, script: str, key_count: int, *values: object) -> object: ...


_ADMIT_SCRIPT = r"""
local now = tonumber(ARGV[1])
local prediction_id = ARGV[2]
local fingerprint = ARGV[3]
local locator = ARGV[4]
local submitted_at = ARGV[5]
local lease_expires_at = ARGV[6]
local capacity = tonumber(ARGV[7])
local record_ttl = tonumber(ARGV[8])
local idempotency_enabled = ARGV[9] == '1'
local job_prefix = ARGV[10]
local result_ttl = tonumber(ARGV[12])
local tombstone_ttl = tonumber(ARGV[13])

local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now)
for _, expired_id in ipairs(expired) do
  local expired_job = job_prefix .. expired_id
  local expired_state = redis.call('HGET', expired_job, 'state')
  if expired_state == 'queued' or expired_state == 'running' then
    local failure_code = 'prediction_worker_lost'
    local failure_detail = 'prediction worker was lost'
    if expired_state == 'queued' then
      failure_code = 'prediction_reservation_expired'
      failure_detail = 'prediction queue reservation expired'
    else
      redis.call('HINCRBY', KEYS[4], 'worker_lost_total', 1)
    end
    redis.call(
      'HSET', expired_job,
      'state', 'failed',
      'completed_at', now,
      'result_expires_at', now + result_ttl,
      'failure_code', failure_code,
      'failure_detail', failure_detail,
      'failure_retryable', '0'
    )
    redis.call('EXPIRE', expired_job, math.ceil(result_ttl + tombstone_ttl))
    redis.call('HINCRBY', KEYS[4], 'failed_total', 1)
  end
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)

if idempotency_enabled then
  local existing_id = redis.call('GET', KEYS[3])
  if existing_id then
    local existing_job = job_prefix .. existing_id
    local existing_fingerprint = redis.call('HGET', existing_job, 'request_fingerprint')
    local existing_state = redis.call('HGET', existing_job, 'state')
    local existing_result_expires_at = tonumber(redis.call('HGET', existing_job, 'result_expires_at') or '0')
    if (existing_state == 'succeeded' or existing_state == 'failed') and existing_result_expires_at > 0 and now >= existing_result_expires_at then
      redis.call('HSET', existing_job, 'state', 'expired')
      redis.call('EXPIRE', existing_job, math.ceil(tombstone_ttl))
      existing_state = 'expired'
    end
    if existing_fingerprint and existing_state ~= 'expired' then
      if existing_fingerprint ~= fingerprint then
        return {-1, existing_id}
      end
      return {2, existing_id}
    end
    redis.call('DEL', KEYS[3])
  end
end

if redis.call('ZCARD', KEYS[1]) >= capacity then
  redis.call('HINCRBY', KEYS[4], 'rejected_total', 1)
  return {0, ''}
end

redis.call(
  'HSET', KEYS[2],
  'prediction_id', prediction_id,
  'locator', locator,
  'request_fingerprint', fingerprint,
  'idempotency_digest', ARGV[11],
  'state', 'queued',
  'submitted_at', submitted_at,
  'lease_expires_at', lease_expires_at,
  'result_expires_at', '',
  'started_at', '',
  'completed_at', '',
  'failure_code', '',
  'failure_detail', '',
  'failure_retryable', '0'
)
redis.call('EXPIRE', KEYS[2], record_ttl)
redis.call('ZADD', KEYS[1], lease_expires_at, prediction_id)
if idempotency_enabled then
  redis.call('SET', KEYS[3], prediction_id, 'EX', record_ttl)
end
redis.call('HINCRBY', KEYS[4], 'admitted_total', 1)
redis.call('EXPIRE', KEYS[4], record_ttl * 10)
return {1, prediction_id}
"""


_REFRESH_SCRIPT = r"""
if redis.call('EXISTS', KEYS[1]) == 0 then
  return {}
end
local now = tonumber(ARGV[1])
local result_ttl = tonumber(ARGV[2])
local tombstone_ttl = tonumber(ARGV[3])
local state = redis.call('HGET', KEYS[1], 'state')
local lease_expires_at = tonumber(redis.call('HGET', KEYS[1], 'lease_expires_at') or '0')
local prediction_id = redis.call('HGET', KEYS[1], 'prediction_id')

if (state == 'queued' or state == 'running') and now >= lease_expires_at then
  local failure_code = 'prediction_worker_lost'
  local failure_detail = 'prediction worker was lost'
  if state == 'queued' then
    failure_code = 'prediction_reservation_expired'
    failure_detail = 'prediction queue reservation expired'
  else
    redis.call('HINCRBY', KEYS[3], 'worker_lost_total', 1)
  end
  redis.call(
    'HSET', KEYS[1],
    'state', 'failed',
    'completed_at', now,
    'result_expires_at', now + result_ttl,
    'failure_code', failure_code,
    'failure_detail', failure_detail,
    'failure_retryable', '0'
  )
  redis.call('ZREM', KEYS[2], prediction_id)
  redis.call('HINCRBY', KEYS[3], 'failed_total', 1)
  redis.call('EXPIRE', KEYS[1], math.ceil(result_ttl + tombstone_ttl))
  state = 'failed'
end

local result_expires_at = tonumber(redis.call('HGET', KEYS[1], 'result_expires_at') or '0')
if (state == 'succeeded' or state == 'failed') and result_expires_at > 0 and now >= result_expires_at then
  local idempotency_digest = redis.call('HGET', KEYS[1], 'idempotency_digest')
  if idempotency_digest and idempotency_digest ~= '' then
    redis.call('DEL', ARGV[4] .. idempotency_digest)
  end
  redis.call(
    'HSET', KEYS[1],
    'state', 'expired',
    'failure_code', '',
    'failure_detail', '',
    'failure_retryable', '0'
  )
  redis.call('EXPIRE', KEYS[1], math.ceil(tombstone_ttl))
end
return redis.call('HGETALL', KEYS[1])
"""


_START_SCRIPT = r"""
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if redis.call('HGET', KEYS[1], 'locator') ~= ARGV[2] then return 0 end
if redis.call('HGET', KEYS[1], 'state') ~= 'queued' then return 0 end
local submitted_at = tonumber(redis.call('HGET', KEYS[1], 'submitted_at'))
redis.call(
  'HSET', KEYS[1],
  'state', 'running',
  'started_at', ARGV[3],
  'lease_expires_at', ARGV[4],
  'queue_wait_ms', (tonumber(ARGV[3]) - submitted_at) * 1000
)
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1])
redis.call('HINCRBYFLOAT', KEYS[3], 'queue_wait_ms_total', (tonumber(ARGV[3]) - submitted_at) * 1000)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return 1
"""


_SUCCEED_SCRIPT = r"""
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if redis.call('HGET', KEYS[1], 'locator') ~= ARGV[2] then return 0 end
local state = redis.call('HGET', KEYS[1], 'state')
if state == 'succeeded' then return 1 end
if state ~= 'running' then return 0 end
redis.call(
  'HSET', KEYS[1],
  'state', 'succeeded',
  'completed_at', ARGV[3],
  'result_expires_at', ARGV[4],
  'failure_code', '',
  'failure_detail', '',
  'failure_retryable', '0'
)
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HINCRBY', KEYS[3], 'succeeded_total', 1)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return 1
"""


_FAIL_SCRIPT = r"""
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if redis.call('HGET', KEYS[1], 'locator') ~= ARGV[2] then return 0 end
local state = redis.call('HGET', KEYS[1], 'state')
if state == 'succeeded' or state == 'expired' then return 0 end
redis.call(
  'HSET', KEYS[1],
  'state', 'failed',
  'completed_at', ARGV[3],
  'result_expires_at', ARGV[4],
  'failure_code', ARGV[5],
  'failure_detail', ARGV[6],
  'failure_retryable', ARGV[7]
)
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HINCRBY', KEYS[3], 'failed_total', 1)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[8]))
return 1
"""


_CANCEL_SCRIPT = r"""
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if redis.call('HGET', KEYS[1], 'locator') ~= ARGV[2] then return 0 end
if redis.call('HGET', KEYS[1], 'state') ~= 'queued' then return 0 end
local idempotency_digest = redis.call('HGET', KEYS[1], 'idempotency_digest')
if idempotency_digest and idempotency_digest ~= '' then
  redis.call('DEL', ARGV[3] .. idempotency_digest)
end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('DEL', KEYS[1])
return 1
"""


_OBSERVE_SCRIPT = r"""
local now = tonumber(ARGV[1])
local result_ttl = tonumber(ARGV[2])
local tombstone_ttl = tonumber(ARGV[3])
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now)
for _, prediction_id in ipairs(expired) do
  local job_key = ARGV[4] .. prediction_id
  local state = redis.call('HGET', job_key, 'state')
  if state == 'queued' or state == 'running' then
    local failure_code = 'prediction_worker_lost'
    local failure_detail = 'prediction worker was lost'
    if state == 'queued' then
      failure_code = 'prediction_reservation_expired'
      failure_detail = 'prediction queue reservation expired'
    else
      redis.call('HINCRBY', KEYS[2], 'worker_lost_total', 1)
    end
    redis.call(
      'HSET', job_key,
      'state', 'failed',
      'completed_at', now,
      'result_expires_at', now + result_ttl,
      'failure_code', failure_code,
      'failure_detail', failure_detail,
      'failure_retryable', '0'
    )
    redis.call('EXPIRE', job_key, math.ceil(result_ttl + tombstone_ttl))
    redis.call('HINCRBY', KEYS[2], 'failed_total', 1)
  end
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local queued = 0
local running = 0
local active_ids = redis.call('ZRANGE', KEYS[1], 0, -1)
for _, prediction_id in ipairs(active_ids) do
  local state = redis.call('HGET', ARGV[4] .. prediction_id, 'state')
  if state == 'queued' then queued = queued + 1 end
  if state == 'running' then running = running + 1 end
end
return {
  queued + running,
  queued,
  running,
  tonumber(redis.call('HGET', KEYS[2], 'admitted_total') or '0'),
  tonumber(redis.call('HGET', KEYS[2], 'rejected_total') or '0'),
  tonumber(redis.call('HGET', KEYS[2], 'succeeded_total') or '0'),
  tonumber(redis.call('HGET', KEYS[2], 'failed_total') or '0'),
  tonumber(redis.call('HGET', KEYS[2], 'worker_lost_total') or '0'),
  tonumber(redis.call('HGET', KEYS[2], 'queue_wait_ms_total') or '0')
}
"""


class RedisPredictionStateRepository:
    """Use Redis Lua transitions so web-process admission is atomic."""

    def __init__(
        self,
        client: _RedisClient,
        *,
        capacity: int,
        result_ttl_seconds: float,
        tombstone_ttl_seconds: float,
        key_prefix: str = "vision-model-serving:predictions",
        clock: Callable[[], float] = time,
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if result_ttl_seconds <= 0 or tombstone_ttl_seconds <= 0:
            raise ValueError("result and tombstone TTLs must be positive")
        if not key_prefix:
            raise ValueError("Redis key prefix must not be empty")
        self._client = client
        self._capacity = capacity
        self._result_ttl_seconds = float(result_ttl_seconds)
        self._tombstone_ttl_seconds = float(tombstone_ttl_seconds)
        self._prefix = key_prefix.rstrip(":")
        self._clock = clock

    def admit(
        self,
        *,
        prediction_id: PredictionId,
        locator: str,
        request_fingerprint: str,
        idempotency_digest: str | None,
        submitted_at: float,
        reservation_expires_at: float,
    ) -> Admission:
        record_ttl = ceil(
            max(1.0, reservation_expires_at - submitted_at)
            + self._result_ttl_seconds
            + self._tombstone_ttl_seconds
        )
        response = self._eval(
            _ADMIT_SCRIPT,
            4,
            self._active_key,
            self._job_key(prediction_id),
            self._idempotency_key(idempotency_digest, prediction_id),
            self._metrics_key,
            self._clock(),
            str(prediction_id),
            request_fingerprint,
            locator,
            submitted_at,
            reservation_expires_at,
            self._capacity,
            record_ttl,
            int(idempotency_digest is not None),
            self._job_prefix,
            idempotency_digest or "",
            self._result_ttl_seconds,
            self._tombstone_ttl_seconds,
        )
        if not isinstance(response, (list, tuple)) or len(response) < 2:
            raise GatewayUnavailable("prediction state response is invalid")
        outcome = int(_text(response[0]))
        returned_id = PredictionId(_text(response[1]))
        if outcome == 0:
            raise QueueSaturated("prediction queue capacity is exhausted")
        if outcome == -1:
            raise IdempotencyConflict(
                "idempotency key is already bound to different input"
            )
        if outcome == 2:
            record = self.record(returned_id)
            return Admission(
                PredictionHandle(
                    prediction_id=returned_id,
                    state=record.state,
                    submitted_at=record.submitted_at,
                    expires_at=record.status().expires_at,
                    idempotent_replay=True,
                ),
                record.locator,
            )
        if outcome != 1 or returned_id != prediction_id:
            raise GatewayUnavailable("prediction state response is invalid")
        return Admission(
            PredictionHandle(
                prediction_id=prediction_id,
                state=PredictionJobState.QUEUED,
                submitted_at=submitted_at,
                expires_at=reservation_expires_at,
            ),
            locator,
        )

    def status(self, prediction_id: PredictionId) -> PredictionStatus:
        return self.record(prediction_id).status()

    def record(self, prediction_id: PredictionId) -> JobStateRecord:
        response = self._eval(
            _REFRESH_SCRIPT,
            3,
            self._job_key(prediction_id),
            self._active_key,
            self._metrics_key,
            self._clock(),
            self._result_ttl_seconds,
            self._tombstone_ttl_seconds,
            f"{self._prefix}:idempotency:",
        )
        values = _mapping(response)
        if not values:
            raise PredictionNotFound("prediction ID is unknown or no longer retained")
        try:
            failure_code = values.get("failure_code", "")
            failure = (
                PredictionFailure(
                    code=failure_code,
                    detail=values.get("failure_detail", "prediction failed"),
                    retryable=values.get("failure_retryable", "0") == "1",
                )
                if failure_code
                else None
            )
            return JobStateRecord(
                prediction_id=PredictionId(values["prediction_id"]),
                locator=values["locator"],
                request_fingerprint=values["request_fingerprint"],
                idempotency_digest=values.get("idempotency_digest") or None,
                state=PredictionJobState(values["state"]),
                submitted_at=float(values["submitted_at"]),
                lease_expires_at=float(values["lease_expires_at"]),
                result_expires_at=_optional_float(values.get("result_expires_at")),
                started_at=_optional_float(values.get("started_at")),
                completed_at=_optional_float(values.get("completed_at")),
                failure=failure,
                queue_wait_ms=_optional_float(values.get("queue_wait_ms")),
            )
        except (KeyError, TypeError, ValueError):
            raise GatewayUnavailable("prediction state response is invalid") from None

    def mark_running(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        started_at: float,
        lease_expires_at: float,
    ) -> bool:
        ttl = ceil(
            max(1.0, lease_expires_at - started_at)
            + self._result_ttl_seconds
            + self._tombstone_ttl_seconds
        )
        return self._truthy_eval(
            _START_SCRIPT,
            3,
            self._job_key(prediction_id),
            self._active_key,
            self._metrics_key,
            str(prediction_id),
            locator,
            started_at,
            lease_expires_at,
            ttl,
        )

    def mark_succeeded(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        completed_at: float,
        result_expires_at: float,
    ) -> bool:
        ttl = ceil(
            max(1.0, result_expires_at - completed_at)
            + self._tombstone_ttl_seconds
        )
        return self._truthy_eval(
            _SUCCEED_SCRIPT,
            3,
            self._job_key(prediction_id),
            self._active_key,
            self._metrics_key,
            str(prediction_id),
            locator,
            completed_at,
            result_expires_at,
            ttl,
        )

    def mark_failed(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        completed_at: float,
        failure: PredictionFailure,
    ) -> bool:
        result_expires_at = completed_at + self._result_ttl_seconds
        ttl = ceil(self._result_ttl_seconds + self._tombstone_ttl_seconds)
        return self._truthy_eval(
            _FAIL_SCRIPT,
            3,
            self._job_key(prediction_id),
            self._active_key,
            self._metrics_key,
            str(prediction_id),
            locator,
            completed_at,
            result_expires_at,
            failure.code,
            failure.detail,
            int(failure.retryable),
            ttl,
        )

    def cancel_admission(self, prediction_id: PredictionId, locator: str) -> bool:
        return self._truthy_eval(
            _CANCEL_SCRIPT,
            2,
            self._job_key(prediction_id),
            self._active_key,
            str(prediction_id),
            locator,
            f"{self._prefix}:idempotency:",
        )

    def observations(self) -> GatewayObservations:
        response = self._eval(
            _OBSERVE_SCRIPT,
            2,
            self._active_key,
            self._metrics_key,
            self._clock(),
            self._result_ttl_seconds,
            self._tombstone_ttl_seconds,
            self._job_prefix,
        )
        if not isinstance(response, (list, tuple)) or len(response) != 9:
            raise GatewayUnavailable("prediction state response is invalid")
        try:
            numbers = [float(_text(value)) for value in response]
        except ValueError:
            raise GatewayUnavailable("prediction state response is invalid") from None
        return GatewayObservations(
            active_jobs=int(numbers[0]),
            queued_jobs=int(numbers[1]),
            running_jobs=int(numbers[2]),
            admitted_total=int(numbers[3]),
            rejected_total=int(numbers[4]),
            succeeded_total=int(numbers[5]),
            failed_total=int(numbers[6]),
            worker_lost_total=int(numbers[7]),
            queue_wait_ms_total=numbers[8],
        )

    @property
    def _active_key(self) -> str:
        return f"{self._prefix}:active"

    @property
    def _metrics_key(self) -> str:
        return f"{self._prefix}:metrics"

    @property
    def _job_prefix(self) -> str:
        return f"{self._prefix}:job:"

    def _job_key(self, prediction_id: PredictionId) -> str:
        return f"{self._job_prefix}{prediction_id}"

    def _idempotency_key(
        self,
        digest: str | None,
        prediction_id: PredictionId,
    ) -> str:
        suffix = digest if digest is not None else f"none:{prediction_id}"
        return f"{self._prefix}:idempotency:{suffix}"

    def _eval(self, script: str, key_count: int, *values: object) -> object:
        try:
            return self._client.eval(script, key_count, *values)
        except Exception:
            raise GatewayUnavailable("prediction state store is unavailable") from None

    def _truthy_eval(
        self,
        script: str,
        key_count: int,
        *values: object,
    ) -> bool:
        response = self._eval(script, key_count, *values)
        return int(_text(response)) == 1


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mapping(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {_text(key): _text(item) for key, item in value.items()}
    if not isinstance(value, (list, tuple)) or len(value) % 2:
        return {}
    return {
        _text(value[index]): _text(value[index + 1])
        for index in range(0, len(value), 2)
    }


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in {None, ""} else None
