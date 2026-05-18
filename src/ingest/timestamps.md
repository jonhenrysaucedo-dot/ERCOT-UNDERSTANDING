# ERCOT Timestamp Conventions

## Canonical column: `interval_start_utc`

All processed data uses `interval_start_utc` (type `datetime64[ns, UTC]`).

## ERCOT source timestamps

ERCOT publishes timestamps in **Central Time (CT)**, which is:
- UTC−6 in standard time (CST, Nov → Mar)
- UTC−5 in daylight time (CDT, Mar → Nov)

ERCOT uses **HourEnding** notation: HourEnding=1 means the interval 00:00–01:00 CT,
which we convert to `interval_start_utc = 00:00 CT` (offset adjusted for DST).

Conversion rule at ingest boundary:
```python
# hour_ending is 1-indexed (1..24); 25 on fall-back hour
ts_ct_naive = date + timedelta(hours=hour_ending - 1)
ts_ct_aware = ERCOT_TZ.localize(ts_ct_naive, is_dst=None)  # ambiguous → raise
ts_utc = ts_ct_aware.astimezone(pytz.utc)
interval_start_utc = ts_utc  # store this column
```

## DST edge cases

### Spring-forward (2nd Sunday March)
Clock jumps 02:00 → 03:00 CT. HourEnding=3 is skipped. ERCOT marks the missing
hour with a `RepeatedHourFlag` value of `N` for the surrounding hours; HourEnding=3
simply does not appear. At ingest, detect and log the gap; do NOT fill it.

### Fall-back (1st Sunday November)
Clock falls 02:00 → 01:00 CT. HourEnding=2 appears twice. ERCOT sets
`RepeatedHourFlag='Y'` on the second occurrence (the standard-time 01:00–02:00 CT).
At ingest, the first occurrence is CDT (UTC−5) and the second is CST (UTC−6).
We use the `RepeatedHourFlag` column to disambiguate:
- First `HourEnding=2`: `is_dst=True` → 07:00 UTC
- Second `HourEnding=2` (`RepeatedHourFlag='Y'`): `is_dst=False` → 08:00 UTC

If `RepeatedHourFlag` is absent (older reports), log a warning and assume the first
occurrence is CDT. Do NOT silently drop or merge the duplicate.

## Timezone objects

```python
import pytz
ERCOT_TZ = pytz.timezone("America/Chicago")
```

Never use `dateutil.tz` or `zoneinfo` for ERCOT timestamps — stick to `pytz` for
consistent `localize(is_dst=None/True/False)` semantics.
