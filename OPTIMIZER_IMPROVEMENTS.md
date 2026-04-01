# Optimizer Improvements

## What Changed

### 1. **Much Higher Gate Miss Penalty**
- **Before:** 5.0 seconds per missed gate
- **After:** 100.0 seconds per missed gate
- **Effect:** Optimizer now STRONGLY prefers completing all gates over going fast

### 2. **Incomplete Course Penalty**
- **New:** 500.0 second penalty if not all gates completed
- **Effect:** Optimizer won't accept "fast but incomplete" runs

### 3. **Minimum Gates Required**
- **New:** Trial is only valid if at least 10 gates completed
- **Effect:** Filters out garbage runs that crash early

### 4. **Smarter Scoring Logic**
Three-tier system:
1. **Crash/Error:** Score = 9999.0 (worst)
2. **Incomplete course:** Score = lap_time + gate_penalties + INCOMPLETE_PENALTY (very bad)
3. **Complete course:** Score = lap_time + (missed_gates × 100.0) (good)

### 5. **Better Reporting**
- Shows gates completed vs total
- Displays scoring breakdown
- Tracks improvement between trials

### 6. **New Tunable Parameters**
Added dynamic CV window scaling to optimizer search space:
- `sky_mask_near` - sky mask percentage when gate is close (0.20-0.50)
- `sky_mask_far` - sky mask percentage when gate is far (0.05-0.25)
- `contour_min_area_far` - minimum pixel area for distant gates (30-100)

**Result:** Optimizer can now tune CV detection quality for each distance range

---

## How This Helps

### Before (Old Objective)
- Optimizer found: lap_time=13.4s, missed=0, score=13.4s ✓
- But: probably missing gates in practice (incomplete course)
- Problem: Penalty too low to force completeness

### After (New Objective)
- Optimizer enforces: **COMPLETE ALL GATES FIRST**
- Then optimizes lap time among valid complete runs
- Prefers: steady 20s lap with 0 gates missed
- Over: fast 13s lap with 2 gates missed
- Result: **Reliable, complete runs that don't crash**

---

## Running the Improved Optimizer

```bash
cd C:\Users\brend\AI-GP
python optimize_params.py
```

This will:
1. Run 100 trials (1-2 hours)
2. Only keep runs that complete all gates
3. Penalize any missed gates by 100 seconds each
4. Find the fastest **complete** lap time
5. Save best config to `config_best.yaml`

---

## Expected Results

**Good trial** (complete, clean):
```
[TRIAL 23] COMPLETE: 12/12 gates, lap_time=22.3s, missed=0, score=22.3s
[BEST] NEW BEST: 22.3s (improved by 0.5s)
```

**Bad trial** (missing gates):
```
[TRIAL 24] INCOMPLETE: 10/12 gates, lap_time=18.5s, missed=0, score=618.5s
(penalized 500s for not completing + 200s if any missed)
```

**Crash trial**:
```
[TRIAL 25] FAILED: timeout
```

---

## Tips

1. **Let it run to completion** - Early results might look bad, but trials 50+ are usually best
2. **Monitor for patterns** - Watch the output to see what works
3. **Interrupt safely** - Ctrl+C anytime, optimizer will resume from checkpoint
4. **Check results** - `cat config_best.yaml` to see winning parameters

---

## Comparison: V14 vs V16 Optimized

| Metric | V14 (Hardcoded) | V16 (Initial) | V16 (Optimized) |
|---|---|---|---|
| Lap Time | ~24s | 24.6s | ~20-22s (est) |
| Gates Completed | All | All | All (forced) |
| Reliability | Good | Good | Better (enforced) |
| Speed Max | 14 m/s | 14 m/s | Tuned (12-18 m/s) |
| CV Tuning | None | Fixed | **Adaptive by distance** |

---

## Implementation Details

**New scoring function:**
```
if crash:
    score = 9999
elif gates < 10:
    score = 9999 - gates_completed  # Rank by partial completion
elif gates < 12:
    score = lap_time + (12 - gates)*100 + 500  # Heavy incomplete penalty
else:
    score = lap_time + missed*100  # Normal: reward fast complete runs
```

This ensures optimizer:
- ✅ Completes the course
- ✅ Completes as many gates as possible
- ✅ Goes fast given those constraints
- ❌ Never sacrifices completion for speed
