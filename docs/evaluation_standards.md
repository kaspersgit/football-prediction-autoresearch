# Evaluation Standards

## 1. Per-league review (mandatory for every iteration)

An improvement is only trusted if **all three** hold:
1. Total ROI improves (primary metric)
2. Stability improves or is neutral
3. A majority of leagues improve — target 5+/7; accept 4/7 only if total ROI and stability both move clearly in the right direction

A total ROI gain driven by 1–2 outlier leagues while others regress is a red flag for noise.
Venue-specific form is a known permanent failure — do not re-test (failed iters 1, 18, 57).

---

## 2. How much variance is just randomness?

Each bet is a binary lottery: win `(odds − 1)` or lose `1` unit. Std per bet ≈ 1.0–1.5 units.
ROI = average of N noisy outcomes, so its standard error = `std_per_bet / √N`.

| Sample | Approx bets | ROI std error | 95% CI width |
|---|---|---|---|
| 1 match week | ~10 | ±32% | ±64% |
| 1 month | ~50 | ±14% | ±28% |
| 1 league × 2 seasons | ~550 | ±4.3% | ±8.6% |
| Full backtest | ~4025 | ±1.6% | ±3.2% |

**Consequence:** per-week and per-month numbers are almost entirely noise. Per-league numbers over 2 seasons are still very noisy (±8.6% at 95%). The full backtest gives a meaningful signal only when effects are several pp in size.

---

## 3. The stability metric is a t-statistic proxy

`stability = mean_profit / std_profit`

The true t-statistic for "ROI ≠ 0" is:

```
t = stability × √(N_bets)
```

Need `|t| > 2` for 95% confidence.

Current state example: `t = −0.009 × √4025 ≈ −0.57` — far from significant.

To detect a genuine edge at our current ROI and stability, we would need ≈ 49,000 bets.
This means **our backtest ROI of −1.33% is statistically indistinguishable from zero.**
We have not proven edge yet; iterations are moving the distribution in the right direction.

To reach significance at threshold=0.0 we need:
- Either ROI/stability to improve substantially (primary goal)
- Or a longer test period (more seasons, more leagues — secondary, slower path)

---

## 4. What to trust vs ignore

| Observation | How to treat it |
|---|---|
| Total ROI changed < 1 pp | Noise — ignore |
| Total ROI changed 2–4 pp | Weak signal — weight by cross-league direction count |
| Total ROI changed > 5 pp | Meaningful — still verify per-league |
| 5+/7 leagues moved same direction | Treat as real signal even if total change is small |
| Single league moved > 8 pp | Real signal for that league specifically |
| Per-week loss/win streak | Completely expected — ignore unless it persists for months |

---

## 5. Data-reduction caution

Features that require a per-context warm-up (venue-split, per-opponent, etc.) reduce the number of usable rows. This is a known failure mode:

- Iter 1 / 18 / 57: venue-split form — bets dropped ~10%, all metrics worsened every time.

**Rule:** before adding a data-reducing feature, count the expected row loss explicitly.
If the per-league improvement disappears when you account for the smaller sample, the feature adds no signal.

---

## 6. Cross-iteration consistency is the real signal

No single iteration result is trustworthy in isolation. What is trustworthy:

- **Same direction across many independent iterations** — if 8 different changes all nudge total ROI upward, that's 8 correlated confirmations
- **Cross-league count** — 6/7 leagues improving is stronger evidence than the total ROI number, because it's 6 semi-independent observations
- **Stability improving alongside ROI** — harder to fake with random luck than ROI alone
- **Known-bad list respected** — if a feature failed 2–3× with clear mechanistic explanation, do not re-test it
