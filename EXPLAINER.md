# EXPLAINER — how this system works, why it is built this way, and what to fix next

A complete walkthrough of *Rupee-Optimal Risk*. The [README](README.md) is the submission; this document is the engineering rationale behind it. Read this if you want to understand or extend the code, or if you are deciding what to demo.

**Contents**

1. [The core idea in one page](#1-the-core-idea-in-one-page)
2. [The cost model — where the maths comes from](#2-the-cost-model--where-the-maths-comes-from)
3. [Data: what IEEE-CIS is and what it isn't](#3-data-what-ieee-cis-is-and-what-it-isnt)
4. [The evaluation protocol and why each fold exists](#4-the-evaluation-protocol-and-why-each-fold-exists)
5. [The baseline ladder](#5-the-baseline-ladder)
6. [The model](#6-the-model)
7. [Calibration — the load-bearing step](#7-calibration--the-load-bearing-step)
8. [The decision policy](#8-the-decision-policy)
9. [The service](#9-the-service)
10. [Results, including the two failures](#10-results-including-the-two-failures)
11. [File-by-file reference](#11-file-by-file-reference)
12. [Can this be a workable demo?](#12-can-this-be-a-workable-demo)
13. [What can be improved](#13-what-can-be-improved)
14. [Known weaknesses, ranked by how much they'd embarrass you](#14-known-weaknesses-ranked-by-how-much-theyd-embarrass-you)

---

## 1. The core idea in one page

Almost every fraud-detection project reports accuracy, or AUC, or F1. None of those are what a merchant pays. A merchant pays rupees, in two distinct ways:

- **A missed fraud (false negative)** → chargeback: the transaction value is clawed back, plus a flat scheme fee, plus ops time.
- **A blocked legitimate customer (false positive)** → lost *margin* on the order, plus some probability the customer churns, plus a support contact.

These two costs are **wildly asymmetric** and they **scale differently with order value**. That has a consequence most projects miss:

> The threshold that maximises F1 is not the threshold that minimises money lost. They are different numbers, and the gap between them is real money.

On our test window that gap is **₹112,750 per 10,000 transactions**. That single number is the project.

Everything else in the codebase exists to make that number *believable*:

| Component | Exists so that... |
|---|---|
| Temporal split | ...the number isn't inflated by the model seeing the future |
| Val-A / val-B separation | ...the threshold isn't tuned on the same rows that fitted the calibrator |
| Calibration + ECE | ...`p` in the cost arithmetic actually means "probability of fraud" |
| Baseline ladder | ...we know how much of the number was free |
| Bootstrap CIs | ...we know whether the number is distinguishable from noise |
| Frozen thresholds | ...we report what we'd have earned, not what we'd have earned with hindsight |

### The mental model

```
                    ┌──────────────────────────────────────┐
   transaction ───▶ │ model  →  raw score                  │
                    │ calibrator  →  p = P(fraud | x)      │  ← must be TRUE probability
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │ cost model:                          │
                    │   approve costs  p · c_FN(amount)    │
                    │   block costs    (1-p) · c_FP(amount)│
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │ policy: approve / review / block     │  ← bounded, reversible
                    └──────────────┬───────────────────────┘
                                   │
                          audit log (immutable)
```

The chain breaks if any link is weak. An uncalibrated `p` makes the cost arithmetic meaningless. A cost model with wrong constants moves the threshold. A threshold picked on test is a fantasy. This is why the project spends more effort on protocol than on modelling.

---

## 2. The cost model — where the maths comes from

Everything lives in [`costs.yaml`](costs.yaml), versioned, and the version is stamped into every audit row so a historical decision can be replayed against the cost model that was in force.

```yaml
false_negative:                       # fraud approved
  chargeback_amount_multiplier: 1.0   # you eat the full transaction value
  chargeback_fee_inr: 1000.0          # flat scheme/gateway dispute fee
  ops_handling_inr: 200.0             # analyst time on the dispute

false_positive:                       # legitimate customer blocked
  margin_rate: 0.12                   # you lose MARGIN, not revenue
  churn_probability: 0.05
  customer_ltv_inr: 3000.0
  support_contact_inr: 100.0
```

Which gives:

```
c_FN(a) = 1.00·a + 1200          # ₹1,000 fee + ₹200 ops
c_FP(a) = 0.12·a + 250           # ₹150 expected churn + ₹100 support
```

### Why margin and not revenue

This is the single most commonly botched line in cost-sensitive fraud work. If you block a legitimate ₹10,000 order, you did not lose ₹10,000 — you lost the *margin* you would have made, because you also didn't ship the goods. At a 12% margin that is ₹1,200, not ₹10,000.

Getting this wrong inflates FP cost roughly **eightfold**, which pushes the optimal threshold far too high, which makes the system block far too little fraud. A project that "accounts for false-positive cost" but uses order value is arguably worse than one that ignores cost entirely, because it is confidently wrong.

### Deriving the optimal threshold

For a single transaction with fraud probability `p` and amount `a`:

```
E[cost | approve] = p · c_FN(a)
E[cost | block]   = (1 - p) · c_FP(a)
```

Block when blocking is cheaper:

```
p · c_FN(a) > (1 - p) · c_FP(a)
p · c_FN(a) > c_FP(a) - p · c_FP(a)
p · (c_FN(a) + c_FP(a)) > c_FP(a)

              c_FP(a)
p  >  ───────────────────────  =  t*(a)
       c_FP(a) + c_FN(a)
```

Two things follow.

**First, with constant costs there is a closed form.** `t* = c_FP / (c_FP + c_FN)`. We use this as a unit test ([`tests/test_costs.py::test_swept_optimum_matches_closed_form_at_constant_costs`](tests/test_costs.py)) — we neutralise the amount-dependent terms, sweep the threshold numerically on simulated data whose scores *are* the true probabilities, and assert the empirical optimum lands on the analytic one. That test is what proves the sweep minimises what we think it minimises. A sign error in the money code would otherwise be invisible.

**Second, because our costs depend on `a`, no single threshold is optimal.** `t*(a)` is a curve. Plugging in:

| Order value | c_FN | c_FP | t*(a) |
|---|---|---|---|
| ₹500 | ₹1,700 | ₹310 | 0.154 |
| ₹5,000 | ₹6,200 | ₹850 | 0.121 |
| ₹50,000 | ₹51,200 | ₹6,250 | 0.109 |

### A subtlety worth knowing

The curve's *direction* is not a law of fraud. Differentiating `t*(a)`, it decreases with amount exactly when:

```
fixed_fp  >  margin_rate × fixed_fn
   250    >     0.12 × 1200 = 144       ✓ (so ours falls)
```

Raise the flat chargeback fee enough and the inequality flips, and it becomes optimal to be *more* permissive on large orders. This is encoded in `CostModel.threshold_limits()` and tested in both directions. Presenting "block big orders on weaker evidence" as universal would be wrong.

**Why this matters for the project:** we predicted this curve would beat a single global threshold. It didn't. See [§10](#10-results-including-the-two-failures).

---

## 3. Data: what IEEE-CIS is and what it isn't

**IEEE-CIS Fraud Detection** (Vesta, 2019). 590,540 transactions, 3.4990% fraud, 434 columns after joining identity.

### What the columns actually are

| Family | Count | What it is | Interpretable? |
|---|---|---|---|
| `TransactionAmt`, `ProductCD`, `card4`, `card6` | 4 | amount, product code, card network, debit/credit | **yes** |
| `card1`, `card2`, `card3`, `card5` | 4 | masked bank/card attributes | no |
| `addr1`, `addr2`, `dist1`, `dist2` | 4 | billing address/country, distances | partly |
| `P_emaildomain`, `R_emaildomain` | 2 | email **domain** only, never the address | yes |
| `C1`–`C14` | 14 | **counting features** — "how many addresses are associated with this card". Definitions masked. | partly |
| `D1`–`D15` | 15 | timedeltas — "days since previous transaction" | partly |
| `M1`–`M9` | 9 | match flags — "name on card matches address" | partly |
| `V1`–`V339` | **339** | Vesta engineered features: ranking, counting, entity relations. Meanings masked. | **no** |
| `id_01`–`id_38`, `DeviceType`, `DeviceInfo` | 40 | device/browser/OS. **Present on only 23.8% of rows.** | partly |

**79% of the feature space is anonymised V-columns.** This has a direct consequence for the explainability story: SHAP will frequently name a `V` column as a top contributor, and the honest reason code is "aggregate risk feature (V258) elevated" — which is not a satisfying explanation. We say so rather than inventing a narrative.

### The three things that aren't there

These killed three features of the original plan and are worth stating plainly, because a judge who knows this dataset will check:

1. **No merchant identifier.** Per-merchant thresholds are impossible. `ProductCD` (5 masked values) is the nearest proxy.
2. **No card / customer / device / IP identifier.** `card1` has ~13,553 distinct values over 590k rows — a bank/BIN bucket, roughly 44 transactions per value. It is not a card. The top Kaggle solutions construct a UID (`card1_addr1 + floor(day − D1)`), but that is a *community invention*, not a provided field — and its predictive power partly comes from the label-propagation artifact described below. **So no true velocity features.**
3. **No payment method, no currency.** Amounts are USD. There is no UPI/cards/netbanking split. Any UPI-specific claim from this dataset would be fabricated.

### The label definition is itself an artifact

The competition host (Lynn@Vesta) stated:

> "The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too. If none of above is reported and found beyond 120 days, then we define as legit transaction (isFraud=0). However, in real world fraudulent activity might not be reported..."

Two implications the code and README both flag:

- **Label propagation.** Once an account is compromised, *all subsequent transactions on it* are labelled fraud. So a model that identifies the account is partly predicting the labelling rule, not the fraud. This is why UID features are so powerful on this dataset and why we treat that power sceptically.
- **Unreported fraud sits in the negative class.** Our recall is measured against *reported chargebacks*, not against fraud. True recall is unknowable here.

### Currency

`TransactionAmt` is USD. We re-denominate at a single declared ₹88/USD in `costs.yaml`. The method transfers to an Indian merchant unchanged; the absolute rupee figures are illustrative. Blurring this would be the fastest way to lose credibility with a judge who knows the dataset.

### Memory handling

`train_transaction.csv` is 683 MB and a naive `float64` load costs 2–3 GB. [`src/data.py`](src/data.py) downcasts floats to `float32` and integers to their narrowest type, then caches to Parquet so subsequent runs load in seconds.

---

## 4. The evaluation protocol and why each fold exists

This is where the project spends its credibility budget.

### Temporal split, never random

Fraud is non-stationary — patterns drift as fraudsters adapt. A random split lets the model train on February and test on January, which is not a thing that can happen in production. It inflates every metric. Published work on this dataset shows the difference starkly: a random 80/20 split reports PR-AUC 0.83–0.89 and ROC-AUC 0.99+; a temporal split reports PR-AUC 0.49–0.65. **The random-split numbers are fiction.**

`assert_no_temporal_leakage()` enforces that every split's time window strictly follows the previous one, and it runs at the top of every pipeline script.

### Four folds, one job each

| Fold | Rows | Fraud | Days | Its single job |
|---|---|---|---|---|
| train (minus tail) | ~318,900 | 3.38% | 1–91 | fit the trees |
| train tail | ~35,400 | — | 91–101 | early stopping |
| **val-A** | 59,054 | **4.32%** | 101–121 | fit the calibrator |
| **val-B** | 59,054 | 3.49% | 121–141 | choose the threshold |
| test | 118,108 | 3.44% | 141–183 | read once |

**Why validation is split in two.** The plan originally said "fit isotonic on validation" and "tune the threshold on validation". Doing both on the same rows means the threshold is tuned against a calibrator that has already partly memorised those rows — so the threshold inherits the overfit and looks better than it is. Splitting costs nothing and removes the coupling.

**Why early stopping uses the tail of train, not val-A.** If early stopping ran on val-A, then val-A would be doing two jobs: model selection *and* calibration. The calibrator would be fitted on data the model was tuned to fit well. Carving the early-stopping fold out of train keeps the chain clean.

**Why val-A calibrates and val-B thresholds (in that order).** Val-B is temporally *later*, so it sits closest to the test window the frozen threshold will actually face. This is decided on first principles, not by looking at which fold gives a better answer.

### The drift finding we did not paper over

Val-A has a **4.32% fraud rate** against 3.38% in train and 3.44% in test — a ~28% relative spike in exactly the window used to fit the calibrator.

We left it. Choosing the calibration fold by checking which one resembles test is peeking, and it would invalidate the whole protocol. It is also a realistic production condition: you always calibrate on the past, and the past is sometimes unrepresentative. The ECE numbers in §7 are what that produced.

### Frozen thresholds vs the oracle

The threshold is chosen on val-B and **frozen**. Test is scored at that frozen value.

The alternative — sweep the threshold on test, find the minimum, report that minimum — is an *oracle* number that assumes you could see the future. It is the exact species of inflated claim this project exists to argue against, and it is easy to do by accident because the code looks almost identical.

We report both:

| | Threshold | Cost / 10k |
|---|---|---|
| Frozen from val-B | 0.1300 | ₹3,295,395 |
| Test oracle (peeking) | 0.1200 | ₹3,287,939 |
| **Price of not seeing the future** | | **₹7,456** |

That ₹7,456 gap being *small* is itself the reassuring result: it means val-B is a good proxy for test, and the threshold is stable.

---

## 5. The baseline ladder

**Why it exists:** to answer "how much of your headline number was free?" Most submissions report a model metric with nothing to compare against. A judge cannot tell whether 0.498 PR-AUC is good without knowing what a one-line rule achieves.

Every rung is scored on the same test window under the same protocol the model faces — threshold tuned on val-B, frozen, applied to test. That fairness matters: tuning the model's threshold properly while letting baselines use a default would be stacking the deck.

| Baseline | Uses | PR-AUC | Cost / 10k | Point |
|---|---|---|---|---|
| Never fraud | nothing | — | ₹4,957,414 | **96.56% accuracy, catches zero fraud** |
| Block everything | nothing | — | ₹16,391,839 | the other extreme is 3.3× worse than doing nothing |
| Random at base rate | nothing | 0.035 | ₹5,328,749 | worse than doing nothing |
| Amount > threshold | 1 feature | 0.037 | ₹4,962,233 | tuning picked "block nothing" — amount alone is useless |
| **Count rule (C12 > 3)** | 1 feature | 0.161 | **₹4,669,678** | the honest foil |
| Logistic regression | 5 features | 0.209 | ₹4,838,403 | **loses on money despite winning on PR-AUC** |
| LightGBM | full | **0.498** | **₹3,295,395** | |

### The most important row-pair in the project

The count rule **loses** to logistic regression on PR-AUC (0.161 vs 0.209) and **beats** it on money (₹4.67M vs ₹4.84M per 10k).

Why: the count rule blocks 1.85% of traffic at 33.6% precision and 18.1% recall. LR blocks 0.76% at 57.6% precision but only 12.7% recall. Since `c_FN ≈ a + 1200` and `c_FP ≈ 0.12a + 250`, a missed fraud costs roughly **8× more** than a false block. Recall is worth far more than precision here — so the rule with worse ranking and worse precision wins on cash.

**This demonstrates the entire thesis before a model is involved.** If you present one table from this project, present this one.

### A substitution we had to make, and disclose

The plan wanted a velocity rule (">N transactions per card per hour") as the domain-rule rung. IEEE-CIS has no card identifier, so it isn't computable. We substituted the strongest single `C`-column, selected on val-B. It plays the same role — a hand-written one-feature count rule — but **the counting was done by Vesta, not by us**, and the definitions are masked. Presenting `C12` as "our velocity feature" would be dishonest.

---

## 6. The model

**LightGBM.** On tabular fraud data, gradient-boosted trees beat neural networks, train in minutes on a laptop, handle missing values natively (important when 76% of rows lack identity features), handle categoricals natively, and give exact tree SHAP for free. Reaching for a transformer here signals inexperience, not sophistication.

```python
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",   # optimise the metric we report
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_child_samples": 100,        # guards against leaves memorising rare fraud
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "lambda_l2": 1.0,
    "seed": 42,
    # Deliberately absent: scale_pos_weight / is_unbalance
}
```

### Why there is no class weighting — the correction that matters

Standard advice says: use `scale_pos_weight` *instead of* SMOTE, because resampling destroys probability calibration.

**The first half is right and the second half is wrong.** `scale_pos_weight` reweights the loss, which shifts the model's implied base rate — its outputs no longer estimate `P(fraud | x)` for the true class balance. That breaks calibration just as surely as resampling does. It is a *less obvious* violation, which makes it more dangerous.

The correct rule is: **any prior-shifting technique requires recalibration afterwards.** Since our entire rupee argument depends on `p` meaning what it says, the cleanest path is to train unweighted with enough trees and let isotonic regression do the calibration explicitly. GBDTs handle a 3.5% positive rate fine given sufficient boosting rounds.

### Results

| Metric | Test | Note |
|---|---|---|
| PR-AUC | **0.4981** | pre-registered expectation 0.50–0.65 |
| ROC-AUC | 0.8825 | reported only for leaderboard comparison |
| Recall @ 0.5% FPR | 0.3632 | the operationally meaningful one |
| Best iteration | 556 | early stopping on train tail |

### Why PR-AUC and not ROC-AUC

At a 3.4% base rate, ROC-AUC is dominated by the vast negative class. A model can score 0.88 ROC-AUC while its top-ranked predictions are mostly false positives, because the false-positive *rate* barely moves when the denominator is 96.6% of the data. Average precision tracks what a review analyst actually experiences.

### Pre-registration as a leakage guard

We fixed the expected range (0.50–0.65) and the alarm threshold (>0.75) **before training**, based on published temporal-split results. Landing at 0.498 is a *pass*: it is consistent with having no UID/velocity features, which is where top solutions get most of their lift. Had we scored 0.85, the correct response would have been to hunt for leakage, not to celebrate. `run_model.py` prints the alarm automatically.

---

## 7. Calibration — the load-bearing step

**Why it is not optional here.** The policy compares `p · c_FN(a)` against `(1−p) · c_FP(a)`. If the model outputs 0.9 for transactions that are actually fraudulent 40% of the time, every rupee figure downstream is wrong. Calibration is what converts a *ranking* into a *probability*, and the cost model needs a probability.

Most submissions never check this. Many SMOTE first and never notice their probabilities have become meaningless.

**Method:** isotonic regression fitted on val-A only. Isotonic (rather than Platt) because we have ~2,550 positives in val-A — enough to fit a flexible monotone mapping without severe overfitting. On a much smaller positive count, Platt scaling would be the safer choice.

**Result:**

| | ECE (test) |
|---|---|
| Before | 0.01360 |
| After | **0.00408** |

A 3.3× improvement, achieved despite the calibration fold having a 4.32% base rate against test's 3.44%.

### An implementation detail worth knowing

`expected_calibration_error()` uses **quantile bins, not equal-width bins**. With a 3.4% base rate, virtually every prediction falls into the first equal-width bin, and the metric goes blind — it would report a flatteringly small number regardless of actual calibration quality. Quantile bins put equal mass in each bin so the metric can actually see.

---

## 8. The decision policy

Binary approve/block is what everyone builds. Three outcomes is better:

```
score < 0.0652            →  AUTO-APPROVE
0.0652 ≤ score < 0.1852   →  MANUAL REVIEW    (5.00% of traffic, capped)
score ≥ 0.1852            →  AUTO-BLOCK
```

**How the band is sized.** It straddles the cost-optimal cut (0.130) and widens outward — taking the transactions where the model is closest to indifferent — until the 5% operational ceiling binds. The band contains **9.31% fraud against a 3.44% base rate**, so it is genuinely selecting ambiguous cases rather than padding a queue.

**Why review rate is a first-class metric.** A model that routes 30% of traffic to human review is unusable at any precision — there aren't enough analysts. Reporting precision without reporting review rate hides that failure mode.

**Note the deployed block threshold (0.1852) differs from the cost-optimal threshold (0.1300).** Transactions between the two are escalated rather than auto-blocked. This trades a small amount of expected cost for reversibility on the least certain decisions — which is the point of having a review band at all.

**Bounded actions.** The service never moves money. Both `BLOCK` and `REVIEW` are reversible. Every decision is appealable through `POST /v1/appeal/{id}`, and an appeal is recorded as labelled feedback — never a deletion. The blast radius of a wrong call is one payment, recoverable.

---

## 9. The service

[`src/service.py`](src/service.py) — FastAPI, ~310 lines.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | Score a transaction, return a bounded decision |
| `GET /v1/health` | Reports degraded state honestly, including load errors |
| `GET /v1/audit/{id}` | Every decision recorded for a transaction |
| `POST /v1/appeal/{id}` | Overturn a decision; logged, never deleted |

### Schema pinning — the train/serve skew guard

This one caused a real bug during the build and is worth understanding.

Training converts 31 object columns to pandas `category` dtype with specific category sets. The service builds a one-row frame from arbitrary JSON. If it infers dtypes from whatever the caller happened to send, the categorical set won't match and **LightGBM rejects the frame outright**:

```
ValueError: train and valid dataset categorical_feature do not match.
```

The fix is `models/schema.pkl` — the exact feature order and the exact 31 categorical dtypes, built once by `build_schema.py` and loaded at startup. Numeric columns are cast to `float64` rather than to training's narrow int types, because an absent field must remain `NaN` and `NaN` has no integer representation.

Unseen category values map to `NaN`, which is correct: a card network the model never saw is *missing information*, not a new level to invent at inference time.

**If the schema fails to load, the engine sets `self.fitted = None` and serves from the fallback.** Serving a model without its schema is worse than serving the rule.

### Graceful degradation

If the model or schema fails to load, or scoring throws at request time, the service falls back to the deterministic `C12 > 3` rule and sets `degraded: true`.

The two wrong ways to fail:

- **Fail open** (approve everything) → unbounded fraud exposure.
- **Fail closed** (block everything) → ₹16.4M per 10k, which the cost model shows is **3.3× worse than doing nothing at all**.

The fallback rule was selected by the baseline ladder and costs ₹4.67M per 10k — less than approving everything. So degraded mode is *genuinely safe*, not a token gesture. That claim is measured, not asserted.

**This path is tested.** `tests/test_service.py` points `MODEL_PATH` at a nonexistent file and asserts the service still starts, still returns correct `APPROVE`/`BLOCK` decisions, still marks them degraded, and still audits them.

### Audit log

Append-only SQLite. Every row carries transaction id, amount, score, threshold used, decision, reason codes, model version, **cost-model version**, degraded flag, latency, and any overturn. Stamping the cost-model version is what makes a historical decision replayable — if the constants change, you can still reconstruct why a decision was made in March.

### Merchant-facing vs internal explanations

The customer sees: *"Payment could not be completed. Please contact support to appeal."*

The internal log sees: *"velocity/count signal (C1) elevated / aggregate risk feature (V258) elevated / time-since-previous-activity signal (D14) unusual."*

Publishing the exact rule set teaches fraudsters what to avoid. A test asserts the merchant-facing string never contains internal feature names.

### Latency

1,000 sequential requests replaying real test-window transactions, **including** SHAP reason codes and the audit write:

| p50 | p95 | p99 | max |
|---|---|---|---|
| 56.7 ms | **75.6 ms** | 89.7 ms | 213.3 ms |

Budget was p95 < 100 ms. Met, but with less headroom than a GBDT deserves — see [§13](#13-what-can-be-improved) for where the time actually goes.

---

## 10. Results, including the two failures

### What worked

| Claim | Result | 95% CI | Verdict |
|---|---|---|---|
| Optimising F1 instead of cost | ₹112,750 / 10k | ₹34,904 – ₹200,873 | **significant** |
| Model vs best simple baseline | ₹1,374,283 / 10k | — | significant |
| Calibration (ECE) | 0.0136 → 0.0041 | — | 3.3× better |

The model captures **29.4%** of the headroom between the best simple baseline and a perfect oracle.

### Failure 1: amount-dependent thresholds — no effect

We predicted `t*(a)` would beat a single global threshold. **Measured: −₹1,830 / 10k, 95% CI [−₹76,937, +₹86,062]. The interval spans zero.**

Two reasons:

1. With these cost constants `t*(a)` only spans **0.107–0.172**, and the global optimum (0.130) already sits inside that band. There is little room for adaptivity to matter.
2. More interestingly — **the model is weakest exactly where the money is**:

| Amount band | PR-AUC | Cost / 10k |
|---|---|---|
| < $25 | 0.623 | ₹798k |
| $25–100 | 0.489 | ₹1.04M |
| $100–250 | 0.388 | ₹2.72M |
| $250–1k | 0.444 | ₹14.9M |
| **> $1k** | **0.120** | **₹37.5M** |

An amount-dependent threshold leans hardest on probabilities precisely where they are least trustworthy.

**A note on how this was nearly reported wrongly.** The first version of the unit test compared `t*(a)` against a global threshold *tuned on the same sample it was evaluated on*. The global rule "won" — because it was an oracle fit to that sample's noise, while the adaptive rule was fitted to nothing. The comparison had to move to held-out data. This is the same trap as the test-oracle threshold, and it is easy to fall into even when you are actively writing a project about it.

### Failure 2: per-band calibration — better on validation, worse on test

The diagnosis above suggests calibrating within amount bands. It looked like it worked:

| Configuration | val-B cost | test cost |
|---|---|---|
| Global calibration + global threshold | ₹3,221,039 | **₹3,295,395** |
| Global calibration + t*(a) | ₹3,225,012 | ₹3,297,224 |
| **Per-band calibration + global threshold** | **₹3,145,724** ← best on val-B | ₹3,510,104 |
| Per-band calibration + t*(a) | ₹3,163,586 | ₹3,213,021 |

Selecting honestly — on val-B, before looking at test — picks per-band calibration. It **improved validation by ₹75,314 / 10k and degraded test by ₹214,710 / 10k** (CI ₹66,366 – ₹345,741).

Thin-band overfitting explains part of it: the > $1k band has only **36 fraud cases** in val-A to fit an isotonic curve on. Requiring 200+ positives shrinks the damage to ₹67,191 / 10k with a CI spanning zero — no longer harmful, still not a gain.

**Note the trap in the bottom-right cell.** Per-band + `t*(a)` costs ₹3,213,021 on test — the best cell in the table, ₹82,374 better than the shipped configuration. We do not claim it, because our own protocol never selected it. Reporting it as a win would require having looked at test to choose it.

**There is also a subtler trap we avoided.** An early version compared per-band + `t*(a)` against per-band + *global threshold* and got "₹297,083 saved, significant." But per-band calibration makes the global-threshold policy *worse*, so that comparison measures against a baseline we had handicapped ourselves. Choosing a weak baseline to make a number look good is precisely what the baseline ladder exists to prevent.

**We ship the simple configuration.**

---

## 11. File-by-file reference

| File | Lines | What it does | Why it exists |
|---|---|---|---|
| [`costs.yaml`](costs.yaml) | 45 | Every rupee assumption, versioned | One auditable place; version stamped into audit rows |
| [`src/costs.py`](src/costs.py) | 220 | `CostModel`, `t*(a)`, threshold sweep, bootstrap CIs | The heart — all money math |
| [`src/data.py`](src/data.py) | 200 | Load, merge, downcast, temporal split, leakage assertions | Splits are the credibility budget |
| [`src/evaluate.py`](src/evaluate.py) | 105 | PR-AUC, ECE (quantile-binned), reliability, recall@FPR | Metrics that survive 3.4% imbalance |
| [`src/baselines.py`](src/baselines.py) | 175 | The six-rung ladder | Answers "how much was free?" |
| [`src/model.py`](src/model.py) | 128 | LightGBM + isotonic, fold discipline | Model, plus the no-class-weighting rationale |
| [`src/service.py`](src/service.py) | 310 | FastAPI, audit log, fallback, schema pinning | Bounded, explainable, degradable |
| `build_schema.py` | 40 | Pins training dtypes to `models/schema.pkl` | Prevents train/serve skew |
| `run_baselines.py` | 60 | Runs the ladder | |
| `run_model.py` | 289 | Train → calibrate → threshold → cost curve → segments | The main pipeline |
| `run_band_calibration.py` | 130 | Per-band diagnosis | Failure 2, diagnosis half |
| `run_configuration_choice.py` | 165 | Honest configuration selection on val-B | Failure 2, verdict half |
| `run_demo.py` | 120 | Replay real transactions + latency benchmark | The demo and the p95 number |
| `run_figures.py` | 155 | Five figures | README and video |
| [`tests/test_costs.py`](tests/test_costs.py) | 200 | 14 tests on the money math | A sign error here is invisible otherwise |
| [`tests/test_service.py`](tests/test_service.py) | 120 | 8 tests incl. degraded mode | Proves the fallback rather than claiming it |

### Execution order

```
src/data.py  →  run_baselines.py  →  run_model.py  →  build_schema.py
                                          ↓
              run_band_calibration.py → run_configuration_choice.py
                                          ↓
                          run_figures.py    run_demo.py
```

---

## 12. Can this be a workable demo?

**Yes — with one real constraint you must design around.**

### What works right now

Verified against a live `uvicorn` server (not just the test client):

```bash
.venv/bin/uvicorn src.service:app --port 8000
```

- `GET /v1/health` → `{"status":"ok","model_loaded":true,...}`
- `POST /v1/score` → full decision with reason codes in ~50–80 ms
- `GET /v1/audit/{id}` → the logged row, with both version stamps
- `POST /v1/appeal/{id}` → overturn recorded, original preserved
- Kill the model file, restart → serves from the fallback rule, `degraded: true`, still audits

`run_demo.py` replays 1,000 real test transactions and prints a live decision stream with all three outcomes plus the latency distribution. **That is your demo.**

### The constraint: you cannot hand-author a convincing fraudulent transaction

This is the important finding. We measured how the score responds to payload completeness, using a real fraud transaction that the full pipeline blocks at 0.9000:

| Payload | Features sent | Score | Decision |
|---|---|---|---|
| Interpretable only (what a human can write) | 6 | 0.2067 | BLOCK |
| + `C` counting features | 20 | 0.7831 | BLOCK |
| + `D` time-delta features | 30 | **0.9000** | BLOCK |
| All 431 features | 333 | 0.9000 | BLOCK |

Good news: **~30 real features reproduce the full score.** You do not need all 431, and the 339 anonymised V-columns add nothing beyond the C and D columns for this row.

Bad news: those values must be *real*. A hand-crafted "card testing" payload with invented counts (`C1=45, C12=28, C13=60`) scored **0.0246 → APPROVE**. The model keys on the *joint pattern* across correlated features, not on individual large numbers. Inventing plausible-looking values produces an out-of-distribution row, and the model — correctly — does not recognise it as anything.

### What this means for your video

**Do this:** replay real test-window rows through the live service. `run_demo.py` already selects examples that produce APPROVE, REVIEW and BLOCK, and prints the reason codes and audit entries for each. Show the terminal, or drive the same rows through `curl` for a more API-flavoured demo.

**Don't do this:** type a "suspicious-looking" transaction into a form live and expect it to be blocked. It won't be, and if a judge asks you to try one, the honest answer is the table above.

**The demo you actually have, in order:**

1. `GET /v1/health` — model loaded, versions shown
2. `POST /v1/score` on a real legit row → APPROVE with reason codes
3. `POST /v1/score` on a real ambiguous row → REVIEW, and note the 5% queue cap
4. `POST /v1/score` on a real fraud row → BLOCK at 0.90 with three reason codes
5. `GET /v1/audit/{id}` — show the immutable row with both version stamps
6. `POST /v1/appeal/{id}` — overturn it, show the original is preserved
7. `mv models/fitted.pkl /tmp && restart` → `degraded: true`, still decides, still audits
8. Then the figures: cost curve, baseline ladder, the two negative results

Step 7 is the one nobody else will have.

### What's missing for a polished demo

- **No web UI.** Everything is terminal or `curl`. A dashboard was cut for time. For a 5-minute video this is survivable — arguably the terminal reads as more credible — but it is less visually engaging than a chart-filled dashboard.
- **No live stream.** `run_demo.py` replays in a batch and prints; it doesn't stream at a realistic transaction rate. Adding a `--rate` flag with a sleep would take ~20 minutes and would look considerably better on video.
- **Cold start is ~3 s** (unpickling the booster). Start the server before recording.

---

## 13. What can be improved

Ordered by value per hour of work.

### High value, low effort

1. **Stream the demo (~20 min).** Add a `--rate` flag to `run_demo.py` so transactions arrive at, say, 5/second with decisions printing live. Purely cosmetic, materially better on video.

2. **Cache the SHAP explainer path (~30 min).** Latency is 56 ms p50, which is slow for a GBDT — a 556-tree model should score in ~2 ms. The time goes to *two* separate `booster.predict()` calls (once for the score, once for `pred_contrib`) on a 431-column frame, plus DataFrame construction, plus a synchronous SQLite write per request. Fixes: request `pred_contrib` once and derive the score from the contributions, build the frame with a preallocated NumPy array instead of a DataFrame, and batch the audit writes. Expect p50 well under 10 ms.

3. **Report per-segment cost with confidence intervals (~30 min).** The `discover` card segment shows ₹20.1M per 10k against ₹3.35M for visa — but it has only 1,257 rows. That is very likely noise, and presenting it without an interval invites a judge to catch you.

4. **Sensitivity analysis on the cost constants (~1 hour).** Every headline number depends on ₹1,000 fee / 12% margin / 5% churn / ₹3,000 LTV. Sweep each ±50% and report how the optimal threshold and the headline gap move. This converts "we assumed" into "we tested our assumptions," which is a much stronger position — and it is exactly the question a sharp judge will ask.

### High value, medium effort

5. **Build the synthetic Razorpay-shaped layer (~4 hours).** This is the biggest gap. It would restore three things the dataset cannot support: real velocity features over real entity IDs, per-merchant thresholds, and a UPI-vs-cards risk-regime split. It also gives you a demo stream where you *can* hand-author transactions, removing the §12 constraint. Keep the honest labelling: public data for the statistics, synthetic for the demo.

6. **Repeated-seed variance (~1 hour).** Every number comes from a single training run. Retraining across 5 seeds and reporting the spread on the headline would show whether ₹112,750 is stable or a lucky draw. Currently the bootstrap CIs capture test-window sampling noise but **not** training variance — a limitation the README states but does not quantify.

7. **A minimal dashboard (~3 hours).** Cost curve, reliability diagram, live decision stream, review queue. Cut for time; would improve the video considerably.

### Genuine research directions

8. **Attack the high-value weakness directly.** The model scores PR-AUC 0.120 above $1k while 76% of all cost concentrates in the $250+ bands. Options: train a **cost-weighted objective** (weight each row by `c_FN(a)`, so the model optimises rupees rather than log-loss directly), or train a **separate high-value model**. This is the highest-upside modelling change available and it follows directly from our own diagnosis. It is also the honest answer to "so what do you do about it?"

9. **Reconsider `t*(a)` after fixing #8.** The amount-dependent threshold failed partly because probabilities are untrustworthy at high amounts. Fix the probabilities and the mechanism may become worth something. Re-run `run_configuration_choice.py` to find out — the harness already exists.

10. **Build the UID feature, carefully.** `card1_addr1 + floor(day − D1)` would enable genuine velocity features and would likely push PR-AUC toward 0.60+. But its power partly comes from the label-propagation artifact (§3), so it must be presented with that caveat. Worth doing *and* worth disclosing.

11. **Delayed-label simulation.** Chargebacks arrive up to 120 days later. Simulating that delay and measuring how fast the model decays without fresh labels would be a genuinely novel result for a hackathon, and it directly addresses "measured outcomes" in a way nobody else will.

### Housekeeping

12. `plan.md` is still committed and now contradicts the README (it promises per-merchant thresholds, velocity features and UPI analysis). Delete it or add a header marking it superseded.
13. Replace the deprecated `@app.on_event("startup")` with a lifespan handler.
14. `run_configuration_choice.py` reloads the parquet and re-scores from scratch twice. Caching would cut its runtime substantially.

---

## 14. Known weaknesses, ranked by how much they'd embarrass you

**If a judge probes, these are where the soft spots are. Better to volunteer them.**

1. **The cost constants are invented.** Not measured from Razorpay data, not sourced from an industry report — chosen as plausible. Every rupee figure scales with them. *Mitigation:* they're isolated in one versioned file, the README says so plainly, and improvement #4 would close this properly.

2. **US data wearing a rupee costume.** IEEE-CIS is US e-commerce in USD. Indian traffic has a different amount distribution and a completely different method mix. The method transfers; the numbers are illustrative.

3. **PR-AUC 0.498 is not a strong model.** Competitive solutions reach 0.60+ on temporal splits. We are at the bottom of our pre-registered range, because we have no UID/velocity features. Honest, but don't oversell the model — the *protocol* is the contribution, not the classifier.

4. **Recall is measured against reported chargebacks, not fraud.** Unreported fraud sits in the negative class by construction. True recall is unknowable on this dataset.

5. **Single seed, single model.** No ensembling, no hyperparameter search, no training-variance estimate.

6. **The calibration fold has a 4.32% base rate vs 3.44% in test.** Known, deliberately unfixed, and disclosed — but it is a real weakness, not a virtue.

7. **`discover` card segment cost (₹20.1M/10k) is almost certainly noise** at 1,257 rows. Currently reported without an interval.

8. **No production concerns addressed:** no auth on the API, no rate limiting, no model monitoring or drift alerting, no retraining pipeline, SQLite rather than a real database, no containerisation.

9. **The three-way policy's review band is heuristic.** It straddles the cost-optimal cut and widens to the capacity ceiling. A more principled construction would size it by expected value of information — where human review actually changes the decision often enough to pay for itself.

---

## Closing note

The most valuable output of this project is not the ₹112,750 headline. It is that **two of the ideas we proposed ourselves were measured and found not to work**, and that the protocol was strict enough to catch the second one *after* it had already looked like a success on validation.

A submission with one significant positive result and two well-diagnosed negatives is more credible than one with four unaudited wins. That was the bet.
