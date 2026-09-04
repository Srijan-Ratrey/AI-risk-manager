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
15. [Artifact reference](#15-artifact-reference)
16. [Subtleties and gotchas](#16-subtleties-and-gotchas)

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

The whole file, since "everything lives here" is only a useful claim if you can see all of it:

```yaml
version: "1.0.0"
currency: INR
usd_to_inr: 88.0                      # an ASSUMPTION, not a spot rate; see §3

false_negative:                       # fraud approved
  chargeback_amount_multiplier: 1.0   # you eat the full transaction value
  chargeback_fee_inr: 1000.0          # flat scheme/gateway dispute fee
  ops_handling_inr: 200.0             # analyst time on the dispute

false_positive:                       # legitimate customer blocked
  margin_rate: 0.12                   # THE load-bearing assumption; see §3
  churn_probability: 0.05
  customer_ltv_inr: 3000.0
  support_contact_inr: 100.0

review:
  cost_per_review_inr: 50.0           # analyst time per escalation
  max_review_rate: 0.05               # operational capacity ceiling
```

Three caveats on that file. `usd_to_inr` multiplies through **every** rupee figure, so it belongs in view rather than in a footnote. `margin_rate` is the one constant the headline actually depends on — see §3. And `cost_per_review_inr` is currently **decorative**: no reported number includes it (§16.2).

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
| `TransactionAmt`, `ProductCD`, `card4`, `card6` | amount, product code, card network, debit/credit | **yes** |
| `card1`, `card2`, `card3`, `card5` | masked bank/card attributes | no |
| `addr1`, `addr2`, `dist1`, `dist2` | billing address/country, distances | partly |
| `P_emaildomain`, `R_emaildomain` | email **domain** only, never the address | yes |
| `C1`–`C14` | **counting features** — "how many addresses are associated with this card". Definitions masked. | partly |
| `D1`–`D15` | timedeltas — "days since previous transaction" | partly |
| `M1`–`M9` | match flags — "name on card matches address" | partly |
| `V1`–`V339` | **339** | Vesta engineered features: ranking, counting, entity relations. Meanings masked. | **no** |
| `id_01`–`id_38`, `DeviceType`, `DeviceInfo` | device/browser/OS. **Present on only 23.8% of rows.** | partly |

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

### Currency, and which assumptions actually matter

`TransactionAmt` is USD. We re-denominate at a single declared ₹88/USD in `costs.yaml`.

**88 is an assumption, not a spot rate.** I should be blunt about how it got there: I wrote it from memory while building the cost model and never verified it. The rate as of September 2026 is ~₹94. That is a real lapse in a project whose pitch is auditable assumptions.

But there is a more interesting problem underneath the sloppy one: **there is no single correct rate.** The transactions are from **2019**, when USD/INR was ~₹70. Valuing them at a 2026 rate is a choice — you can argue for 70 (what those amounts were actually worth when transacted) or for today's 94 (what an Indian merchant thinks in current money). The file previously made that choice silently, which was the worse sin.

`run_rate_sensitivity.py` settles whether any of it matters. Reproduce with:

```bash
.venv/bin/python run_rate_sensitivity.py   # -> reports/rate_sensitivity.csv
```

| `usd_to_inr` | Cost-optimal threshold | `t*(a)` range | Cost / 10k | Headline / 10k |
|---|---|---|---|---|
| 70 (2019 rate) | 0.130 | 0.107–0.172 | ₹2,675,089 | ₹88,660 |
| 80 | 0.130 | 0.107–0.172 | ₹3,019,703 | ₹102,043 |
| **88 (shipped)** | **0.130** | 0.107–0.172 | **₹3,295,395** | **₹112,750** |
| 94 (Sept 2026) | 0.130 | 0.107–0.172 | ₹3,502,163 | ₹120,780 |
| 100 | 0.130 | 0.107–0.172 | ₹3,708,932 | ₹128,810 |

**The decision never changes.** One distinct operating point across the whole range — 0.130 — to the resolution of the 0.001 sweep grid. (Unchanged *at that resolution*, not exactly invariant in the mathematical sense; overclaiming here would repeat the original error in a new place.)

`t*(a)`'s range does not move **at all**, and the reason is worth seeing:

```
t*(a) → fixed_fp / (fixed_fp + fixed_fn)                    as a → 0     = 0.172
t*(a) → margin_rate / (margin_rate + chargeback_multiplier) as a → ∞     = 0.107
```

Neither limit contains the exchange rate. The rate only relocates where each transaction sits *along* the curve; it cannot reshape the curve. Only absolute magnitudes scale, and even they scale sub-proportionally (+6.3% for a +6.8% rate change) because the flat ₹1,200 and ₹250 terms don't move.

### The assumptions that do matter, and where the thesis stops holding

Sweeping all eight constants at ±50%, **six leave the operating point at 0.130.** The exchange rate, chargeback fee, ops cost, churn probability, customer LTV and support cost all fail to move it.

Two do move it — `margin_rate` and `chargeback_amount_multiplier` — and they turn out to be **one mechanism, not two.** Both control the *ratio* between what a missed fraud costs and what a false block costs. Raising the margin makes a false positive dearer; recovering part of a chargeback makes a false negative cheaper. Either narrows the ratio, and the cost-optimal threshold climbs toward the F1-optimal one (0.222) — which is precisely what closes the gap the headline measures.

Ordered by that ratio, the picture is monotone and the boundary is sharp:

| FN:FP ratio (at ₹1,000) | Driven by | Threshold | Headline / 10k | 95% CI | |
|---|---|---|---|---|---|
| 7.3× | chargeback multiplier 1.5 | 0.041 | ₹460,879 | 247,957 – 692,695 | significant |
| 7.1× | margin 6% | 0.037 | ₹392,790 | 240,246 – 556,373 | significant |
| **5.9×** | **shipped** | **0.130** | **₹112,750** | 34,904 – 200,873 | significant |
| 5.1× | margin 18% | 0.142 | −₹3,419 | −70,590 – 74,592 | **not significant** |
| 4.6× | chargeback multiplier 0.5 | 0.142 | **−₹48,814** | −85,776 – −6,229 | **significantly negative** |

**Around a 5× ratio the advantage vanishes. Below it, cost-optimising is measurably worse than optimising F1** — a result I did not expect and would not have found without sweeping. The mechanism is a generalisation effect: as the ratio narrows, the cost curve flattens near its minimum, so the threshold chosen on val-B is a noisier estimate of the test-optimal one, and it can land worse than F1's 0.222. The same flatness that makes the choice less consequential also makes it less reliable.

**So the thesis is a claim about merchants with thin margins and unrecoverable chargebacks** — and that should have been stated up front rather than discovered by sweeping. Electronics, marketplaces and groceries live at the profitable end. A digital-goods merchant at a 60% margin, or one who wins half their disputes, should expect nothing here and possibly a small loss.

That is a boundary condition, not a refutation. But anyone quoting ₹112,750 without the 12%-margin assumption behind it is overstating the result, and blurring it would be the fastest way to lose credibility with a judge who knows retail economics.

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
| train (minus tail) | 318,892 | 3.401% | 1–91 | fit the trees |
| train tail | 35,432 | 3.223% | 91–101 | early stopping |
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
| Block everything | nothing | — | ₹16,391,839 | the other extreme is 3.31× worse than doing nothing |
| Random at base rate | nothing | 0.035 | ₹5,328,749 | worse than doing nothing |
| Amount > threshold | 1 feature | 0.037 | ₹4,962,233 | tuning could find nothing worth blocking — amount alone is useless |
| **Count rule (C12 > 3)** | 1 feature | 0.161 | **₹4,669,678** | the honest foil |
| Logistic regression | 5 features | 0.209 | ₹4,838,403 | **loses on money despite winning on PR-AUC** |
| LightGBM | full | **0.498** | **₹3,295,395** | |

Precisely: the tuned amount rule blocks **1 row out of 118,108** — a block rate of 8.5e-06. That single blocked legitimate transaction is *why* its cost is ₹4,820/10k **worse** than never-fraud rather than identical to it. Amount alone carries no usable signal at this base rate.

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
    "bagging_freq": 1,               # without this LightGBM IGNORES bagging_fraction
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": SEED,                    # 42
    "num_threads": 0,                # all cores
    # Deliberately absent: scale_pos_weight / is_unbalance
}
```

Trained with `num_boost_round=3000` and `stopping_rounds=100`; early stopping fired at iteration **556**. `bagging_freq` is the non-obvious one — LightGBM silently ignores `bagging_fraction` unless a frequency is set, so omitting it would mean no bagging at all while the config appeared to ask for it.

### Why there is no class weighting — the correction that matters

Standard advice says: use `scale_pos_weight` *instead of* SMOTE, because resampling destroys probability calibration.

**The first half is right and the second half is wrong.** `scale_pos_weight` reweights the loss, which shifts the model's implied base rate — its outputs no longer estimate `P(fraud | x)` for the true class balance. That breaks calibration just as surely as resampling does. It is a *less obvious* violation, which makes it more dangerous.

The correct rule is: **any prior-shifting technique requires recalibration afterwards.** Since our entire rupee argument depends on `p` meaning what it says, the cleanest path is to train unweighted with enough trees and let isotonic regression do the calibration explicitly. GBDTs handle a 3.5% positive rate fine given sufficient boosting rounds.

### Results

| Metric | Test | Note |
|---|---|---|
| **PR-AUC (raw)** | **0.4981** | ranking quality; pre-registered expectation 0.50–0.65 |
| PR-AUC (calibrated) | 0.4792 | **what the policy actually consumes** |
| ROC-AUC | 0.8825 | reported only for leaderboard comparison |
| Recall @ 0.5% FPR | 0.3632 | the operationally meaningful one |
| Best iteration | 556 | early stopping on train tail |

**Both PR-AUC figures matter and they differ.** Every headline ranking metric here is computed on the *raw* booster output. Isotonic is a monotone step function, so it collapses many raw scores onto one probability — 171 distinct values over 118,108 rows — and those ties cost **0.019 PR-AUC**. Quoting only 0.4981 would be reporting the ranking of a score the decision policy never sees. It is a genuine trade: ranking resolution given up in exchange for probabilities the cost model can multiply.

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
score < 0.0664            →  AUTO-APPROVE
0.0664 ≤ score < 0.2000   →  MANUAL REVIEW    (4.03% of traffic on test)
score ≥ 0.2000            →  AUTO-BLOCK
```

**How the band is sized.** `derive_review_band` (`src/costs.py`) starts at the distinct probability nearest the cost-optimal cut (0.130) and grows outward, always taking whichever neighbouring value sits closer to the cut, stopping before the band's mass would exceed the 5% capacity ceiling. Like every threshold in this project it is **derived on val-B and frozen**; 4.03% is the *realised* rate on test.

The band contains **9.76% fraud against a 3.44% base rate**, so it is genuinely selecting ambiguous cases rather than padding a queue.

**Why this is fiddlier than taking the nearest 5% of rows — and a bug it caused.** The first version did exactly that: select the nearest `k` rows by `|p − t_cost|`, then record that set's `[min, max]` as the band. The service then applied that pair as an interval. Because isotonic emits only 171 distinct probabilities over 118,108 rows, the *tie groups are large*: the row-selection excluded some rows at the boundary that the interval then swept back in. A band documented at **5.00%** actually routed **6.22%** of traffic — a review queue 24% over its stated ceiling, and a number in the README that was not true of the shipped system.

The fix is to expand by whole tie-groups and return the interval itself, so the stated rate is the served rate. `tests/test_costs.py::test_review_band_interval_respects_the_ceiling_despite_ties` is the regression guard, and it reproduces the tie structure that caused the original bug.

It also has to be derived on **val-B**, not test. The original derived the band from test probabilities while every other threshold came from val-B — the same oracle mistake the threshold protocol exists to avoid, hiding in the one place nobody was looking.

Note that on val-B the band uses only **3.75%** of its 5% budget: the next tie-group would have overshot, so it stops short. Coarse quantisation means the ceiling is rarely hit exactly.

**Why review rate is a first-class metric.** A model that routes 30% of traffic to human review is unusable at any precision — there aren't enough analysts. Reporting precision without reporting review rate hides that failure mode.

**Note the deployed block threshold (0.2000) differs from the cost-optimal threshold (0.1300).** Transactions between the two are escalated rather than auto-blocked. This trades a small amount of expected cost for reversibility on the least certain decisions — which is the point of having a review band at all.

**The serving path applies this global band, not `t*(a)`.** Since the amount-dependent rule measured as a null (§10), shipping it would add complexity for no measured gain. `amount_inr` is required by the request schema but is recorded for audit and cost attribution only — it does not enter the decision. The `Field` description in `src/service.py` says so, so the API docs cannot drift from this.

**Bounded actions.** The service never moves money. Both `BLOCK` and `REVIEW` are reversible. Every decision is appealable through `POST /v1/appeal/{id}`, and an appeal is recorded as labelled feedback — never a deletion. The blast radius of a wrong call is one payment, recoverable.

---

## 9. The service

[`src/service.py`](src/service.py) — FastAPI.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | Score a transaction, return a bounded decision |
| `GET /v1/health` | Reports degraded state honestly, including load errors |
| `GET /v1/audit/{id}` | Full history for a transaction: decisions + overturns |
| `POST /v1/appeal/{id}` | Overturn a decision; appended, never overwritten |

### Request and response schema

**`POST /v1/score` request** (`ScoreRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `transaction_id` | `str` | auto `txn_<uuid12>` | Caller's id; generated if omitted |
| `amount_inr` | `float` | **required**, `≥ 0` | Audit and cost attribution only — **does not enter the decision** (see §8) |
| `features` | `dict[str, Any]` | `{}` | Model features by name. Omitted ⇒ missing, which is a branch direction LightGBM learned, not an error |

**Response** (`ScoreResponse`):

| Field | Type | Notes |
|---|---|---|
| `transaction_id` | `str` | echoed |
| `decision` | `str` | `APPROVE` / `REVIEW` / `BLOCK` |
| `score` | `float` | calibrated probability; `0.0`/`1.0` in degraded mode |
| `threshold_used` | `float` | block threshold (0.2000), or `FALLBACK_THRESHOLD` (3.0) when degraded |
| `reason_codes` | `list[str]` | top-3 SHAP contributors, humanised |
| `merchant_message` | `str` | customer-safe text; never contains feature names |
| `model_version` | `str` | `lgbm-1.0.0` or `fallback-rule-1.0.0` |
| `cost_model_version` | `str` | from `costs.yaml`, stamped for replay |
| `degraded` | `bool` | whether the rule path served this request |
| `latency_ms` | `float` | server-side, includes SHAP and the audit write |

**`POST /v1/appeal/{id}`** takes `overturn_to` and optional `note` as **query parameters**, not a body. Returns `201` with the new `appeal_id`; `422` for a decision outside `{APPROVE, REVIEW, BLOCK}`; `404` if the transaction was never scored.

**`GET /v1/audit/{id}`** returns `{transaction_id, decisions: [...], appeals: [...]}` — raw table rows, so `decisions` entries carry the 12 columns above plus `id` and `ts`.

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
- **Fail closed** (block everything) → ₹16.4M per 10k, which the ladder measures at **3.31× worse than doing nothing at all**.

The fallback rule was selected by the baseline ladder and costs ₹4.67M per 10k — less than approving everything. So degraded mode is *genuinely safe*, not a token gesture. That claim is measured, not asserted.

**A third way to fail, which we originally had.** The rule reads one feature, `C12`, and nothing in the request schema requires it. The first implementation did `value = 0.0 if value is None else float(value)` — so a payload *without* `C12` scored 0.0 and was **APPROVED**, with a confident-looking response. That is failing open on precisely the requests we know least about, and it was reachable with an ordinary payload.

The rule now escalates to `REVIEW` when its own input is absent or non-numeric. This is the same principle the review band exists for: when unsure, do not guess with money. `_fallback` returns the decision directly rather than a score to be banded, because running a 0/1 rule output through a probability band is meaningless arithmetic.

Two related fixes went in alongside it:

- **`threshold_used` in degraded mode** used to record `0.1852`, a probability threshold that played no part in a rule decision — so a replayed audit row looked as though it had been judged against a threshold nobody chose. It now records the rule's own cut (3.0).
- **`Engine.__init__` used to hardcode a fallback band** `[0.065, 0.185]` that merely *resembled* the real `[0.0664, 0.2000]`. If `thresholds.json` failed to load, the service would serve a different policy while reporting `degraded: false` and `status: ok`. There is now no default: absent thresholds degrade, exactly like an absent model.

**This path is tested.** `tests/test_service.py` points `MODEL_PATH` at a nonexistent file and asserts the service still starts, returns correct `APPROVE`/`BLOCK` decisions, escalates when `C12` is absent, records the rule threshold, and audits all of it.

### Audit log

**Two** append-only SQLite tables.

`decisions` carries transaction id, amount, score, threshold used, decision, reason codes, model version, **cost-model version**, degraded flag and latency. Stamping the cost-model version is what makes a historical decision replayable — if the constants change, you can still reconstruct why a decision was made in March.

`appeals` carries overturns. This is a correction: the original implementation ran `UPDATE decisions SET overturned_to = ?`, which *mutated the audit row in place* while the docstring claimed "never a deletion". True — it wasn't a deletion, it was an overwrite, which for an audit log is no better. Retraining needs the **pair** (what the model said, what the human said), so overwriting the decision destroys the very label the appeal exists to collect. Appeals now append, and `GET /v1/audit/{id}` returns both lists.

`POST /v1/appeal/{id}` also validates `overturn_to` against `{APPROVE, REVIEW, BLOCK}` — it previously accepted any string, including `"BANANA"` — and returns 404 for an unknown transaction rather than reporting success with `rows_updated: 0`.

### Merchant-facing vs internal explanations

The customer sees: *"Payment could not be completed. Please contact support to appeal."*

The internal log sees: *"velocity/count signal (C1) elevated / aggregate risk feature (V258) elevated / time-since-previous-activity signal (D14) unusual."*

Publishing the exact rule set teaches fraudsters what to avoid. A test asserts the merchant-facing string never contains internal feature names.

### Latency

1,000 sequential requests replaying real test-window transactions, **including** SHAP reason codes and the audit write:

| p50 | p95 | p99 | max |
|---|---|---|---|
| 58.5 ms | **78.2 ms** | 81.4 ms | 144.3 ms |

Budget was p95 < 100 ms. Met, but with less headroom than a GBDT deserves — see [§13](#13-what-can-be-improved) for where the time actually goes.

**Measured in-process**, via `fastapi.testclient.TestClient`, which calls the ASGI app directly. These figures therefore exclude uvicorn, HTTP parsing and socket overhead. The live-server verification in §12 covers endpoint *behaviour*, not this table — do not conflate the two.

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

| File | What it does | Why it exists |
|---|---|---|
| [`costs.yaml`](costs.yaml) | Every rupee assumption, versioned | One auditable place; version stamped into audit rows |
| [`src/costs.py`](src/costs.py) | `CostModel`, `t*(a)`, threshold sweep, bootstrap CIs | The heart — all money math |
| [`src/data.py`](src/data.py) | Load, merge, downcast, temporal split, leakage assertions | Splits are the credibility budget |
| [`src/evaluate.py`](src/evaluate.py) | PR-AUC, ECE (quantile-binned), reliability, recall@FPR | Metrics that survive 3.4% imbalance |
| [`src/baselines.py`](src/baselines.py) | The six-rung ladder | Answers "how much was free?" |
| [`src/model.py`](src/model.py) | LightGBM + isotonic, fold discipline | Model, plus the no-class-weighting rationale |
| [`src/service.py`](src/service.py) | FastAPI, audit log, fallback, schema pinning | Bounded, explainable, degradable |
| `build_schema.py` | Pins training dtypes to `models/schema.pkl` | Prevents train/serve skew |
| `run_baselines.py` | Runs the ladder | |
| `run_model.py` | Train → calibrate → threshold → cost curve → segments | The main pipeline |
| `run_band_calibration.py` | Per-band diagnosis | Failure 2, diagnosis half |
| `run_configuration_choice.py` | Honest configuration selection on val-B | Failure 2, verdict half |
| `run_demo.py` | Replay real transactions + latency benchmark | The demo and the p95 number |
| `run_figures.py` | Five figures | README and video |
| [`tests/test_costs.py`](tests/test_costs.py) | 17 tests on the money math | A sign error here is invisible otherwise |
| [`tests/test_service.py`](tests/test_service.py) | 12 tests incl. degraded mode and fail-safe | Proves the fallback rather than claiming it |

### Execution order

```
src/data.py  →  run_baselines.py  →  run_model.py  →  build_schema.py
                                          ↓
              run_band_calibration.py → run_configuration_choice.py
                                          ↓
                          run_figures.py    run_demo.py
```

`src/data.py` is a **diagnostic**, not a required first step. It prints the split table and writes `data/splits.json`, but nothing reads that file — every downstream script recomputes `make_splits(df)` from the cached parquet. The real dependency is that `run_model.py` must precede `build_schema.py`, `run_figures.py` and `run_demo.py`, since they consume `models/fitted.pkl` and `models/thresholds.json`.

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
- `GET /v1/audit/{id}` → `{decisions: [...], appeals: [...]}`, with both version stamps
- `POST /v1/appeal/{id}` → `201` with a new `appeal_id`; the decision row is untouched
- Kill the model file, restart → serves from the fallback rule, `degraded: true`, still audits

`run_demo.py` replays 1,000 real test transactions and prints a live decision stream with all three outcomes plus the latency distribution. **That is your demo.**

### The constraint: you cannot hand-author a convincing fraudulent transaction

This is the important finding. `run_payload_sensitivity.py` measures how the score responds to payload completeness, using a real fraud transaction (`TransactionID` 3513412, $34.00) that the full pipeline blocks at 0.9000. Reproduce it with:

```bash
.venv/bin/python run_payload_sensitivity.py   # -> reports/payload_sensitivity.csv
```

| Payload | Features sent | Score | Decision |
|---|---|---|---|
| Interpretable only (what a human can write) | 6 | 0.2067 | BLOCK |
| + `C` counting features | 20 | 0.7831 | BLOCK |
| + `D` time-delta features | 30 | **0.9000** | BLOCK |
| All features | 333 | 0.9000 | BLOCK |
| **Invented "card testing" values** | 8 | **0.0246** | **APPROVE** |

Good news: **~30 real features reproduce the full score.** You do not need all 431, and the 339 anonymised V-columns add nothing beyond the C and D columns for this row.

Bad news, and it is the last row: those values must be *real*. A hand-crafted payload with invented counts (`C1=45, C12=28, C13=60`) scores 0.0246 and is approved. The model keys on the *joint pattern* across correlated features, not on individual large numbers. Inventing plausible-looking values produces an out-of-distribution row, and the model — correctly — does not recognise it as anything.

Note also that 0.7831 and 0.0246 are **plateau values** shared by many transactions, not row-specific estimates: isotonic emits only 171 distinct probabilities (§16.1).

### What this means for your video

**Do this:** replay real test-window rows through the live service. `run_demo.py` already selects examples that produce APPROVE, REVIEW and BLOCK, and prints the reason codes and audit entries for each. Show the terminal, or drive the same rows through `curl` for a more API-flavoured demo.

**Don't do this:** type a "suspicious-looking" transaction into a form live and expect it to be blocked. It won't be, and if a judge asks you to try one, the honest answer is the table above.

**The demo you actually have, in order:**

1. `GET /v1/health` — model loaded, versions shown
2. `POST /v1/score` on a real legit row → APPROVE with reason codes
3. `POST /v1/score` on a real ambiguous row → REVIEW, and note the 4.03% queue against a 5% ceiling
4. `POST /v1/score` on a real fraud row → BLOCK at 0.90 with three reason codes
5. `GET /v1/audit/{id}` — show the immutable row with both version stamps
6. `POST /v1/appeal/{id}` — overturn it, then re-fetch the audit: a new `appeals` entry, the `decisions` row unchanged
7. `mv models/fitted.pkl /tmp && restart` → `degraded: true`, still decides, still audits. Then POST a payload **without** `C12` → `REVIEW`, not a silent approve
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

4. ~~**Sensitivity analysis on the cost constants.**~~ **Done** — `run_rate_sensitivity.py`, results in §3. Seven of eight constants leave the operating point untouched at ±50%; `margin_rate` is the only one that moves it, and it bounds the whole thesis to thin-margin merchants. What remains: sweep the constants *jointly* rather than one at a time, since a merchant's margin and LTV are unlikely to be independent.

### High value, medium effort

5. **Build the synthetic Razorpay-shaped layer (~4 hours).** This is the biggest gap. It would restore three things the dataset cannot support: real velocity features over real entity IDs, per-merchant thresholds, and a UPI-vs-cards risk-regime split. It also gives you a demo stream where you *can* hand-author transactions, removing the §12 constraint. Keep the honest labelling: public data for the statistics, synthetic for the demo.

6. **Repeated-seed variance (~1 hour).** Every number comes from a single training run. Retraining across 5 seeds and reporting the spread on the headline would show whether ₹112,750 is stable or a lucky draw. Currently the bootstrap CIs capture test-window sampling noise but **not** training variance — a limitation the README states but does not quantify.

7. **A minimal dashboard (~3 hours).** Cost curve, reliability diagram, live decision stream, review queue. Cut for time; would improve the video considerably.

### Genuine research directions

8. **Attack the high-value weakness directly.** The model scores PR-AUC 0.120 above $1k while 76% of all cost concentrates in the $250+ bands. Options: train a **cost-weighted objective** (weight each row by `c_FN(a)`, so the model optimises rupees rather than log-loss directly), or train a **separate high-value model**. This is the highest-upside modelling change available and it follows directly from our own diagnosis. It is also the honest answer to "so what do you do about it?"

9. **Reconsider `t*(a)` after fixing #8.** The amount-dependent threshold failed partly because probabilities are untrustworthy at high amounts. Fix the probabilities and the mechanism may become worth something. Re-run `run_configuration_choice.py` to find out — the harness already exists.

10. **Build the UID feature, carefully.** `card1_addr1 + floor(day − D1)` would enable genuine velocity features and would likely push PR-AUC toward 0.60+. But its power partly comes from the label-propagation artifact (§3), so it must be presented with that caveat. Worth doing *and* worth disclosing.

11. **Delayed-label simulation.** Chargebacks arrive up to 120 days later. Simulating that delay and measuring how fast the model decays without fresh labels would be a genuinely novel result for a hackathon, and it directly addresses "measured outcomes" in a way nobody else will.

12. **Wire the review cost into the reported figures.** `cost_per_review_inr: 50.0` is in the config and enters no number (§16.2). Every ₹/10k figure currently treats manual review as free, which flatters the three-way policy. Passing `reviewed=` through `sweep_thresholds` and the policy tables is maybe an hour, and it would make the review-band argument cost-complete rather than cost-adjacent.

### Housekeeping

13. Replace the deprecated `@app.on_event("startup")` with a lifespan handler.
14. `run_configuration_choice.py` reloads the parquet and re-scores from scratch twice. Caching would cut its runtime substantially.
15. Delete `model.top_reason_codes()` and `Splits.load()`, or wire them into the paths that reimplement them (§16.7).
16. Derive `MODEL_VERSION` from the artifact rather than hand-editing a constant (§16.8).

---

## 14. Known weaknesses, ranked by how much they'd embarrass you

**If a judge probes, these are where the soft spots are. Better to volunteer them.**

1. **The cost constants are invented** — not measured from Razorpay data, not sourced from an industry report, chosen as plausible. The ₹88/USD rate is the clearest example: written from memory, never verified, and wrong by about 6% against the September 2026 spot rate.

   *Mitigation, and it is now a real one:* their influence is **measured**, not asserted (§3). Seven of eight constants do not move the operating point at ±50%. But the eighth does, and it does not merely shift the number — **at an 18% margin the headline claim stops being statistically distinguishable from zero.** That is the single biggest caveat in this project. It is a boundary condition on the thesis rather than a refutation of it, but anyone quoting ₹112,750 without also quoting the 12%-margin assumption behind it is overstating the result.

2. **US data wearing a rupee costume.** IEEE-CIS is US e-commerce in USD, and 2019-vintage at that — converting those amounts at any 2026 rate is a choice, not a conversion. Indian traffic has a different amount distribution and a completely different method mix. The method transfers; the numbers are illustrative. At least this one provably doesn't change the decision (§3).

3. **PR-AUC 0.498 is not a strong model.** Competitive solutions reach 0.60+ on temporal splits. We are at the bottom of our pre-registered range, because we have no UID/velocity features. Honest, but don't oversell the model — the *protocol* is the contribution, not the classifier.

4. **Recall is measured against reported chargebacks, not fraud.** Unreported fraud sits in the negative class by construction. True recall is unknowable on this dataset.

5. **Single seed, single model.** No ensembling, no hyperparameter search, no training-variance estimate.

6. **The calibration fold has a 4.32% base rate vs 3.44% in test.** Known, deliberately unfixed, and disclosed — but it is a real weakness, not a virtue.

7. **`discover` card segment cost (₹20.1M/10k) is almost certainly noise** at 1,257 rows. Currently reported without an interval.

8. **No production concerns addressed:** no auth on the API, no rate limiting, no model monitoring or drift alerting, no retraining pipeline, SQLite rather than a real database, no containerisation.

9. **The three-way policy's review band is heuristic.** It straddles the cost-optimal cut and widens to the capacity ceiling. A more principled construction would size it by expected value of information — where human review actually changes the decision often enough to pay for itself.

---

## 15. Artifact reference

Everything under `reports/` is generated and committed — `.gitignore` excludes `*.csv` for the 683 MB dataset but re-includes `!reports/**/*.csv`, because these tables *are* the evidence. `data/` and `models/` are not committed.

| Artifact | Written by | Contains |
|---|---|---|
| `reports/results.json` | `run_model.py` | Every headline number: both PR-AUCs, ROC-AUC, recall@FPR, ECE before/after, the frozen and oracle thresholds, review band and rates, and both bootstrap CIs |
| `reports/baseline_ladder.csv` | `run_baselines.py` | 6 rungs × PR-AUC, recall@FPR, precision, recall, block rate, ₹/10k |
| `reports/policies.csv` | `run_model.py` | 4 operating points (cost-, F1-, accuracy-optimal, `t*(a)`) with realised test cost |
| `reports/cost_curve.csv` | `run_model.py` | 1,001 rows — cost per 10k at every threshold on the 0.001 grid |
| `reports/segments.csv` | `run_model.py` | 16 segments across ProductCD / card4 / card6 / amount band |
| `reports/reliability_before.csv` · `_after.csv` | `run_model.py` | 20 and 14 quantile bins: mean predicted vs observed rate |
| `reports/band_diagnosis.csv` | `run_band_calibration.py` | Per-amount-band fraud rate, mean predicted, ECE, PR-AUC — the evidence for the Failure 1 diagnosis |
| `reports/band_calibration.csv` | `run_band_calibration.py` | Per-band ECE, global vs per-band calibration |
| `reports/configuration_choice.json` | `run_configuration_choice.py` | The 2×2 selection under permissive and strict per-band rules — the Failure 2 verdict |
| `reports/rate_sensitivity.csv` | `run_rate_sensitivity.py` | Operating point and headline at `usd_to_inr` 70–100 |
| `reports/cost_sensitivity.csv` | `run_rate_sensitivity.py` | All six other cost constants at ±50%, 18 rows |
| `reports/margin_sensitivity.csv` | `run_rate_sensitivity.py` | Headline **with bootstrap CIs** at 6% / 12% / 18% margin — the boundary condition on the thesis |
| `reports/payload_sensitivity.csv` · `.json` | `run_payload_sensitivity.py` | Score vs payload completeness; the demo constraint in §12 |
| `reports/latency.json` | `run_demo.py` | p50/p95/p99/max over 1,000 requests, and whether the budget was met |
| `reports/demo_decisions.csv` | `run_demo.py` | 1,000 replayed decisions with score, decision, reason codes, true label |
| `reports/figures/*.png` | `run_figures.py` | cost_curve · baseline_ladder · reliability · threshold_curve · segments |
| `models/fitted.pkl` · `schema.pkl` · `thresholds.json` | `run_model.py`, `build_schema.py` | Booster + calibrator, pinned dtypes, frozen thresholds. Gitignored. |
| `data/splits.json` | `src/data.py` | Frozen split boundaries. Written for reproducibility; **nothing reads it** (see §11). |
| `audit.db` | `src/service.py` | SQLite `decisions` + `appeals`. Gitignored. |

**One artifact deserves more attention than it gets.** `band_calibration.csv` shows per-band calibration made ECE *worse* in 3 of 5 bands and worse overall (0.00492 vs 0.00408). That is independent corroboration of Failure 2 — the refinement was not merely unhelpful on cost, it was not even better calibrated — and it currently sits in a committed file that the narrative never cites.

---

## 16. Subtleties and gotchas

Things in the code that are surprising, deliberate, or would mislead a reader who found them alone.

**1. The "probability" takes only 171 distinct values.** Isotonic is a monotone step function over 118,108 test rows. In the 1,000-request demo, 183 rows share the single score `0.002863`. Anywhere this document quotes a score, it is a plateau value shared by many transactions, not a row-specific estimate. This is also why the review band needs tie-group logic (§8).

**2. The ₹50 review cost never enters any reported number.** `CostModel.total_cost` accepts a `reviewed` argument, and every pipeline call passes `None`. Only `tests/test_costs.py` exercises it. So every ₹/10k figure treats manual review as free, which flatters the three-way policy against a pure block/approve baseline. Wiring it in is a modelling change we did not make in time; it is listed in §13.

**3. Categorical dtypes are fitted over the full frame, test included.** `prepare_features` in `src/data.py`. This is defensible — a category *mapping* is not fitted knowledge about the label, and letting the encoding differ between train and test would silently corrupt inference — but in a project whose thesis is protocol hygiene, it deserves stating rather than hiding in a docstring. No label or aggregate statistic crosses a split boundary anywhere.

**4. `TransactionDT` is deliberately excluded from the features.** `NON_FEATURES` in `src/data.py`. A tree will happily memorise the time axis and then collapse on an out-of-time window. This looks like an oversight and is not.

**5. Latency is measured in-process.** `run_demo.py` uses `TestClient`, which calls the ASGI app directly — no uvicorn, no sockets. And 58 ms p50 is *slow* for a 556-tree model that should score in ~2 ms: the time goes to two separate `booster.predict()` calls (score, then `pred_contrib`) over a 431-column frame, DataFrame construction per request, and a synchronous SQLite write. §13 has the fix.

**6. The demo sample is 25% fraud by construction.** `run_demo.py` oversamples so all three decisions appear on video. The printed decision mix is not production-representative; the base rate is 3.44%.

**7. Dead code.** `model.top_reason_codes()` has zero callers — `service.py` reimplements it inline, so anyone tracing "where do reason codes come from?" will read the wrong function. `Splits.load()` likewise has none.

**8. `MODEL_VERSION` is a hand-edited constant.** `"lgbm-1.0.0"` in `src/service.py`, not derived from the artifact. Retraining without editing it produces audit rows that claim a version the model no longer is.

**9. Thresholds beyond the review band are written and never read.** `models/thresholds.json` carries `cost_optimal`, `f1_optimal` and `accuracy_optimal`; the service uses only `review_band`. They are kept for replay and analysis.

**10. Threshold precision is not uniform.** The sweep runs on `linspace(0, 1, 1001)`, so `0.1300` and `0.1200` are 3-decimal grid points, while band edges like `0.066367` are isotonic plateau values carried at full precision. Tables mixing the two imply a uniformity that is not there.

**11. `shap` is pinned in `requirements.txt` and never imported.** SHAP values come from LightGBM's own `pred_contrib`, which computes exact tree SHAP without a separate explainer object. Correct engineering, misleading dependency list.

**12. `/v1/health` returns raw exception reprs** — including absolute filesystem paths — to unauthenticated callers. Fine locally, wrong in production. Listed under §14.8.

---

## Closing note

The most valuable output of this project is not the ₹112,750 headline. It is that **two of the ideas we proposed ourselves were measured and found not to work**, and that the protocol was strict enough to catch the second one *after* it had already looked like a success on validation.

A submission with one significant positive result and two well-diagnosed negatives is more credible than one with four unaudited wins. That was the bet.
