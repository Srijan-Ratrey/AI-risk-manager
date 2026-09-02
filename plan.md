> [!WARNING]
> **SUPERSEDED — kept for provenance, do not read as current.**
>
> This is the ORIGINAL plan, written before the dataset was inspected. Three of
> its signature features turned out to be uncomputable on IEEE-CIS, which has no
> merchant identifier, no card/device/IP identifier, and no payment-method or
> currency column:
>
> - per-merchant thresholds (section 6)
> - per-card velocity features (section 2) and the velocity baseline (section 4)
> - the UPI-vs-cards analysis (sections 1, 3)
>
> It also contains a technical error: section 5 recommends `scale_pos_weight`
> instead of SMOTE on calibration grounds, but reweighting shifts the predicted
> base rate and breaks calibration just as surely.
>
> **[README.md](README.md) is the submission. [EXPLAINER.md](EXPLAINER.md) is
> the engineering rationale.** Where this file disagrees with either, they are
> right and this is wrong.

---

# Razorpay Buildathon — Track 2: AI Risk Manager
## Full build plan

**Working title:** *Rupee-Optimal Risk* — a fraud/chargeback decision engine that optimises money lost, not F1.

---

## 0. The positioning (read this first, it drives every other choice)

Almost every entry in this track will be: train a classifier on a fraud dataset, report "97% accuracy" or "0.92 AUC", ship a dashboard.

The judging criteria explicitly ask for something else:

> "Honest metrics including false-positive costs" · "explainable, bounded money actions with audit trails" · "measured outcomes" · "exception handling and graceful failure"

So the thesis of this project is one sentence:

> **A fraud model's job is not to be accurate. It is to lose the merchant the least money — and those are different objectives with different optimal thresholds.**

Everything below exists to prove that claim with numbers.

**The three things that will differentiate this from every other submission:**
1. A **cost model in rupees**, so the operating point is chosen by expected loss rather than F1.
2. **Honest baselines** — a null-model ladder showing how much of the headline number was free.
3. **A three-way decision policy** (approve / review / block) with calibrated confidence and an audit trail, instead of a naive binary.

---

## 1. Data approach

You have no real Razorpay fraud data, and you must be transparent about that. The plan is a **two-layer data strategy** — public data for statistical rigour, synthetic Razorpay-shaped data for the demo.

### Layer A — public dataset (for the modelling rigour)

Pick **one** primary, and say clearly in the README which it is and why:

| Dataset | What it is | Why / why not |
|---|---|---|
| **IEEE-CIS Fraud Detection** (Kaggle) | ~590k e-commerce transactions, rich features (device, card, email domain, many engineered), ~3.5% fraud | **Recommended primary.** Closest to a payment-gateway setting, realistic feature richness, has timestamps for temporal splits. |
| **Credit Card Fraud (ULB)** | 284k transactions, PCA-anonymised features, 0.17% fraud | Good *secondary* for demonstrating extreme imbalance, but features are anonymised (V1..V28) so no explainability story. |
| **PaySim / Sparkov** | Synthetic mobile-money simulators | Useful for generating volume and for the streaming demo, not for headline claims. |

**Decision:** IEEE-CIS as primary (rich + timestamped + explainable features), ULB as an extreme-imbalance stress test in an appendix.

### Layer B — synthetic Indian-payments stream (for the demo + API shape)

Generate a transaction stream shaped like Razorpay's real domain, so the demo isn't obviously a Kaggle notebook:

- **Payment methods:** UPI, cards, netbanking, wallets, EMI — with different base rates. (Note the domain reality: cards carry chargebacks; UPI fraud is largely social-engineering/APP fraud, not card-not-present. Model them as separate risk regimes — this detail alone signals domain understanding.)
- **Merchant categories** with different fraud base rates and margins (electronics vs fashion vs digital goods vs travel).
- **Entities:** customer id, device fingerprint, IP/geo, card BIN, email domain, order value, timestamp.
- **Injected fraud patterns:** card testing (many small txns fast), account takeover (new device + high value), friendly fraud / chargeback abuse, refund abuse, velocity bursts.
- Wire it to **Razorpay test-mode APIs** so payloads match the real object shape.

**Be explicit in the README:** which numbers come from the public dataset (the real, defensible metrics) and which from synthetic data (the live demo). Judges reward that honesty; blurring it is the fastest way to lose credibility.

---

## 2. Feature engineering

The rule that governs everything: **every feature must be computable at decision time.** No feature may use information from after the transaction. This is where most fraud projects silently leak.

**Feature families:**

- **Transaction-level:** amount, amount vs merchant's typical ticket size, payment method, currency, hour-of-day, day-of-week, is-odd-hour.
- **Velocity (the highest-signal family):** count/sum of transactions from this card / device / IP / email in the last 1m, 10m, 1h, 24h, 7d. Card testing and bursts live here.
- **Entity history:** age of customer account, days since first seen, historical chargeback rate for this card/device (computed **only from data before this transaction**).
- **Mismatch / consistency:** billing vs shipping geo distance, BIN country vs IP country, email domain vs name, new device for known customer.
- **Merchant context:** merchant category, merchant's own base fraud rate, merchant's average order value.
- **Derived ratios:** amount / customer's historical mean, txn count / account age.

**Leakage traps to avoid explicitly (and say so in the README):**
- Never compute "customer's total chargebacks" over the whole dataset — only over the past.
- Never use fields only populated after settlement/dispute.
- Watch for identifier features that memorise the test set.

---

## 3. Evaluation protocol (get this right or nothing else counts)

**Temporal split, never random.** Fraud is non-stationary — patterns drift, and a random split lets the model see the future. Split by time:

- **Train:** earliest ~60% of the time range
- **Validation:** next ~20% (threshold + calibration tuned here)
- **Test:** final ~20% (out-of-time, touched once)

Report that you did this prominently. A random split inflates every number, and a judge who knows fraud will check.

**Second-order rigour (cheap, high signal):**
- **Frozen splits + fixed seed**, documented.
- Report metrics **per payment method and per merchant category**, not just globally — a global number hides that you're failing on UPI or on one merchant type.
- A short **drift check**: does fraud rate / feature distribution shift between train and test windows? Say what you found.

---

## 4. The baseline ladder (your signature move)

Before any model, establish what "free" looks like. Report all of these on the same test set:

| Baseline | What it uses |
|---|---|
| Predict "never fraud" | nothing — shows the accuracy trap (~96–99% accuracy, 0 recall) |
| Random at base rate | nothing |
| Single-rule: amount > threshold | one feature |
| Simple velocity rule (>N txns per card per hour) | hand-written domain rule |
| Logistic regression on 5 features | cheap linear model |
| **Your model** | full pipeline |

Then report the **novelty margin**: of the headroom available above the best simple baseline, how much did the model actually capture? This is exactly the "honest metrics" the judges asked for, and it will be the single most memorable table in your submission.

**Expect a real finding:** a good velocity rule is often startlingly strong. If your model only marginally beats it, *say so* — that honesty will score better than a hidden weakness.

---

## 5. The model

**Use gradient-boosted trees (XGBoost or LightGBM). Not deep learning.** On tabular fraud data GBDTs beat neural nets, train in minutes, and give you SHAP explanations for free. Choosing the right tool and justifying it is a maturity signal; reaching for a transformer here is a red flag.

**Imbalance handling — and the subtle point that will impress:**
- Use `scale_pos_weight` / class weights rather than aggressive oversampling.
- **Critical:** resampling (SMOTE, undersampling) **destroys probability calibration**. If your predicted "0.9" no longer means "90% likely fraud," your entire rupee-cost thresholding is built on sand. If you resample at all, you **must recalibrate afterwards**.
- State this explicitly in the README — most entries will SMOTE blindly and never notice their probabilities are meaningless.

**Calibration step (required, not optional):**
- Fit **isotonic regression** (or Platt scaling) on the validation set.
- Show a **reliability diagram** before and after, plus **Expected Calibration Error (ECE)**.
- Justification: cost-optimal thresholding requires *true probabilities*. Calibration is what makes the money math valid.

---

## 6. The cost model — the heart of the project

Define, in rupees, what each error actually costs. Put these assumptions in a config file and in the README so they're auditable.

**False negative (fraud approved):**
- chargeback amount (transaction value)
- chargeback fee (flat, e.g. ₹500–1,500)
- goods/service already delivered
- dispute-handling ops time
- penalty risk if the merchant's chargeback ratio breaches scheme thresholds

**False positive (legit customer blocked):**
- lost margin on that order (not full order value — **margin**; a subtle detail worth calling out)
- customer-churn probability × lifetime value
- support contact cost
- reputational/repeat-purchase damage

**Then:**
1. Sweep the decision threshold from 0 → 1.
2. At each threshold compute **total expected cost = (FN count × FN cost) + (FP count × FP cost)**.
3. Plot the **cost curve** and mark: the F1-optimal threshold, the accuracy-optimal threshold, and the **cost-optimal threshold**.
4. **Report the rupee gap between them.** That number is your headline.

**The likely headline:** *"Optimising F1 instead of cost would have cost this merchant ₹X per 10,000 transactions."*

**Then go one better — per-merchant thresholds.** An electronics merchant (high ticket, high fraud, thin margin) and a digital-goods merchant have different cost structures, so they have different optimal thresholds. Show that one global threshold is wrong for both, and that per-merchant calibration recovers ₹Y. This is a genuinely novel angle for a hackathon.

---

## 7. Decision policy — three outcomes, not two

Binary approve/block is what everyone builds. Do this instead:

```
   score < low_threshold        →  AUTO-APPROVE
   low ≤ score < high_threshold →  MANUAL REVIEW   (bounded queue)
   score ≥ high_threshold       →  AUTO-BLOCK
```

- The **review band** is sized by cost: widen it where the model is uncertain and the cost of being wrong is high; narrow it where review capacity is limited.
- Report **review rate** as a first-class metric — a system that sends 40% to review is useless operationally, even at perfect precision. Target something defensible (e.g. <5%).
- This is your **"graceful failure"** story: when the model is unsure, it escalates rather than guessing with money.

**Bounded actions (the judges asked for this):** the agent should never move money irreversibly on its own. Block and review are reversible; every auto-block is appealable and logged. State the blast radius explicitly.

---

## 8. Metrics to report

Lead with these; skip vanity numbers.

| Metric | Why |
|---|---|
| **PR-AUC (average precision)** | The right ranking metric under heavy imbalance. **Not ROC-AUC** — ROC looks great on imbalanced data and hides failure. Say why you chose it. |
| **Precision & recall at the chosen operating point** | With the threshold explicitly stated. |
| **Recall at a fixed low FPR** (e.g. FPR = 0.5%) | Operationally meaningful: "how much fraud do we catch while blocking ≤0.5% of good customers." |
| **Rupees saved vs baselines** | The headline. Model vs velocity-rule vs no-model. |
| **Cost per 10,000 transactions** at F1-optimal vs cost-optimal threshold | The core thesis, quantified. |
| **ECE + reliability diagram** | Proves the probabilities mean something. |
| **Review rate** | Operational viability. |
| **Per-segment breakdown** (payment method, merchant category) | Shows where it fails. |

Report **accuracy exactly once**, next to the "predict never-fraud" baseline, to demonstrate why you're not using it. That single juxtaposition tells the judges everything about how you think.

---

## 9. Explainability & audit trail

- **SHAP values** per decision → convert the top 3 contributing features into human-readable **reason codes** ("4 transactions on this card in 6 minutes", "first time on this device", "billing/IP country mismatch").
- **Audit log**: for every decision store transaction id, model version, score, threshold used, decision, reason codes, timestamp, and the config/cost-model version in force. Immutable, queryable.
- **Appeal path**: a reviewed/blocked transaction can be overturned, and the override is logged as labelled feedback.
- **Merchant-facing explanation** vs **internal explanation** — don't leak the exact rule set to the outside, or fraudsters learn it. Mention this; it's a real production consideration.

---

## 10. System architecture

Keep it simple and real. This also reuses the FastAPI work you're already learning.

```
  txn ──▶ POST /v1/score  (FastAPI)
              │
              ├─ feature builder  (velocity counters ← Redis)
              ├─ model (GBDT, loaded once at startup)
              ├─ calibrator (isotonic)
              ├─ policy engine (per-merchant thresholds from config)
              └─ audit logger ──▶ Postgres
              │
              └──▶ { decision, score, confidence, reason_codes, threshold_used }

  /v1/feedback   ← label arrives later (chargeback/appeal) → feeds retraining
  /v1/metrics    ← live precision/recall/cost/review-rate
  dashboard      ← cost curve, reliability diagram, per-merchant thresholds, decision log
```

- **Redis** for velocity counters (must be fast, per-request).
- **Postgres** for the audit log and feedback labels.
- **Latency budget:** payment decisions must be quick — target **p95 < 100ms** and report it. Latency is in the judging criteria's spirit ("production... where latency matters").
- **Graceful degradation:** if the model service fails or times out, fall back to the deterministic velocity rule rather than failing open or blocking everything. Say this explicitly — it's the "exception handling" criterion, and almost no one will have it.

---

## 11. The dashboard / demo

Build a small dashboard showing:
- Live transaction stream with decisions and reason codes
- The **cost curve** with the three thresholds marked (this is the money shot)
- Reliability diagram before/after calibration
- The baseline-ladder table
- Per-merchant threshold panel
- Review queue

---

## 12. The 5-minute video (structure it deliberately)

- **0:00–0:30** — The claim: "Fraud models are judged on F1. Merchants pay in rupees. Those disagree, and I'll show you by how much."
- **0:30–1:30** — Live demo: transactions flowing, one auto-approved, one escalated to review with reason codes, one blocked. Show the audit log entry.
- **1:30–3:00** — The numbers: baseline ladder table (including "predict never-fraud scores 96.5% accuracy"), PR-AUC, precision/recall at operating point.
- **3:00–4:00** — **The cost curve.** F1-optimal vs cost-optimal threshold, and the rupee gap. Then per-merchant thresholds and the extra savings.
- **4:00–4:30** — Calibration: reliability diagram before/after, and why it matters for the money math.
- **4:30–5:00** — Failure handling: review band, fallback rule, appeal path. Close on limitations, honestly stated.

**End on limitations, not a victory lap.** Given the rubric explicitly asks for honest metrics, stating where it fails is a scoring move, not a weakness.

---

## 13. Build timeline (assuming ~2 weeks part-time)

| Days | Work |
|---|---|
| 1–2 | Data: load IEEE-CIS, temporal splits, EDA, base rates per segment |
| 3–4 | Feature engineering + leakage audit; **baseline ladder** run and tabulated |
| 5–6 | GBDT training, class weighting, PR-AUC, per-segment metrics |
| 7 | Calibration (isotonic) + reliability diagrams + ECE |
| 8–9 | **Cost model + cost curve + per-merchant thresholds** (the core contribution) |
| 10–11 | FastAPI service: /score, Redis velocity, audit log, fallback rule, latency test |
| 12 | Synthetic Razorpay-shaped stream + test-mode API wiring |
| 13 | Dashboard |
| 14 | README, architecture doc, video |

If time runs short, **cut the dashboard and the synthetic stream — never the cost model or the baseline ladder.** Those two are the entire differentiation.

---

## 14. Gotchas that will sink other entries (and shouldn't sink yours)

- **Random train/test split** → inflated metrics. Use temporal.
- **ROC-AUC on 0.5% positives** → looks amazing, means little. Use PR-AUC.
- **SMOTE without recalibration** → probabilities become meaningless, cost math invalid.
- **Leaky features** (post-hoc chargeback fields, future aggregates).
- **Reporting accuracy** as a headline on imbalanced data.
- **Ignoring review rate** → a "great" model that flags 30% of traffic is unusable.
- **Unbounded automated actions** → the rubric explicitly wants bounded, reversible, audited money actions.
- **No failure path** → what happens when the model service is down?

---

## 15. README structure (judges read this first)

1. One-paragraph thesis: accuracy vs rupees.
2. The baseline-ladder table (lead with it — it's your hook).
3. Data provenance: what's public, what's synthetic, stated plainly.
4. Evaluation protocol: temporal split, frozen seeds.
5. Metrics table with the operating point stated.
6. **The cost curve figure + the rupee headline.**
7. Calibration evidence.
8. Architecture diagram + latency numbers + fallback behaviour.
9. Audit-trail example (a real logged decision).
10. **Limitations** — synthetic demo data, no real chargeback labels, cost assumptions are estimates, per-merchant thresholds need real merchant data to validate.

That last section is the one most people omit and the one this rubric most rewards.