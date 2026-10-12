## ModelRoute eval

1500 gold-labeled + 6000 large-model-labeled training tickets · 1000 validation · 2000 test tickets (no text overlap with training) · 12 intents

| System | Accuracy | Macro-F1 | Quality vs large | Fallback rate | p50 latency | p95 latency | Cost / 1K requests |
|---|---|---|---|---|---|---|---|
| Large model only | 91.9% | 0.914 | 100.0% | 100.0% | 711.04 ms | 1445.45 ms | $0.8912 |
| Small, gold labels only | 91.6% | 0.911 | 99.7% | 0.0% | 0.04 ms | 0.07 ms | $0.0000 |
| Small, distilled | 92.0% | 0.914 | 100.1% | 0.0% | 0.01 ms | 0.03 ms | $0.0000 |
| Router (distilled small -> large) | 91.9% | 0.914 | 100.0% | 0.1% | 0.01 ms | 0.03 ms | $0.0009 |

**Router vs. large model only:** cost per 1K requests $0.8912 → $0.0009, p95 latency 1445 ms → 0.03 ms, quality retained 100.0%.

Routing threshold 0.300 on calibrated confidence (temperature 0.70), chosen on validation to keep ≥ 99% of the large model's accuracy (validation: router 92.0% vs large 91.9%, fallback 0.0%).

Small model: word 1-2grams, 2^15 dims, 125,880 non-zero parameters. Large model: large (offline stand-in) (measured compute p50 4.32 ms; latency above uses a simulated hosted-API profile, median 700 ms).

**Distribution shift** (1000 tickets in casual phrasing never seen in training):

| System | Accuracy | Macro-F1 | Fallback rate |
|---|---|---|---|
| Large model only | 59.7% | 0.588 | 100.0% |
| Small, gold labels only | 50.9% | 0.493 | 0.0% |
| Small, distilled | 50.9% | 0.484 | 0.0% |
| Router (distilled small -> large) | 54.0% | 0.520 | 14.7% |

Threshold sweep (what an operator tunes): fallback rate / accuracy / cost per 1K

| Threshold | In-distribution | Shifted |
|---|---|---|
| 0.30 | 0% / 91.9% / $0.001 | 15% / 54.0% / $0.132 |
| 0.50 | 1% / 91.9% / $0.008 | 46% / 58.1% / $0.411 |
| 0.60 | 3% / 91.9% / $0.032 | 64% / 58.7% / $0.567 |
| 0.70 | 6% / 91.9% / $0.057 | 77% / 58.6% / $0.684 |
| 0.80 | 10% / 92.0% / $0.089 | 85% / 58.7% / $0.760 |
| 0.90 | 15% / 91.9% / $0.137 | 91% / 59.2% / $0.814 |
| 0.95 | 21% / 91.9% / $0.195 | 95% / 59.5% / $0.849 |
| 0.99 | 42% / 91.9% / $0.380 | 99% / 59.7% / $0.882 |

Per-intent F1, distilled small model:

| Intent | F1 |
|---|---|
| BILLING_REFUND | 0.937 |
| DOUBLE_CHARGE | 0.876 |
| LOGIN_ISSUE | 0.907 |
| PASSWORD_RESET | 0.916 |
| SHIPPING_DELAY | 0.953 |
| ADDRESS_CHANGE | 0.899 |
| DAMAGED_ITEM | 0.924 |
| CANCEL_SUBSCRIPTION | 0.907 |
| UPGRADE_PLAN | 0.940 |
| BUG_REPORT | 0.910 |
| FEATURE_REQUEST | 0.915 |
| INVOICE_REQUEST | 0.890 |
