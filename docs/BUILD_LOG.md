# Build log

A running record of what was done, what was found, and why each decision was made. The final documentation is written from this file.

## Day 1 (10 Nov 2026)

### Setup

- Project created with `uv` (Python 3.12). PyTorch is installed in its CPU-only build to keep the install small; there is no GPU.
- Dataset downloaded from Hugging Face: `Tobi-Bueck/customer-support-tickets`, 61,765 tickets, licence CC BY-NC 4.0.

### What the data looks like

Explored with DuckDB SQL queries.

| Finding | Number |
|---|---|
| German tickets | 33,504 |
| English tickets | 28,261 |
| Distinct labels in `queue` | 52 |
| Distinct ticket bodies | 53,364 of 61,765 (many repeats) |
| Tickets without a subject | 5,299 |

The 52 labels are two different things mixed together:

- **10 company departments** (Technical Support, Billing and Payments, and so on), in German and English, 48,587 tickets.
- **42 topic categories** (Sports, Real Estate, Pets, and so on), German only, about 300 each. These are topics, not teams.

### Decisions

| Decision | Reason |
|---|---|
| Keep only the 10 departments | Routing to a team is the business problem. "Sports" is not a team. |
| Join subject and body into one text | Both carry signal; 9% of tickets have no subject. |
| Remove duplicate texts | The same text in train and test would inflate the score (leakage). |
| Remove texts that appear with two different labels | The label cannot be trusted. |
| Four splits: train 60%, calibration 15%, validation 10%, test 15% | Each split has one job, so the final numbers are measured on data that nothing was tuned on. |
| Assign splits by a hash of the text | The split never changes between runs or machines. |

After cleaning: **40,275 tickets** (16,531 German, 23,744 English).

### First baseline: word counts (TF-IDF) + logistic regression

Measured on the validation split.

| Metric | Value |
|---|---|
| Accuracy, all | 53.6% |
| Accuracy, German | 45.7% |
| Accuracy, English | 59.2% |
| Macro-F1 | 0.434 |
| Always guessing the largest team | 28.1% |

### What this means

- The task is hard and the labels are noisy. Example: a ticket asking about secure hospital data management is labelled "Billing and Payments". The texts read as machine-generated.
- 54% accuracy would be useless as a fully automatic router. That is the case for abstention: the question becomes how accurate the system is on the tickets it chooses to answer.
- German is clearly harder than English in this data. This will be reported separately everywhere.

### Next

- Compare the embedding model against this baseline.
- Calibrate, then build the abstention rule.

## Day 1, continued: the design changed

### Embedding classifier: worse than the baseline

| Model (test set) | Accuracy |
|---|---|
| Logistic regression on embeddings (the planned model) | 40.3% |
| Word counts (TF-IDF) + logistic regression | 52.6% |
| Always the largest team | 29.3% |

With the planned model, a 10% error limit allowed only 6% of tickets to be routed. Computing the embeddings took 22 minutes on the laptop CPU.

### Why: near-repeated tickets

A simple model beating a stronger one is a warning sign, so the next step was to look at each validation ticket's most similar training ticket.

- Routing a ticket like its single most similar past ticket gave 69.3% on validation, better than every classifier.
- When a near copy exists, it almost always has the same team. When nothing similar exists, every model is at 37–40%, barely above guessing.

Conclusion: the team labels are predictable only through resemblance to earlier tickets. A classifier averages over all tickets and blurs exactly that signal.

### New design

| Decision | Reason |
|---|---|
| Route like the most similar past ticket (nearest neighbour on embeddings) | Best accuracy (70.1% on test) and every decision can show its reason |
| Confidence from a small logistic model on 3 signals: closest similarity, agreement among 5 neighbours, gap between 1st and 5th neighbour | Raw similarity is not a probability (calibration error 0.27). The fitted model is calibrated (0.021) and ranks decisions better |
| Guaranteed-risk threshold instead of conformal prediction sets | The business promise is "at most X% wrong among routed tickets". Conformal sets guarantee something else (that a set of teams contains the right one) |
| Error limits offered: 1%, 2%, 5%, 10%; default 5% | 15% and 20% gave the same result as 10% once the novelty gate was added |

Confidence models compared (fitted on validation, checked on calibration):

| Confidence score | Area under risk–coverage curve (lower is better) | Calibration error |
|---|---|---|
| Raw similarity | 0.103 | 0.278 |
| Isotonic regression on similarity | 0.103 | 0.013 |
| Logistic model on similarity + agreement | 0.100 | 0.068 |
| **Logistic model on similarity + agreement + gap (chosen)** | **0.083** | **0.019** |
| Gradient boosting with monotonic constraints | 0.081 | 0.010 |

Gradient boosting was marginally better but harder to explain and to store; the logistic model is three numbers and an intercept.

### A flaw found by trying it

The first live request, "Hallo, es funktioniert nicht. Können Sie mir bitte helfen?", was routed to Billing at 75% confidence. Its closest past ticket had similarity 0.92, in the range where proposals are right only 37% of the time. The confidence model had never seen such short, generic messages and extrapolated badly.

Fix: a **novelty gate**. If the closest past ticket is below 0.964 similarity, hand over regardless of the confidence. 0.964 is the similarity at which the closest ticket becomes right more often than wrong on validation data. The thresholds are chosen with the gate active, so the guarantee still holds. Cost: at the 5% limit, coverage went from 56.7% to 54.1%. Four tests now check that vague or nonsense messages are handed over even at the most lenient setting.

### Final numbers (test set, 6,164 tickets)

| Error limit | Routed | Wrong among routed |
|---|---|---|
| 1% | 41.4% | 0.2% |
| 2% | 49.1% | 1.0% |
| 5% | 54.1% | 3.7% |
| 10% | 58.7% | 8.5% |

German is weaker than English: at the 5% limit, German alone is at 5.5% errors with 37.8% routed; English 3.0% with 65.5% routed.

### Engineering

- FastAPI service (`/health`, `/info`, `/route`) and Streamlit page that talks to it over HTTP.
- 27 tests: abstention logic, API behaviour, and a regression test tied to the recorded metrics. Lint with ruff.
- Response time on the laptop: median 157 ms, 95th percentile 225 ms (300 real test tickets).
- Docker: first build failed because the working folder belonged to root; fixed by creating it for the non-root user. Image is 3.46 GB (PyTorch 0.8 GB, embedding model files 0.47 GB); memory use 832 MB. Training-only libraries were moved to a separate dependency group to keep them out of the image.
- The container was run locally: page reachable, health check passing, both a routed and a handed-over request answered correctly.

### Not done yet

- Deployment to a public host and pushing to GitHub (needs the account owner's go-ahead).
- CI has been written but has not run on GitHub yet.
- Demo video.
- Optional extras: LLM-drafted reply, thresholds per language, drift monitor.

## Day 2 (11 Nov 2026): published, then made small enough for a free host

The numbers in the sections above are from the first, full-precision model. The final numbers are at the end of this section.

### Published

- Code pushed to github.com/mimjamam/handover (public, MIT licence for the code; data and model file stay CC BY-NC 4.0). CI passed on the first run.
- Hugging Face refused a Docker Space: that tier now needs a paid subscription. Render's free tier was chosen instead; it allows 512 MB of memory, and the service used 832 MB.

### Slimming, measured step by step

| Step | Memory of the running service |
|---|---|
| Start: PyTorch, Streamlit page, separate API process | 832 MB |
| ONNX Runtime + 8-bit quantised embedding model, page served by the API (one process) | Killed at start-up under a 512 MB cap |
| Measured where it went: `tokenizers` library about 300 MB, model session about 420 MB with that set-up | |
| Plain SentencePiece instead of `tokenizers` (about 50 MB) | **277 MB**, stable over 300 requests under a 512 MB cap |

Image size went from 3.46 GB to 751 MB.

### Two problems found on the way

1. **Batch-dependent vectors.** The quantised model returned slightly different vectors for the same text depending on its batch (cosine about 0.995). Training used batches, the service embeds single messages. The test "a stored ticket sent through the API finds itself" failed with similarity 0.996. Fix: embed every text on its own everywhere; new test in `tests/test_embed.py`; all embeddings recomputed (9 minutes) and the model re-evaluated.
2. **Tokenizer swap.** SentencePiece and `tokenizers` were compared on all 40,275 cleaned tickets: identical tokens for every one, so no further recomputation was needed.

### Final numbers (test set, 6,164 tickets, quantised model, one text at a time)

| Error limit | Routed | Wrong among routed | German routed / wrong | English routed / wrong |
|---|---|---|---|---|
| 1% | 40.5% | 0.3% | 24.4% / 0.3% | 51.8% / 0.3% |
| 2% | 47.1% | 0.8% | 30.8% / 1.3% | 58.6% / 0.7% |
| 5% | 52.6% | 3.1% | 36.5% / 4.3% | 64.0% / 2.6% |
| 10% | 58.1% | 8.8% | 43.7% / 14.5% | 68.2% / 6.3% |

Accuracy without handover 69.6% (was 70.1% with the full-precision model). Calibration error 0.018. Novelty gate 0.965. 30 tests pass. In the container on half a CPU core: median response 283 ms, 95th percentile 304 ms.

### Still open

- Deployment on Render (needs the account owner to connect the repository).
- Demo video.
- Thresholds per language, drift monitoring, LLM-drafted replies.
