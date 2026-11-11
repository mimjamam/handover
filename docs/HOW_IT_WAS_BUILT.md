# How Handover was built

This document explains what was built, how, and why, in the order it happened. Read it with the code open. Every number comes from `reports/metrics.json` or `reports/latency.json`.

## 1. The problem

A company receives customer messages and must send each one to the right team. Automating this saves time, but a wrongly routed message costs more than a slowly routed one. So the useful question is not "how accurate is the model?" but "which messages can it be trusted with, and how often is it wrong on those?"

Handover answers that. For each message it either routes it, or hands it to a person. The share of wrong automatic decisions stays under a limit the operator chooses.

## 2. The data

**Source.** `Tobi-Bueck/customer-support-tickets` on Hugging Face: 61,765 tickets in German and English, each with a subject, body and a `queue` label. Licence CC BY-NC 4.0 (non-commercial).

**What was found** (explored with SQL in DuckDB):

- The 52 `queue` values are two different things. 10 are company departments such as "Billing and Payments", in both languages. 42 are topics such as "Sports", German only, about 300 tickets each.
- 8,401 ticket bodies are repeats of another ticket.
- 5,299 tickets have no subject.

**Cleaning** (`src/handover/data.py`):

| Step | Why | Tickets left |
|---|---|---|
| Keep the 10 departments | A topic is not a team; routing means departments | 48,587 |
| Join subject and body | Both carry meaning; some tickets have no subject | |
| Remove exact duplicates | The same text in training and test data would make the score look better than it is. This is called leakage | |
| Remove texts that appear with two different teams | The label is unreliable | 40,275 |

Result: 16,531 German and 23,744 English tickets.

**Splitting.** Four parts, each with one job:

| Split | Share | Tickets | Job |
|---|---|---|---|
| Train | 60% | 24,091 | The memory the router searches |
| Validation | 10% | 4,052 | Fit the confidence model and the novelty gate |
| Calibration | 15% | 5,968 | Choose the thresholds |
| Test | 15% | 6,164 | Final numbers, used once at the end |

A ticket's split is decided by a hash of its text. The same ticket always lands in the same split, on any machine, so results can be reproduced exactly.

Why four splits and not two: every time data is used to make a choice, the model "learns" a little from it, and a score on that data becomes too optimistic. Keeping the thresholds on data that nothing else touched is what makes the error guarantee valid.

## 3. First attempt, and why it failed

**Plan.** Turn each ticket into a vector with an embedding model, train logistic regression on the vectors.

- Embedding model: `intfloat/multilingual-e5-small`. It maps text to 384 numbers so that texts with similar meaning get similar vectors. One model handles German and English. It is small enough for a CPU. It is used as-is, not fine-tuned.
- Computing vectors for all 40,275 tickets took 22 minutes on the laptop with PyTorch (9 minutes with the lighter runtime used in the end; see section 6a).

**Result on the test set:**

| Model | Accuracy |
|---|---|
| Logistic regression on embeddings | 40.2% |
| Word counts (TF-IDF) + logistic regression | 52.6% |
| Always choosing the largest team | 29.3% |

With abstention added, the embedding classifier could route only 6% of tickets at a 10% error limit. That is not a useful product.

**Diagnosis.** A simpler model beating a stronger one usually means something unusual in the data. For each validation ticket, I looked up its most similar training ticket:

| Closest past ticket (test set) | Share of tickets | Closest ticket has the right team |
|---|---|---|
| Near copy, similarity ≥ 0.985 | 24.3% | 99.7% |
| Similar, 0.97–0.985 | 26.2% | 89.7% |
| Loosely similar, 0.95–0.97 | 32.8% | 47.9% |
| Nothing similar, below 0.95 | 16.7% | 36.6% |

Two conclusions:

1. About half of all tickets are close variants of an earlier ticket and go to the same team.
2. For tickets with nothing similar, the team is almost unpredictable from the text. On those, embeddings with logistic regression reach 38.8%, word counts 37.2%, and guessing the largest team 29.4%.

The word-count model was winning because it partly memorised the repeated tickets. The labels in this dataset are noisy: for example, a ticket about using TensorFlow for investment analytics is labelled "Customer Service".

## 4. The design that works

If the only reliable signal is "this looks like a ticket we have seen", the system should use that signal directly and say so.

**Step 1: propose** (`Router.propose` in `src/handover/model.py`). Compare the new message's vector with all 24,091 stored vectors using cosine similarity (a dot product, because the vectors have length 1). Take the 5 most similar. Propose the team of the single closest one. This is a nearest-neighbour classifier. Accuracy if it always answered: 69.6% on test.

Why the closest one and not a vote of five: the single closest was more accurate on validation (69.3% against 67.4% for a vote of 5). A near copy is strong evidence; four looser neighbours can outvote it wrongly.

**Step 2: estimate confidence.** Similarity is not a probability. A similarity of 0.97 does not mean "97% likely right". Its calibration error was 0.27. So a small logistic regression is fitted on the validation split to predict "is the proposal right?" from three signals:

| Signal | Meaning | Learned weight |
|---|---|---|
| (1 − closest similarity) × 100 | How far the closest past ticket is | −0.201 |
| Agreement | Share of the 5 neighbours with the same team as the closest | +2.428 |
| (1st − 5th similarity) × 100 | How much the closest stands out from the rest | +1.573 |
| Intercept | | −1.888 |

The signs make sense: farther away lowers confidence; agreement and a clear stand-out raise it. Because this model is fitted to real right/wrong outcomes on held-out data, its output behaves like a probability. Calibration error on test: 0.018.

**Step 3: novelty gate.** Testing the running service showed a flaw. "Hallo, es funktioniert nicht. Können Sie mir bitte helfen?" was routed to Billing at 75% confidence, with a closest similarity of only 0.92. The confidence model had only ever seen tickets like the training tickets; on a short, generic message it extrapolated.

The fix is a hard rule: if the closest past ticket is below similarity 0.965, hand over, whatever the confidence says. The value 0.965 is where the closest ticket becomes right more often than wrong on validation (found with isotonic regression, which fits a rising step curve of accuracy against similarity). 40.6% of test tickets fall below the gate.

**Step 4: choose the threshold with a guarantee** (`choose_threshold` in `src/handover/abstain.py`). The promise: among routed tickets, at most α are wrong.

- Take the calibration split. For a candidate threshold, count how many tickets would be routed (n) and how many of those are wrong (k).
- k/n is only an estimate. The Clopper–Pearson bound gives a value that the true error rate stays below with 95% probability.
- Try thresholds from strict to lenient. Keep going while the bound is under α. Stop at the first failure. This order is called fixed-sequence testing, and it is what allows trying many thresholds without weakening the guarantee.
- The lowest threshold that passed is the answer.

This is selective classification with guaranteed risk (Geifman and El-Yaniv, 2017), with the testing procedure from "Learn then Test" (Angelopoulos et al., 2021).

**Why not conformal prediction, as first planned.** Standard conformal prediction returns a set of labels and guarantees that the set contains the right one. The business promise here is about the error rate among the decisions taken, which is a different quantity. I used the method that matches the promise.

## 5. Results

On the 6,164 test tickets:

| Error limit | Threshold | Routed | Wrong among routed | Guarantee kept |
|---|---|---|---|---|
| 1% | 0.932 | 40.5% | 0.3% | Yes |
| 2% | 0.806 | 47.1% | 0.8% | Yes |
| 5% (default) | 0.559 | 52.6% | 3.1% | Yes |
| 10% | 0.253 | 58.1% | 8.8% | Yes |

Coverage cannot exceed 59.4% because of the novelty gate.

**By language at the 5% limit:**

| | Routed | Wrong among routed |
|---|---|---|
| English | 64.0% | 2.6% |
| German | 36.5% | 4.3% |

German is weaker throughout (accuracy 60.0% against 76.3%) and is routed far less often. At the 10% limit the German-only error is 14.5%, above the limit. The guarantee is for all tickets together, not per language. This is the most important limitation and it is stated in the README and model card.

**Other metrics:**

| Metric | Value | Meaning |
|---|---|---|
| Area under the risk–coverage curve | 0.095 | 0.295 if confidence were random, so confidence is informative |
| Macro-F1 | 0.674 | Average quality across the 10 teams, small teams counted equally |
| Latency, median / 95th percentile | 283 / 304 ms | 300 real test tickets, in the container limited to half a CPU core and 512 MB |

## 6. The service

**Shape.** One container, one process. The core logic has no web code. The API imports the core and also serves the demo page.

| Part | File | What it does |
|---|---|---|
| API | `src/handover/api.py` | FastAPI. `GET /` (the page), `GET /health`, `GET /info` (metrics and settings), `POST /route` (decision, confidence, reason, the matched past ticket, response time). Loads the model once at start-up. Rejects empty and over-long input |
| Page | `src/handover/static/index.html` | One HTML file with plain JavaScript. Buttons for the error limit, a text box with real test tickets as examples, the decision with its reason, and a section on limits. It calls `/info` and `/route` from the browser |
| Model file | `artifacts/router.joblib` | 20 MB. Stored vectors at half precision, teams, ticket texts, confidence weights, gate, thresholds, and the test metrics |
| Training | `scripts/train.py` | Builds everything above and writes `reports/metrics.json` and the figure |

Training runs offline. The service only loads the file, so the running model is exactly the one that was evaluated.

## 6a. Making it fit a free host

The first version used PyTorch, a Streamlit page and a separate API process. It needed 832 MB of memory and a 3.46 GB image. The free host allows 512 MB, so the service was rebuilt to be smaller. Each step was measured.

| Change | Effect |
|---|---|
| PyTorch replaced by ONNX Runtime with the 8-bit quantised version of the same embedding model (118 MB instead of 470 MB) | No PyTorch in the image |
| Streamlit page replaced by one HTML file served by the API | One process instead of two |
| `tokenizers` library replaced by plain SentencePiece | Tokenizer memory from about 300 MB to about 50 MB |
| Training-only libraries (pandas, scikit-learn, SciPy, matplotlib) moved to a separate dependency group | Not installed in the image |
| **Result** | **Memory 832 MB → 277 MB; image 3.46 GB → 751 MB** |

**Two problems this caused, and how they were caught.**

1. *The quantised model is not consistent across batches.* The same text got a slightly different vector depending on which other texts were in its batch (cosine 0.995 instead of 1.0). Training embedded tickets in batches; the service embeds one message at a time. With thresholds at 0.965 and 0.985, a difference of 0.005 matters: the evaluated system and the deployed system would not have been the same. A test that looks up a stored ticket through the API failed (similarity 0.996 instead of above 0.999), which exposed it. Fix: every text is embedded on its own, everywhere, and `tests/test_embed.py` checks that a text gets the same vector alone or in a group. All embeddings were recomputed and the model re-evaluated; the numbers in this document are from that final run.
2. *A different tokenizer must give identical tokens.* SentencePiece and the `tokenizers` library were compared on all 40,275 cleaned tickets: zero differences. (They differ only on text with trailing whitespace, which the cleaning step and the API both strip.)

Quantisation cost little: accuracy 69.6% against 70.1% with the full-precision model, and coverage at the 5% limit 52.6% against 54.1%.

**Docker.** `python:3.12-slim`, dependencies installed with uv, the embedding model downloaded during the build, a non-root user, a health check. The port comes from the `PORT` variable that hosts such as Render set (7860 by default). Tested locally under a hard 512 MB memory cap and half a CPU core with 300 requests: memory stayed at 277 MB, median response 283 ms.

## 7. Tests and CI

30 tests, run with `uv run pytest`:

| File | What is checked |
|---|---|
| `tests/test_abstain.py` | The error bound behaves correctly; over 40 simulated datasets the chosen threshold keeps its promise on fresh data; a stricter limit routes fewer tickets; an impossible limit routes nothing |
| `tests/test_embed.py` | Vectors have length 1; a text gets the same vector alone or in a group |
| `tests/test_api.py` | The page is served; the endpoints work with the real model; a decision is consistent with its threshold; a stricter setting never routes what a looser one hands over; bad input is rejected; a ticket identical to a stored one finds it; vague and nonsense messages are handed over |
| `tests/test_regression.py` | The model file matches the report; accuracy, calibration and coverage have not dropped below recorded floors (`tests/floors.json`); every promised limit is kept on the test set |

`.github/workflows/ci.yml` runs lint (ruff) and all tests on every push. It runs on GitHub on every push.

## 8. What is not done

| Item | Status |
|---|---|
| GitHub repository with CI | Done: github.com/mimjamam/handover |
| Public deployment | Prepared for Render (`render.yaml`); see the README for the link once it is live |
| Demo video | Not recorded |
| Thresholds per language | Not built; the first improvement to make |
| Monitoring of the handover rate over time (drift) | Not built |
| LLM-drafted replies | Not built; the project contains no LLM |

## 9. What I would do with real company data

1. Measure the repeat rate first. It decides how much can be automated.
2. Use time-based splits: train on older tickets, test on newer ones, because real ticket streams change.
3. Set thresholds per language and per team where the cost of an error differs.
4. Add handed-over tickets and their human-assigned team back into the memory, so the system learns from every handover.
5. Watch the handover rate and the similarity distribution; a rise means the incoming tickets have changed and the threshold needs re-choosing.
6. Restrict who can see the matched past ticket, since it is customer text.
