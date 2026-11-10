# Model card: Handover ticket router

## Summary

| Item | Value |
|---|---|
| Task | Route a customer message to one of 10 teams, or hand it to a person |
| Languages | German, English |
| Type | Nearest-neighbour lookup on sentence embeddings, a logistic confidence model, and an abstention threshold |
| Embedding model | `intfloat/multilingual-e5-small` (frozen, not fine-tuned) |
| Built | November 2026 |
| Licence of the data | CC BY-NC 4.0 (non-commercial) |

## Intended use

A demonstration of selective prediction in a business process. It shows how to decide which messages are safe to route automatically and how to state the error rate of that automation.

**Not intended for:** real customer messages, any commercial use, or decisions about people. The teams and the training tickets come from a public, machine-generated dataset and do not match any real organisation.

## Data

Source: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), 61,765 tickets.

| Step | Tickets left |
|---|---|
| Raw | 61,765 |
| Keep the 10 company departments (drop 42 topic categories that are not teams) | 48,587 |
| Remove empty texts, exact duplicates, and texts that appear with two different teams | 40,275 |

| Split | Tickets | Used for |
|---|---|---|
| Train | 24,091 | The memory that is searched |
| Validation | 4,052 | Fitting the confidence model and the novelty gate |
| Calibration | 5,968 | Choosing the abstention thresholds |
| Test | 6,164 | Final numbers, used once |

Splits are assigned by a hash of the ticket text, so they are identical on every machine.

## Metrics (test set)

| Metric | Value |
|---|---|
| Accuracy without handover | 70.1% |
| Accuracy, English / German | 76.3% / 61.1% |
| Macro-F1 | 0.677 |
| Majority-class accuracy | 29.3% |
| Calibration error of confidence (ECE, 15 bins) | 0.021 |
| Area under the risk–coverage curve | 0.090 (0.284 if confidence were random) |

| Error limit | Coverage | Error among routed | English coverage / error | German coverage / error |
|---|---|---|---|---|
| 1% | 41.4% | 0.2% | 52.5% / 0.1% | 25.4% / 0.3% |
| 2% | 49.1% | 1.0% | 60.6% / 0.9% | 32.7% / 1.2% |
| 5% | 54.1% | 3.7% | 65.5% / 3.0% | 37.8% / 5.5% |
| 10% | 58.7% | 8.5% | 69.3% / 6.5% | 43.5% / 13.0% |

The error limit is guaranteed with 95% confidence for all tickets together, under the assumption that new tickets come from the same distribution as the calibration tickets.

## Known limits and risks

- **Language gap.** German tickets are routed less often and with more errors than English ones. At the 5% and 10% limits the German-only error rate is above the limit. The guarantee does not hold per language.
- **Depends on repeats.** 40.1% of test tickets fall below the novelty gate (similarity 0.964) and are always handed over. The system adds no value on messages unlike anything seen before; it only avoids harm there.
- **Label noise.** Team labels in the source data are often not explained by the text. On tickets with nothing similar in memory, no model tested was above 40% accuracy.
- **Machine-generated text.** Results will not transfer to real tickets without re-measuring.
- **Short or unusual input.** Very short messages have low similarity to everything and are handed over by the gate. This is tested.
- **Shows stored text.** The reason for each decision is a past ticket's text. With real data this would expose customer content.

## How to check it

`uv run pytest` runs a regression test that fails if any recorded metric drops, if a promised error limit is broken on the test set, or if the confidence score stops being informative. `reports/metrics.json` holds every number in this card.
