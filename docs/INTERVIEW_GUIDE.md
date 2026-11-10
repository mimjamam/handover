# Interview guide for Handover

Questions you are likely to get, with answers that use the real results. Learn the reasoning, not the wording. If you cannot explain an answer in your own words with the code open, go back to [HOW_IT_WAS_BUILT.md](HOW_IT_WAS_BUILT.md).

## Before any interview

- [ ] Run the demo yourself and try ten messages of your own
- [ ] Read all six files in `src/handover/` and `scripts/train.py` line by line
- [ ] Open `reports/metrics.json` and find every number quoted below
- [ ] Change the default error limit in `scripts/train.py`, retrain, and watch which tests react
- [ ] Explain `choose_threshold` aloud without looking

**On using AI tools.** This project was built with an AI coding assistant. Say so if asked; many teams in the postings use these tools daily (adesso and BCG name them in their student roles), and being caught hiding it is far worse than saying it. What an interviewer then checks is whether you understand and can defend the work. That is what the checklist above is for.

---

## The pitches

**30 seconds**

> I built a system that routes customer messages to the right team, in German and English. What makes it different is that it knows when it is unsure and hands those messages to a person. On held-out data, with a 5% error limit, it routes 54% of tickets automatically and is wrong on 3.7% of them; without handover it would be wrong on 30%. It runs as a tested API in Docker, and it applies the method from my paper on brain MRI classification.

**3 minutes**

1. **Problem.** A wrong routing costs more than a slow one, so the question is which messages are safe to automate.
2. **Demo.** One message that is routed, with the past ticket shown as the reason. One vague message that is handed over. Then move the slider.
3. **How.** Find the most similar past ticket, propose its team, estimate the chance of being right, answer only above a threshold chosen with a statistical guarantee.
4. **The finding.** My first model was a standard classifier and it reached 40%. I found that half the tickets are near-repeats and the rest are almost unpredictable, so I redesigned around that.
5. **Limits.** Machine-generated data, and German is weaker than English.
6. **Next.** Thresholds per language, time-based evaluation on real tickets, monitoring the handover rate.

---

## Project questions

**Walk me through the project.**
Use the 3-minute pitch.

**Why does abstention matter here?**
Without it the system is wrong on 29.9% of tickets, which no support team would accept. With a 5% limit it still handles 54.1% of the volume and is wrong on 3.7% of those. The value is in separating the tickets it can be trusted with from the ones it cannot.

**Why nearest neighbour and not a neural classifier?**
I started with logistic regression on embeddings: 40.3%. A word-count model got 52.6%, which was suspicious. When I checked, 27% of test tickets had a near copy in the training data with the right team 99.6% of the time, and tickets with nothing similar were at 37%. The predictable signal was resemblance to past tickets, so I used it directly: 70.1%. It also means every decision can show the ticket it was matched to.

**Is that not just data leakage?**
Exact duplicates and texts with conflicting labels were removed before splitting. What remains are near-repeats, which real support queues also have: many customers report the same issue in similar words. But in this dataset the repeats come from how it was generated, so the 54% automation rate describes this dataset, not a real company. I state that in the README. On real data I would measure the repeat rate first and use a time-based split.

**Why not fine-tune a transformer?**
No GPU, and computing embeddings alone took 22 minutes on my CPU. More important, on tickets with nothing similar in memory, two different models both stayed under 40%, close to the 29.7% from guessing. That points to label noise, which a bigger model does not fix.

**What does the model do with a message it has never seen anything like?**
It hands it over. If the closest past ticket is below 0.964 similarity, the novelty gate overrides the confidence score. I added it after the first live test: a vague German message was routed to Billing at 75% confidence. The confidence model had only seen tickets like the training data and extrapolated on short generic text. There are now tests for exactly those messages.

**How did you choose 0.964?**
It is the similarity at which the closest ticket becomes right more often than wrong on validation data. I fitted isotonic regression of correctness against similarity and took the point where it crosses 50%.

---

## Method questions

**What is calibration?**
A confidence score is calibrated if, among all cases where it says 80%, about 80% are right. Raw cosine similarity is not: its calibration error was 0.27. The confidence model's is 0.021.

**How is your confidence computed?**
A logistic regression on three signals: how far the closest past ticket is, how many of the five neighbours share its team, and how much the closest stands out from the fifth. It is fitted on a held-out split to predict whether the proposal is right.

**What exactly is guaranteed?**
With 95% probability over the draw of the calibration data, the error rate among routed tickets is at most the chosen limit, provided new tickets come from the same distribution as the calibration tickets.

**How do you get that guarantee?**
On the calibration split, for each candidate threshold I count the routed tickets and the errors among them, and compute a Clopper–Pearson upper bound on the true error rate. I test thresholds from strict to lenient and stop at the first one whose bound exceeds the limit. Testing in a fixed order and stopping at the first failure keeps the overall guarantee valid even though many thresholds are tried.

**What does it not guarantee?**
Three things. It is not per group: German tickets alone had 5.5% errors at the 5% limit. It does not hold if the incoming tickets change. And it is about the average over many tickets, not about any single decision.

**Why not conformal prediction?**
That was my first plan. Standard conformal prediction outputs a set of labels and guarantees the set contains the right one. The promise a support manager needs is a limit on the error rate among automated decisions. Those are different quantities, so I used selective classification with guaranteed risk. Both belong to the same family of distribution-free methods.

**What is a risk–coverage curve?**
Sort tickets by confidence, answer the most confident first, and plot the error rate against the share answered. It shows the whole trade-off; a threshold is one point on it. The area under it is 0.090 here, against 0.284 if confidence were random.

**Why four data splits?**
Each choice made on a split makes scores on that split too optimistic. Train is the memory, validation fits the confidence model and gate, calibration chooses thresholds, test is used once. The guarantee needs the calibration data to be untouched by earlier steps.

**How does this connect to your research?**
NeuroAbstain decides when a classifier should not label an MRI scan. This decides when a router should not assign a ticket. Same question: when should a model defer to a person, and what can be promised about the cases it keeps?

---

## Engineering questions

**How is it deployed?**
One Docker image with two processes: a FastAPI service and a Streamlit page that calls it over HTTP. The model is built offline into one 20 MB file and only loaded at start-up, so what runs is what was evaluated. The embedding model is downloaded at build time. *(Add the public link once it is live; until then say it runs in Docker locally.)*

**What would you change at company scale?**
Separate the API and the front end. Put the stored vectors in a vector index or database once the memory grows well beyond tens of thousands of tickets; today it is one matrix multiplication over 24,091 rows. Add a staging step and monitoring. Replace PyTorch with an ONNX runtime to shrink the 3.46 GB image.

**What do your tests check?**
27 tests in three groups. The abstention logic, including a simulation over 40 datasets that the threshold keeps its promise on new data. The API with the real model, including bad input and vague messages. And a regression test that fails if accuracy, calibration or coverage drop below recorded values or if any promised limit is broken on the test set.

**How fast is it, and what does it cost?**
Median 157 ms, 95th percentile 225 ms on a laptop CPU, almost all of it the embedding model. No GPU and no paid service, so the running cost is the host; 832 MB of memory.

**What happens when the data changes?**
The guarantee no longer holds, so it must be detected. The handover rate is the cheapest signal: if incoming tickets drift away from the memory, more fall below the gate. I would track that and the similarity distribution and re-choose the threshold on fresh labelled tickets. This monitoring is not built yet.

**How would the system improve over time?**
Every handed-over ticket gets a team from a person. Adding those to the memory teaches the system new kinds of ticket without retraining anything.

---

## Critical questions

**What are the weaknesses?**
Four. The data is machine-generated with noisy labels, so the numbers will not transfer. German is clearly worse than English and exceeds the limit on its own. It adds nothing on new kinds of ticket; it only avoids harm there. And it shows stored ticket text as the reason, which needs access control with real data.

**Why is German worse?**
I measured it but have not fully explained it. Accuracy is 61.1% for German against 76.3% for English, and German tickets have lower similarity to their closest neighbour. Likely causes are fewer German tickets in memory (about 9,900 against 14,200) and an embedding model that is stronger in English. I would test both by matching the memory sizes. Say this as it is; do not invent a cause.

**Would 54% automation hold in a real company?**
I cannot claim that. It depends on how repetitive the real tickets are. What transfers is the method: measure, choose a limit, and get a threshold with a guarantee.

**There is no LLM in this. Why?**
The task did not need one, and a nearest-neighbour lookup is cheaper, faster and easier to guarantee. The evaluation approach is what carries over to LLM systems: a fixed test set, regression tests in CI, and a rule for when the system should not answer. An LLM could draft replies for the confidently routed tickets; that is not built.

**What was the hardest part?**
Accepting that the planned model did not work and finding out why, instead of tuning it. The second was the vague message that got routed at 75% confidence: the numbers on held-out data looked fine, and only trying the running system showed the problem.

---

## Basics you may be asked

| Topic | Be ready to |
|---|---|
| Python | Explain lists, dictionaries, sets; write a function that counts items; explain a dataclass (see `Router`) |
| pandas | Filter, group and aggregate; `data.py` has examples |
| SQL | Write a `GROUP BY` with a count and a `WHERE`; a join of two tables. The data exploration used queries such as `select queue, count(*) from tickets group by 1 order by 2 desc` |
| ML | Train/test leakage, overfitting, precision and recall, why accuracy misleads on unbalanced classes (hence macro-F1) |
| Statistics | What a confidence interval is; what a binomial proportion bound does |
| Git and CI | What the workflow file does on each push |

## Motivation and logistics

| Question | Points to make |
|---|---|
| Why this role? | Name one task from their posting and connect it to a part of this project |
| Hours and duration | Up to 20 hours per week in the semester, available now, for at least 12 months |
| How is your German? | "A1, I am taking a course and aim for B1 by [month]. I work in English and can follow simple German." Then one sentence in German: *Ich lerne Deutsch und kann einfache Gespräche führen.* |
| Work permission | Know your residence permit's rule before the interview (see section 4 of the market scan) |
| Questions for them | What data the team works with, how they evaluate models before release, what a student owned in the last six months |
