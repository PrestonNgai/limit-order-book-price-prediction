# Methods

## Source and reconstruction

The initial source is the public Coinbase Exchange BTC-USD feed. Raw messages are
recorded before parsing, along with local receive time, a monotonic clock, session
identity, and available exchange timestamps. Disconnects and quality notices are
stored in the same journal.

L2 updates contain absolute displayed quantities. Zero removes a level. The book
uses Decimal prices as keys and applies a complete message atomically. A failed
update rolls back and blocks reads until a valid snapshot is loaded. Crossed,
locked, and one-sided states are rejected.

Trade messages identify the maker side; the adapter converts it to the opposite
aggressor side. A last_match message is a reference message, not a new execution.
Global exchange sequences are tracked separately from trade IDs. Uncertain trade
coverage is flagged.

## Sampling and features

The sampler uses a 100 ms UTC-aligned receive-time grid. At time t, it includes only
messages received by t; exchange timestamps never move an event backward in time.
Equal-time events retain journal order. No rows are extrapolated past the final
message. A stale book suspends sampling, and resumed observations begin a new segment.

The default depth is ten levels per side; missing levels remain null. The 99 features are:

| Group | Count | Examples |
| --- | ---: | --- |
| Book | 5 | Midprice, spread, relative spread, L1 sizes |
| Imbalance | 4 | Cumulative imbalance at levels 1, 3, 5, 10 |
| Microprice | 2 | Microprice and displacement from mid |
| Depth | 6 | Bid/ask depth at levels 3, 5, 10 |
| Flow | 20 | Added/removed sampled depth and signed depth change |
| Trades | 32 | Counts, volume, aggressor imbalance, VWAP |
| Dynamics | 30 | Lagged returns and changes in spread, imbalance, and depth |

Book imbalance is (bid depth - ask depth) / (bid depth + ask depth).
Microprice weights the ask price by bid size and the bid price by ask size.
The flow features measure changes between sampled displayed depths; they do not
recover individual order submissions or cancellations.

Flow and trade windows span 100/500/1000/5000 ms. Dynamics use lags of
100/500/1000/2000/5000 ms. Trailing windows stay within one continuity segment and
require full history; otherwise the affected features are null.

## Labels and splitting

Future relative mid return is (mid at t+h - mid at t) / mid at t. Default horizons
are 100, 500, 1000, and 5000 ms. The class is UP or DOWN only when the absolute price
change exceeds one quoted tick; exact ties are STABLE. A fixed relative threshold
can be supplied instead.

Labels require a future observation in the same uninterrupted segment. Default
alignment is exact; unavailable endpoints are dropped rather than labeled STABLE.
Features and labels are stored in separate tables and joined by verified identity.

Splits prefer whole chronological collection sessions. If there are too few
sessions, they use 70/15/15 chronological time ranges. A target is purged if its
endpoint reaches the next partition. Boundaries do not depend on class labels.

Imputation uses the mean of observed, eligible TRAIN values. All-missing TRAIN
columns use zero. Scaling uses the population variance after mean imputation;
constant columns use scale one. Validation and TEST use this frozen transformation.
Input hashes and fit-row fingerprints detect changed datasets.

## Models and comparison controls

The majority rule uses TRAIN class frequencies with Laplace smoothing. The imbalance
rule predicts the sign of L1 imbalance, with missing or deadband values assigned
STABLE. Its probabilities come from TRAIN class frequencies conditional on the sign
bin; the most likely class can therefore differ from its hard prediction.

Logistic regression uses L2 regularization. XGBoost uses CPU histogram trees. The MLP
uses 64- and 32-unit hidden layers with ReLU and dropout. The GRU uses a 32-unit hidden
state and a three-class head. Neural training uses seeded CPU operations and AdamW.
Checkpoints are selected by validation loss; settings are in configs.

GRU inputs contain 20 observations, ending at t, with a 1.9-second first-to-last span.
A history cannot cross a split or segment. The loader stores one feature matrix and
materializes only the requested minibatch. Hidden state resets for every window.

The sequence_mlp control uses the GRU's eligible target rows but only the final
feature vector. Snapshot and sequence models have separate cohorts. Causal
rolling features may summarize earlier history across a split boundary, even though
the GRU's explicit observation window cannot cross it.

At 500 ms, XGBoost and MLP ablations add book, imbalance, flow, and trade groups:
5, 9, 29, then 61 inputs. The 99-input comparison adds depth, microprice, and dynamics
together. Rows, seeds, and optimization budgets stay fixed; checkpoints are selected
separately.

## Evaluation

Metrics include accuracy, observed-class balanced accuracy, per-class precision,
recall and F1, three-class macro scores, log loss, Brier score, and one-vs-rest AUC.
Undefined per-class metrics and AUC are null. Macro precision/recall/F1 average three
classes with undefined entries set to zero; balanced accuracy averages only classes
present in the target sample.

Directional coverage is the fraction of non-STABLE predictions. Directional accuracy
counts a STABLE outcome as a miss on a directional prediction. Confidence is the
probability assigned to the hard prediction, not necessarily the largest probability.
Calibration uses ten fixed bins and reports expected calibration error.

The regime cutoffs are TRAIN medians of trailing absolute 1-second return, spread,
trailing 1-second trade count, and top-ten total depth. Ties enter the lower bin;
unknown observations and empty bins remain visible.

For each fixed confidence threshold, the economic diagnostic measures
prediction sign * future return * 10,000 in basis points. A win is a strictly positive
signed move; zero moves are non-wins. Signal frequency is a fraction of eligible
observations. This measures price movement without an execution or cost model.

Uncertainty resamples whole collection sessions, retaining within-session dependence.
Defaults are ten sessions minimum and 2,000 bootstrap draws. Independence between
sessions is still an assumption. Paired accuracy differences require identical rows.
These intervals would remain exploratory after validation checkpoint selection;
they do not correct for repeated model comparisons.

The research runner reports validation results. The
[final evaluation workflow](final_evaluation.md) applies frozen models to TEST.

## Engineering checks

Tests cover atomic book rollback, malformed messages, reconnects, exact-price labels,
causal windows, endpoint purging, TRAIN-only fitting, sequence alignment, model reloads,
absent-class metrics, resource limits, and artifact corruption.

Recorded inference times are warmed, in-process Python measurements. Snapshot timing
includes preprocessing; sequence timing includes window assembly but excludes base
preprocessing. Neither includes network transport or exchange order handling.
