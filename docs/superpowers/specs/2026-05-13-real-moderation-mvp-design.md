# Real Moderation MVP Design

## Context

Safe Tweet Detector is currently a research-oriented notebook project for 4-class harm detection on the BeaverTails dataset. The existing work compares Logistic Regression, LinearSVC, and an LSTM, with future work calling out transformer models and deployment.

This design turns the project into a local production-style moderation MVP. The goal is not to copy a full social media moderation platform, but to add the core pieces real systems rely on: transformer inference, confidence thresholds, a moderation queue, human review, analytics, and audit-friendly prediction records.

## Goals

- Fine-tune a transformer classifier for the existing 4 harm labels.
- Serve automated moderation decisions through a local FastAPI service.
- Route uncertain or risky content into a moderation queue.
- Store predictions, review outcomes, thresholds, and model metadata in SQLite.
- Provide simple abuse analytics and threshold-tuning support.
- Keep the system small enough to run locally and extend gradually.

## Non-Goals

- Building a distributed streaming system in the first MVP.
- Deploying GPU infrastructure or cloud autoscaling.
- Building a polished web dashboard.
- Treating model output as final truth without human review.
- Implementing heavyweight explainability such as SHAP or LIME in the first pass.

## Architecture

The project will be organized into four layers.

1. Model layer: a fine-tuned `distilroberta-base` transformer classifier, trained on the same 4-label BeaverTails scheme used by the notebook.
2. Inference API: a FastAPI service exposing moderation endpoints and returning labels, confidences, class probabilities, threshold decisions, and explanation fields.
3. Moderation queue: SQLite-backed review storage for predictions that are harmful, severe, low-confidence, or otherwise review-worthy.
4. Analytics layer: simple endpoints or scripts for class counts, queue volume, confidence distributions, model version tracking, and threshold experiments.

The first usable system should run locally: train or load an exported model, start the API, submit text, receive an automated decision, and store review-worthy cases.

## Components

### `src/safetweet/data/`

Loads local `dataset/train.jsonl.xz` and `dataset/test.jsonl.xz`, maps BeaverTails categories into the existing 4 labels, and prepares Hugging Face `Dataset` objects for transformer training.

### `src/safetweet/training/`

Fine-tunes one transformer model, saves the model and tokenizer under `models/transformer/`, and writes evaluation metrics to `reports/metrics/`.

### `src/safetweet/inference/`

Loads the saved model once, runs text classification, applies threshold policy, and returns a structured moderation result.

### `src/safetweet/api/`

Provides FastAPI endpoints:

- `GET /health`: service and model readiness.
- `POST /moderate`: classify one text input and apply policy.
- `GET /queue`: list queued moderation items.
- `PATCH /review/{item_id}`: record human review decisions.
- `GET /analytics`: summarize abuse and review activity.

### `src/safetweet/storage/`

Uses SQLite for prediction logs, queued items, review decisions, threshold settings, and model metadata.

### `src/safetweet/explainability/`

Provides pragmatic explanation fields for the MVP:

- predicted label and confidence;
- probabilities for all 4 labels;
- decision reason, such as severe harm above threshold or low-confidence prediction;
- category grouping used by the label mapper where available.

The MVP should avoid presenting attention weights as reliable explanations.

## Data Flow

1. Training reads local BeaverTails JSONL files.
2. Each row becomes `text = prompt + "\n\n" + response`.
3. The 14 BeaverTails harm categories are mapped into 4 labels:
   - `0`: Severe Harm
   - `1`: Non-violent Harm
   - `2`: Social / Contextual Harm
   - `3`: Safe
4. The transformer is fine-tuned and evaluated against the test split.
5. The trained model is saved with tokenizer, label map, metrics, and model metadata.
6. Runtime API receives text from `/moderate`.
7. Inference returns probabilities for all classes.
8. Threshold policy decides `allow`, `flag_for_review`, or `block`.
9. Prediction is logged.
10. Items needing review are written to the moderation queue.
11. Human review updates queue status and corrected label.
12. Analytics summarizes harmful-class volume, queue volume, confidence bands, review outcomes, and threshold effects.

## Threshold Policy

The MVP uses configurable thresholds rather than hard-coded confidence assumptions.

- Block: Severe Harm above a strict threshold.
- Flag: harmful predictions with medium confidence, or any low-confidence prediction.
- Allow: Safe prediction above the safe threshold.
- Review: ambiguous cases, low-confidence predictions, and harmful classes that do not meet the block threshold.

Initial thresholds can be conservative defaults stored in code and later persisted in SQLite. Threshold changes should be tracked with model version and timestamp so analytics can explain behavior changes.

## MVP Runtime Decisions

The first implementation should support two inference providers behind one service interface.

1. Mock provider: a deterministic local provider for tests, API smoke checks, and development on machines without a saved transformer model.
2. Transformer provider: a Hugging Face sequence classification provider that loads a saved model and tokenizer from `models/transformer/`.

The API should use the mock provider when no model directory is configured, which keeps local development and endpoint tests lightweight. When a model directory is configured, the API should use the transformer provider and fail clearly if expected model files or metadata are missing.

The first transformer backbone should be `distilroberta-base`. It is a reasonable first choice because it is smaller than full RoBERTa, generally stronger than older uncased BERT variants for short text classification, and practical for local fine-tuning experiments. If local hardware is constrained, the training script should allow small sample limits and one-epoch smoke runs rather than requiring a full GPU training job.

Training should be designed in two modes:

- Smoke mode: small train/test sample limits, one epoch, and deterministic output checks to validate the pipeline quickly.
- Full mode: larger local BeaverTails files, normal evaluation metrics, and exported model artifacts for the API.

This keeps CI and development feedback fast while preserving a path to a real transformer model.

## Storage Requirements

SQLite is the MVP system of record. It should store enough information to reproduce each moderation decision without relying on transient API logs.

Prediction records should include:

- input text;
- predicted label id and label name;
- confidence and all class probabilities;
- policy action and reason;
- model version;
- timestamp.

Queue records should include prediction id, status, created timestamp, review timestamp, and the reason the item entered review through the linked prediction. Review updates should preserve the original model prediction and add reviewer status, reviewer label, notes, and review timestamp.

Threshold records should include threshold names, values, model version, and timestamp. The MVP can start with defaults in code, but the storage layer should be able to persist and return the active threshold set so analytics can explain behavior changes over time.

## API Contract

The API should expose stable JSON contracts that are easy to consume from scripts, notebooks, or a future dashboard.

`POST /moderate` accepts one text field and returns:

- predicted label id and label name;
- confidence;
- probabilities keyed by label id or label name;
- action: `allow`, `flag_for_review`, or `block`;
- reason;
- `automation_level: "assistive"`;
- model version;
- queue status and queue id when the item is routed for review;
- warning messages if prediction succeeded but logging failed.

`GET /queue` returns queued items with enough context for a human reviewer to triage them. It should support returning open items first; pagination can be deferred until the queue grows beyond MVP scale.

`PATCH /review/{item_id}` records a final human decision. It should reject unknown queue ids, invalid statuses, and labels outside the 4-class scheme.

`GET /analytics` returns aggregate counts for predictions, actions, labels, queued items, review outcomes, confidence bands, and model version activity.

## Error Handling

- If the model is missing, the API should fail startup with a clear error.
- If text is empty or too long, `/moderate` should return validation errors.
- If model confidence is low, the system should route to review rather than overclaim certainty.
- If SQLite logging fails, the API may still return the prediction, but the response should include a logging warning.
- If model metadata or label map does not match the expected 4 classes, startup should fail.
- Human review updates should preserve original prediction, reviewer label, timestamp, and notes.

## Safety And Product Positioning

The API response should include `automation_level: "assistive"` so the system is clearly framed as moderation support, not perfect judgment. The project should also keep severe decisions auditable by logging probabilities, thresholds, model version, and review outcomes.

## Testing

Tests should cover:

- BeaverTails category-to-label mapping.
- Loading compressed local JSONL data.
- Threshold decisions for allow, flag, review, and block.
- Inference response shape using a lightweight mocked model.
- API validation and moderation queue creation.
- Analytics counts from known seeded records.
- Startup failure when model metadata and label map are incompatible.

## Acceptance Criteria

The MVP is complete when a developer can:

1. Install the project locally as a Python package.
2. Run unit tests for labels, data loading, policy decisions, storage, inference, API endpoints, analytics, and training helpers.
3. Start the FastAPI app with the mock provider and submit a moderation request.
4. See review-worthy requests persisted to SQLite and returned by the queue endpoint.
5. Record a human review decision without overwriting the original prediction.
6. Run analytics against seeded records and receive deterministic counts.
7. Run a transformer training smoke command that validates the data-to-export path without requiring a full GPU job.
8. Configure the API to load an exported transformer model and fail startup clearly if expected model metadata or label maps are missing or incompatible.

## Risks And Constraints

- BeaverTails categories are safety labels for prompt-response pairs, not tweets specifically. The MVP should describe itself as general short-text moderation support unless future data is collected for tweets.
- The 4-label hierarchy intentionally prioritizes severe harm over other category flags. This improves operational clarity but can hide multi-label nuance.
- Mock inference is useful for development but must not be presented as a real safety model.
- Confidence thresholds are policy tools, not proof of correctness. The API should keep human review in the loop for uncertain or harmful outputs.
- Class imbalance, especially for Social / Contextual Harm, remains a known risk and should be visible in evaluation reports.

## Implementation Order

1. Package the project with a `src/safetweet` module layout.
2. Add data loading and label mapping tests.
3. Add transformer training/export scripts.
4. Add inference service with model metadata validation.
5. Add threshold policy tests and implementation.
6. Add SQLite storage and moderation queue.
7. Add FastAPI endpoints.
8. Add analytics summaries.
9. Update README with local setup, training, and API usage.

## Resolved Implementation Decisions

- First transformer backbone: `distilroberta-base`.
- Training default: smoke-test-friendly by command-line limit, with full training available when local compute allows.
- API development mode: deterministic mock provider for tests and local API checks.
- API model mode: Hugging Face provider loading an exported local model from `models/transformer/`.
- Storage: SQLite with prediction, queue, review, threshold, and model metadata records.
- Product framing: assistive moderation support with auditable records, not autonomous final judgment.
