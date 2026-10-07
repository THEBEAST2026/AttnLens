# AttnLens: Score, Execute, Check Attention Edits for Open-Weight LLMs

> Hacktober Fest (Elevate x IIIT Nagpur) | Track 1: Best Open-Source AI Project
> Team: `Team Everest` | Members: `Ayush Sahay, Smit Rodge`

## 1. Problem

Long-context LLM applications constantly make decisions about attention and the KV cache: which prompt spans can be evicted to save memory, which spans should be emphasised to steer an answer, and which tokens actually drive a prediction.

Today these decisions are made by heuristics (for example, "evict low-attention tokens") or by brute force (run the model once per candidate edit). Heuristics are cheap but unreliable, and brute force is accurate but too expensive when there are hundreds of candidates. Developers get no clear, checkable signal for how much an edit will change the model's answer.

## 2. Proposed Solution

**AttnLens** is an open-source toolkit and web app that scores many candidate attention and KV-cache edits from **one cached baseline run and one backward pass**, ranks them by predicted effect on the answer, and then **verifies only the top candidates by actually executing them**.

It follows a three-step loop:

1. **Capture:** run the prompt once, cache queries, keys, values, and the gradient of the answer margin.
2. **Score:** compute the closed-form local effect of each candidate edit on the attention output, then contract it with the baseline gradient to predict the change in answer.
3. **Check:** execute the top-ranked edits on the real model and show predicted vs. actual effect side by side.

The scoring math builds on recent published work (see Credits). Our contribution is the engineering around it: an independent implementation, a verification harness, and a usable interface.

## 3. Target Users

- Developers building long-context or RAG applications who need principled KV-cache eviction.
- Interpretability students and researchers who want fast attribution for attention edits.
- Practitioners experimenting with attention steering (emphasising or suppressing prompt spans).

## 4. Selected Open-Source AI Technology

- **Model:** Qwen2.5-0.5B-Instruct (open-weight, Apache 2.0). It is small enough for free Colab GPUs or CPU, so the project stays reproducible. Qwen2.5-1.5B-Instruct is an optional second model.
- **Framework:** PyTorch and Hugging Face Transformers, with eager attention so internals are accessible.
- **Core idea:** exact finite-change response of softmax attention (details in Section 6).

## 5. Role of AI

The open-weight LLM is the object being analysed and edited, not an add-on. Every feature depends on it:

- Its cached attention internals (queries, keys, values, probabilities) feed the scoring engine.
- Its backward-pass gradient turns local attention changes into predicted answer changes.
- Its real forward passes verify the predictions.

Without the model, the tool has nothing to score or check. No proprietary API is used anywhere.

## 6. Core Method (Summary)

For a fixed query and attended token set, if some tokens' attention scores change by `d_j` and values change by `eps_j`, the exact change in the attention output is a closed-form expression that fully accounts for softmax renormalisation, with no linear approximation. We use three consequences:

- **Deletion effect:** removing a token group G with attention mass `a_G` and weighted value `w_G` changes the head output by `(a_G * y - w_G) / (1 - a_G)`. This lets us rank spans for eviction.
- **Joint key/value edits:** the interaction term `C_KV` (omitted when key and value attributions are simply added) is retained for better predictions.
- **Distortion budget:** the norm of the projected output change can be checked against a user-set tolerance, giving a checkable eviction criterion.

Predictions are contracted with a frozen baseline gradient, so they are estimates to be verified, not guarantees.

## 7. Architecture

```
+-------------------+     +--------------------+     +--------------------+
|   Web UI          | --> |   API layer        | --> |  Capture module    |
| (prompt, spans,   |     |   (FastAPI)        |     |  (cache Q/K/V,     |
|  budget, results) | <-- |                    | <-- |  probs, gradient)  |
+-------------------+     +--------------------+     +--------------------+
                                   |                          |
                                   v                          v
                          +--------------------+     +--------------------+
                          |  Scoring engine    |     |  Open-weight LLM   |
                          |  (deletion, KV,    |     |  (Qwen2.5-0.5B)    |
                          |   joint edits)     |     +--------------------+
                          +--------------------+              ^
                                   |                          |
                                   v                          |
                          +--------------------+              |
                          |  Ranker +          | -- top-k --> |
                          |  Verifier          | <-- actual --+
                          +--------------------+
```

## 8. Data Flow

1. The user enters a prompt, picks candidate spans (or auto-segments by sentence or record), and chooses an objective (evict, emphasise, or attribute) plus an optional distortion budget.
2. The Capture module runs one forward pass, stores per-head attention statistics and cache entries, and runs one backward pass to get the answer-margin gradient.
3. The Scoring engine computes the exact local response for every candidate and contracts it with the gradient.
4. The Ranker sorts candidates and filters those that exceed the distortion budget.
5. The Verifier executes the top-k edits on the real model and records the actual change.
6. The UI shows a token heatmap, a ranked table, and a predicted-vs-actual chart with error metrics.

## 9. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Model runtime | PyTorch, Hugging Face Transformers |
| Backend | FastAPI |
| Frontend | Streamlit (fast to build) or a lightweight React page |
| Visualisation | Plotly / Matplotlib |
| Testing | pytest, numerical equivalence checks against dense recomputation |
| Hardware | Free Colab GPU or CPU |

## 10. Implementation Plan (Final Round)

**Must-have (MVP)**
- Capture module for one head, then all heads at one layer.
- Deletion-effect scoring and ranking.
- Verifier that executes top-k deletions and reports predicted vs. actual.
- Basic UI with heatmap and comparison chart.
- Numerical tests: scoring agrees with brute-force dense recomputation.

**Should-have**
- Distortion-budget filter for KV-cache eviction.
- Multi-layer support and sign-accuracy / rank-correlation metrics.
- Preset demo prompts (retrieval and sentiment tasks).

**Stretch**
- Joint key/value edit scoring with the interaction term.
- Attention-steering mode (emphasise a span to shift an answer).
- Minimum-norm query control for requested attention ratios.

## 11. Expected Output

- A public GitHub repository (MIT or Apache 2.0) with documented code.
- A working web app where a user pastes a prompt, ranks spans, and sees verified results.
- A reproducible evaluation script reporting rank correlation and sign accuracy of predicted vs. executed effects.
- A short demo showing a cache-eviction decision made and verified end to end.

## 12. Scalability

- The expensive steps (baseline forward and backward pass) happen once per prompt. Scoring each extra candidate touches only the edited entries, so cost grows slowly with the number of candidates.
- The design is model-agnostic for standard RoPE softmax attention, so other open-weight families (Llama, Mistral, Gemma where compatible) can be added through adapters.
- Candidate batches can be scored on GPU, and prepared statistics can be reused across edit strengths.

## 13. Dependencies

`torch`, `transformers`, `numpy`, `fastapi`, `uvicorn`, `streamlit`, `plotly`, `pytest`. All open-source.

## 14. Expected Challenges

- **Hooking attention internals:** grouped-query attention and rotary embeddings need careful handling. Mitigation: start with the small model, verify against a dense recomputation test.
- **Prediction accuracy limits:** predictions rely on a frozen gradient, so large edits can drift from reality. Mitigation: always verify top candidates by execution and report error honestly.
- **Numerical stability:** near-complete attention concentration can cause cancellation. Mitigation: compute in log space and use stable primitives (`expm1`, log-softmax).
- **Scope and time:** the hackathon is short. Mitigation: a fixed MVP, with stretch goals only if time remains.
- **Generalisation:** the method is validated on a small set of models and tasks. Mitigation: state this clearly and test on at least two tasks.

## 15. Credits and Attribution

The mathematical foundation of this project is from:

> Julie Huang, Maggie Chlon, Gregory Gutin, Leon Chlon. **"Exact finite attention responses from RoPE derivatives."** arXiv:2609.14127 [stat.ML], 2026.


## 16. License

To be released under the MIT License.
