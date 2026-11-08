# VReST Batched Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch MCTSv8 reward prompts, enforce explicit `answer is` extraction, save runs to `newoutput`, and produce rollout 2/4/6/8/10 timing results.

**Architecture:** Refactor reward prompt construction away from immediate model calls, add a vLLM multi-prompt reward API, and batch the two reward dimensions separately. Add structured extraction at the MCTS vote boundary, then drive deterministic benchmark runs through a standalone runner.

**Tech Stack:** Python, pytest, Hydra, vLLM, Qwen2.5-VL.

**Spec:** `docs/superpowers/specs/2026-11-05-vrest-batched-inference-design.md`

## Global Constraints

- Preserve the remote repository's existing dirty working tree; no reset, checkout, or unrelated formatting.
- Default output root is `/home/jncsnlp4/SSD2/ZJJ/VReST/newoutput`.
- Candidates without case-insensitive `answer is` are marked and excluded from voting.
- Preserve MCTS rewards, candidate ordering, pruning, and P3/P3Pro behavior.

## Review Focus

- A single chat conversation is a list of dictionaries and must not be mistaken for a batch of prompts.
- Empty text after `answer is` must be marked, not voted.
- Multiple conflicting `answer is` clauses must be excluded as ambiguous.
- vLLM results must remain aligned with input prompt order across micro-batches.
- Benchmark runs must not reuse existing trees or generation files.

---

### Task 1: Explicit answer extraction and vote filtering

**Files:**
- Modify: `prompt_methods/mctsv8/utils_call.py`
- Modify: `prompt_methods/mctsv8/chartqa_mcts.py`
- Modify: `utils/generate.py`
- Create: `tests/test_mctsv8_answer_extraction.py`

**Interfaces:**
- Produces: `extract_explicit_answer(response: str) -> dict` with raw response, extracted answer, status, and matches.
- Produces: MCTS result fields `raw_response`, `response`, and `answer_extract_status`.

- [ ] Write failing tests for case-insensitive matching, missing marker, empty marker, repeated equal markers, conflicting markers, vote exclusion, and all-invalid candidates.
- [ ] Run the focused tests and confirm they fail for the missing interface/old fallback behavior.
- [ ] Implement structured extraction using `answer is` and integrate valid-only majority voting while preserving raw responses in memory and generation output.
- [ ] Run the focused tests and existing MCTSv8 tests.

### Task 2: Batched vLLM reward evaluation

**Files:**
- Modify: `mllms/base_models.py`
- Modify: `prompt_methods/mctsv8/utils_call.py`
- Modify: `configs/prompt_method/mctsv8_a.yaml`
- Modify: `configs/prompt_method/mctsv8_b.yaml`
- Modify: `configs/prompt_method/mctsv8_c.yaml`
- Create: `tests/test_mctsv8_reward_batching.py`
- Modify: `tests/test_mctsv8_qwen25vl.py`

**Interfaces:**
- Produces: `Qwen2_VL_Model.get_reward_batch(messages_batch, n, node_depth, is_p3pro=False, is_p3=False) -> list[dict]`.
- Produces: `reward_call_batch(...) -> list[float]` with serial fallback for models lacking batch support.
- Consumes: configurable `prompt_method.reward_prompt_batch_size` defaulting to 5.

- [ ] Write failing tests that assert five candidate prompts become two reward batch calls, ordering is stable, micro-batching works, and conditional P3Pro re-scores only eligible candidates.
- [ ] Write failing model tests asserting vLLM receives multiple prepared prompts in one `generate` call and returns one reward per prompt.
- [ ] Implement prompt builders, batch dispatch, result parsing, serial fallback, and conditional P3Pro batch re-scoring.
- [ ] Run batching tests and the full existing MCTSv8 test set.

### Task 3: Output root and rollout scaling benchmark

**Files:**
- Modify: `configs/config.yaml`
- Create: `scripts/benchmark_mctsv8_rollouts.py`
- Create: `tests/test_rollout_benchmark.py`

**Interfaces:**
- Produces: deterministic subset creation and aggregation for rollout values `[2, 4, 6, 8, 10]`.
- Produces: `rollout_scaling.json`, `rollout_scaling.csv`, and a readable Markdown summary below `newoutput`.

- [ ] Write failing tests for deterministic subset selection, unique run directories, existing-output refusal, percentile aggregation, and parse-failure counts.
- [ ] Set the Hydra output root to `newoutput` and implement the benchmark runner with warm-up, unique directories, and subprocess failure propagation.
- [ ] Run benchmark unit tests and all relevant unit tests.
- [ ] Run a one-item smoke benchmark at two rollouts and verify artifacts and explicit-answer fields.
- [ ] Run the fixed 20-item benchmark for rollouts 2, 4, 6, 8, and 10; collect timing/scaling artifacts.
- [ ] Compare a ten-rollout pre-batch reference with the optimized ten-rollout result when comparable cached baseline data is available; otherwise label the comparison unavailable rather than mixing datasets.

### Task 4: Final verification

**Files:**
- Verify all files above without modifying unrelated paths.

- [ ] Run the complete MCTSv8 test suite.
- [ ] Inspect the remote diff and confirm only intended hunks plus new tests/docs/scripts are present.
- [ ] Validate every benchmark artifact parses and all runs reside under `newoutput`.
- [ ] Report unit-test evidence, smoke evidence, rollout timing table, extraction-failure rate, and any GPU-memory limitation.
