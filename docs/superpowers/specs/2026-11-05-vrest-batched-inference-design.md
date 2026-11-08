# VReST Batched Inference and Rollout Timing Design

## Intent

Reduce MCTSv8 inference time by submitting reward prompts to vLLM in batches while preserving candidate order and reward semantics. Make explicit `answer is` extraction the default result path, mark non-explicit answers, exclude them from majority voting, and save new runs under `/home/jncsnlp4/SSD2/ZJJ/VReST/newoutput`.

## Goals

- Batch the five sub-question reward prompts and five sub-answer reward prompts instead of scoring candidates serially.
- Keep `return_num`, P3/P3Pro behavior, pruning, and MCTS selection semantics unchanged.
- Match `answer is` case-insensitively, including phrases such as `the answer is` and `final answer is`.
- Preserve raw responses, store extraction status, and exclude invalid candidates from voting.
- Mark a sample as `parse_failed` when no candidate has a valid explicit answer.
- Measure test-time scaling for 2, 4, 6, 8, and 10 rollouts on one fixed deterministic sample set.

## Non-goals

- No multi-question thread pool.
- No change to `return_num`, tree width, reward definition, or model weights.
- No quantization, speculative decoding, or accuracy-oriented prompt redesign.

## Approach

Use two vLLM reward batches per expanded node: one containing all sub-question reward prompts and one containing all sub-answer reward prompts. A configurable micro-batch size defaults to the tree width so lower-memory systems can reduce it without changing results. This is preferred over one combined mega-batch because five prompts with ten returns already create up to fifty active sequences per reward type.

Answer extraction returns structured status (`ok`, `missing_explicit_answer`, `empty_after_marker`, or `ambiguous_multiple_markers`). Only `ok` candidates enter majority voting. The selected raw response remains available for auditing while the normal `response` field contains the extracted answer.

The benchmark uses a fixed 20-item subset with seed 42, one warm-up outside recorded per-sample timings, unique output directories per rollout value, and no saved-tree reuse. It reports mean, median, P90, total time, throughput, extraction failures, model-call counts where available, and scaling relative to two rollouts.

## Constraints

- Preserve all pre-existing uncommitted changes in the remote repository.
- Write run artifacts only below `/home/jncsnlp4/SSD2/ZJJ/VReST/newoutput`.
- Do not silently fall back to extracting an answer from responses without `answer is`.
