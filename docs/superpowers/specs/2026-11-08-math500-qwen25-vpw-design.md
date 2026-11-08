# Math500 Qwen2.5 VPW Design

## Intent

Run the existing `mctsv8_vpw` method unchanged on the single-modal MATH500
dataset with the local Qwen2.5-7B-Instruct checkpoint.  This is an input/model
adaptation only; VPW search, scoring, activation, voting, and generation
parameters remain unchanged except for `mcts_rollouts=8` as requested.

## Fixed paths and execution

- Model: `/home/jncsnlp4/SSD2/model/Qwen2.5-7b-Instruct`
- Dataset: `/home/jncsnlp4/SSD2/ZJJ/rStar/data/MATH/test_all.json`
- Output root: `/home/jncsnlp4/SSD2/ZJJ/VReST/newoutput`
- Full run: existing GPU-0-only queue, after a one-problem smoke succeeds.

## Adaptation boundaries

- Add a Hydra data config for MATH500 and a sequential/resumable text-only
  loader that preserves the source record and stores VPW decisions and timing.
- Add a Hydra model config for the requested Qwen2.5 checkpoint.
- Make the existing Qwen2.5 local wrapper expose the same response-batch and
  self-consistency reward methods already consumed by VPW.
- Make the existing MCTSv8 message builder omit image content for text-only
  models while retaining the same prompts and assistant continuation.
- Set `mctsv8_vpw.mcts_rollouts` to 8.
- Add a `math500_vpw` task set to the existing GPU 0 scheduler.

## Non-goals

- No change to VPW widths, thresholds, activation, reward semantics, answer
  voting, prompts, sampling values, or legacy multimodal configurations.
- No full MATH500 launch until one problem produces a persisted result.
- No reset, checkout, commit, or cleanup of the dirty remote repository.

## Approach

Use small compatibility adapters inside the existing architecture rather than
copying VPW or creating a parallel inference program.  This keeps the method
identical and makes the new dataset/model selectable with normal Hydra
overrides.
