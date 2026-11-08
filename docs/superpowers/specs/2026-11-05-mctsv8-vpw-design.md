# MCTSv8 VPW Design

## Intent

Add a new, isolated MCTS method named `mctsv8_vpw` (Virtual Progressive
Widening).  Preserve every existing method unchanged.  The new method spends
more generation budget near the root, keeps a fixed middle width, and narrows
near the answer.  Shallow nodes precompute three diverse reserve children in a
second batched generation phase, but those children remain invisible to MCTS
until the parent's activation evidence is strong enough.

P3Pro and embedding-based similarity checks are explicitly out of scope for
this first version.

## Isolation

- Add `prompt_methods/mctsv8_vpw/` with its own node and MCTS selection logic.
- Reuse the stable batched generation/reward and answer-extraction helpers from
  `prompt_methods/mctsv8` where behavior is identical.
- Add `configs/prompt_method/mctsv8_vpw.yaml`.
- Add an explicit `mctsv8_vpw -> mcts_forward_vpw` dispatch before the existing
  `mctsv8* -> mcts_forward_v8` fallback.
- Do not change the behavior or configuration defaults of `mctsv8_a/b/c`.

## Search stages

For root depth 1 and configured maximum depth `D`:

```text
progress = clamp((depth - 1) / (D - 1), 0, 1)
```

- Shallow: `progress < 0.375`; five visible normal children plus three hidden
  precomputed diversity children, for a maximum visible width of eight.
- Middle: `0.375 <= progress < 0.75`; five normal children.
- Deep: `progress >= 0.75`; three normal children.
- Terminal nodes do not expand.

## Two-phase shallow generation

1. Generate five normal candidate paths.
2. Build three distinct diversity prompts after the normal candidates are
   known.  Each prompt receives the main problem, current reasoning history,
   and all five normal candidates.  The three slots request different
   perspectives and explicitly prohibit repetition.
3. Submit all three diversity prompts to vLLM in one batch.
4. Parse all eight candidates and batch their question/answer reward calls.
5. Add the five normal candidates to `children`; store the three diverse
   candidates in `reserve_pool`, ordered by their own actual reward.

No embedding similarity calculation is performed.  Parse failures do not
enter either visible children or the reserve pool.

## Activation and selection

Each node stores an immutable `activation_original_score`:

- Root: `1.0`, used only for activation and never as MCTS reward.
- Non-root: the node's local reward at creation.

After at least one rollout has passed through a shallow parent:

```text
backprop_mean = Q(parent) / N(parent)
activation_score = 0.5 * activation_original_score + 0.5 * backprop_mean
```

Immediately before child selection, if `activation_score > 0.8` and the
reserve pool is nonempty, move at most one reserve child into visible
`children`.  The newly visible child keeps its own independently computed
question/answer reward.  It participates alongside all original children in
the same immediately upcoming UCT comparison.  Opening a reserve child does
not force it to win selection.

At most one reserve child is opened per parent-selection event.  Reserve
children never recursively create extra reserve slots beyond the normal
shallow-node rule applied to their own depth.

## MCTS behavior

The new method uses prior-aware UCT so scored but unvisited real children can
compete by value rather than being selected solely by list order.  Existing
methods retain their current selection behavior.

Backpropagation retains cumulative `Q`, visit count `N`, and the existing
reward aggregation needed for final answer selection.  Activation uses
`Q/N`, not the historical maximum, to avoid opening all reserve slots because
of a single lucky rollout.

## Observability

Persist enough metadata in the tree/memory to audit:

- `progress` and `search_stage`
- `activation_original_score`
- latest `backprop_mean` and `activation_score`
- reserve count generated/opened/remaining
- child `generation_source` (`normal` or `divgen`)
- child actual question and answer rewards

## Testing

- Progress boundaries and root activation score.
- Widths 5+3 hidden / 5 / 3 across the three stages.
- Two-phase generation and batch ordering.
- Hidden children excluded from UCT, traversal, and backpropagation.
- One reserve child opens when the mixed activation score exceeds `0.8`.
- At most one reserve opens per selection event.
- Opened child joins UCT with its own reward and is not forced-selected.
- `Q/N` activation behavior over multiple rollout returns.
- Legacy `mctsv8_a/b/c` dispatch and behavior remain unchanged.
- Small MathVista smoke run before any full experiment.

## Approaches considered

1. Modify `mctsv8` behind flags: least code, but risks changing the queued and
   historical baselines.
2. Fully copy all MCTSv8 code: strongest isolation, but duplicates reward,
   extraction, and batching fixes.
3. **Selected:** isolate the node/search implementation while reusing stable
   helper functions.  This preserves baseline behavior without duplicating
   unrelated infrastructure.

