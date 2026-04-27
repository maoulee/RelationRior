# Relation-Prior Forward Walk

## Goal

Replace the previous backward-target / bidirectional matching with a strictly forward layer-walk:

1. start from anchor
2. match layer 1 relation-prior
3. use matched endpoints as the start frontier for layer 2
4. continue layer by layer
5. if a layer has no hit, skip that layer
6. if explicit endpoint targets exist, connect final frontier to those targets with a short unconstrained path

This is intended to reduce:

- repeated relation penetration like `r1 -> r1`
- backward template bias from the final layer
- noisy meeting paths introduced by bidirectional BFS

## Current implementation

File:
- `scripts/test_chain_decompose.py`

Function:
- `relation_prior_expand()`

The function now performs **forward layer-by-layer expansion**.

## Core behavior

### Layer search

For each relation layer `R_i`:

- from the current frontier paths
- search paths within `max_hops`
- record only those paths whose **last hop relation** is in `R_i`

Those matched paths become the next frontier.

### Layer skip

If layer `R_i` has no hit:

- do not terminate
- keep the current frontier unchanged
- continue to the next layer

This implements:

> if first layer has no relation hit, skip it and continue matching later layers

### Endpoint connection

If explicit endpoint targets exist:

- take the final frontier endpoints
- run a shortest unconstrained path search to those targets
- if connected within `max_hops`, treat it as matched

This implements:

> after reasoning layers are matched, connect final endpoints to explicit endpoints if possible

## CVT handling

CVT-like nodes (`m.xxx`, `g.xxx`) are still treated as passthrough-like nodes in hop counting:

- entity → CVT costs `0`
- CVT → entity costs `1`
- entity → entity costs `1`

This keeps the previous intent:

- bridge through CVT should not consume the same budget as a full semantic hop

## Ranking

The implementation now uses a pure pattern-quality ranking:

- `covered_steps`
- path depth
- shorter bridge is preferred via existing coverage ranking

Important:

- pattern ranking does **not** use `support`
- the number of raw paths merged into a pattern is no longer a ranking signal
- only the pattern's own best layer-hit quality matters

So the traversal changed, but downstream compression and LLM reasoning interfaces remain unchanged.

## Why this is better than the previous bidirectional version

The previous design:

- used the last layer as the primary backward template
- matched from both directions
- allowed repeated relation penetration when the same relation family appeared in adjacent layers

The new design:

- only walks forward
- only promotes paths whose **last hop** matches the current layer
- carries endpoints forward explicitly

That makes the search closer to:

> anchor -> entity relation -> next entity -> next relation

rather than:

> guess final relation first, then try to stitch the rest backward

## Remaining limitation

This change does **not** solve:

- poor relation priors
- poor anchor priors
- weak bridge entity retrieval
- final LLM path selection bias

It only changes the traversal policy.

The upstream prior quality is still the main determinant of end-to-end quality.
