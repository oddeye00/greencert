# Frozen directional three-sweep event audit

Frozen after completion of the directional-block closure diagnostic and before
evaluating any corrected-path event margin under the new remainder.

## Inputs and firewall

- Closure parent:
  `results/transformer_directional_block_remainder_diagnostic.json`.
- Event comparator parent:
  `results/transformer_fully_recentered_three_sweep_audit.json`.
- Evaluate exactly the rows for which the directional closure passed.
- Read no future-outcome or revealed-event file.
- Make no Green query.  Use the closure radii already produced under the
  frozen directional protocol.
- Replay the three-sweep corrected path and require its SHA-256 to match both
  parents before evaluating logits.

## Unchanged event map

Use the existing corrected-centered neural first-derivative envelope, raw
logit-slack construction, persistence 25, and strict positive logic-slack
issuance rule.  The known three-sweep correction remains part of the reference
path and is not charged again in the event margin.

## Prespecified practical-promotion gate

The directional theorem supports an adaptive three-sweep implementation claim
only if:

1. all three newly closed nondevelopment rows issue;
2. each issued bracket is identical to its sealed four-sweep bracket; and
3. every evaluated stage-value envelope closes and no outcome file is read.

Otherwise the result remains a closure-only theorem improvement.

