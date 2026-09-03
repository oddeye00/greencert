# Directional-envelope source-isolation amendment

Recorded on 2026-09-03 before the public v1.5 release and before the definitive
source-isolated replay.

## What the release audit found

The first outcome-blind implementation audit passed all mathematical and
numerical gates, but its helper functions had been added directly to four
filenames listed in the historical Transformer v3 method seal.  The ordinary
release smoke test correctly rejected those post-freeze byte changes.  The
first audit record had SHA-256
`73D9196BE2090A26C573C92777F3F74BFECDEADF2F53FA795611C078E7275402`;
it is retained as
`results/transformer_directional_envelope_transport_audit_preisolation_v1.json`.

This was a source-provenance defect, not a changed bracket or failed numerical
inequality.  Nevertheless, it blocks release because prospective seals must
remain literal byte commitments.

## Correction

The historical filenames were restored byte-for-byte to their committed v3
hashes:

- `scripts/transformer_block_envelope.py`:
  `9AA1B18F26F7DE214DED1280026EBFBA7A3E332C86FA9228757ACB40265DB7C0`;
- `scripts/transformer_optimizer_probe.py`:
  `3C7665274245731141F4E0F451652F486CB65D7A16B0DCF233BAA66E8BFD5C79`;
- `scripts/transformer_hvp_grokking.py`:
  `7860D3F9B14427CC7AD49020943922DD07A655C269D846E17FB731F27C823F4F`;
- `scripts/transformer_modal_forecast.py`:
  `1FE76CA893EC84E880EC6917274CBE51766C5A1857D96132F55D99532BC71E06`.

All post-seal fused-derivative and transported-envelope additions now live in
explicit `*_v15.py` modules.  The maintained audit imports those modules.  No
historical method, candidate, certificate, random stream, cohort membership,
constant, inequality, or promotion gate was changed.

The same isolation was applied to two later files already bound by their own
post-seal audit records: `scripts/transformer_mixed_directional_jet.py` was
restored to
`EFAC989E5DE42AAE9B33413FF817A8931D9FB3A8E4634219983EE4E7B9A3B059`, and
`scripts/streaming_variational_centerline.py` was restored to
`CEA0ADE5FC1255969B7A93CB1D8525EBBEE9020709CEB5399DB9A090E450228A`.
Their extensions now live in `transformer_mixed_directional_jet_v15.py` and
`streaming_variational_centerline_v15.py`.  The original transitive
directional-replay audit must pass before release.

## Definitive replay

The complete four-case audit is rerun after this isolation.  It must still:

1. reconstruct the frozen centerline and corrected-path hashes;
2. pass every one-sided stage, geometry, and derivative dominance check;
3. retain all four sealed brackets, including all three nondevelopment cases;
4. use no new randomized query; and
5. record zero outcome-file reads.

The resulting source-isolated JSON supersedes the first audit record for all
paper and release claims.  The first record remains solely as an immutable
execution-history artifact.
