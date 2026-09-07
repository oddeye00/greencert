# A completed 451,008-parameter training-event certificate

The registered five-update-persistent 35-of-58 event occurred at offset
**44**, inside the pre-observation bracket **[44,44]**. The reference anchor
was update 3925, so the observed onset is update 3969. The observation
completed all 64 registered optimizer updates before its outcome was read.

The [full evidence graph](https://github.com/oddeye00/greencert/releases/tag/evidence-451k-20260907)
is now public. A fresh download and reconstruction on DGX reproduced the
complete recorded enclosure and all 17,713 input hashes. The separately
downloaded public outcome package also reproduced the logit/event audit.
The result-freeze record is `451K_RESULT_FREEZE.md`.

This is a four-block, width-96 Transformer with learned LayerNorm, 451,008
parameters, and full-batch momentum on mod-17 addition. It extends the
complete construction beyond the earlier 13,792-parameter one-block case.
It is an outcome-sealed development case, not an additional member of the
earlier fresh-seed cohort. AdamW, minibatching and dropout were not used.

## Result and numerical scope

| Quantity | Recorded result |
| --- | ---: |
| Registered horizon | 64 updates |
| Certification examples | 58 |
| Target / persistence | 35 correct / 5 updates |
| Frozen bracket / point prediction | [44,44] / 44 |
| Observed persistent onset | 44 |
| Logged observations | 65 |
| Independently decoded logits | 64,090 |
| Argmax ties | 0 |
| Observed counts inside certified count bounds | 65/65 |

The original certificate's numerical scope is outward validated-producer
arithmetic conditional on its ideal-Gaussian projection event, with a
registered failure probability of 1e-7. The separate saved-evidence replay
recomputes the physical-anchor encoding, saved norm inequalities, Green
bound, closure and event assembly; it authenticates the neural producer
records rather than rerunning every neural kernel. Full recorded replays
on Windows and ARM agree on the assembly and Green bound.

The outcome is a CPU-float64 continuation, not an independent exact-real
state-tube enclosure. The initial parameter and unscaled-velocity bytes
match the stored physical checkpoint. Neither this observation nor the
certificate establishes population generalization or certifies the entire
preceding floating-point training program.

## Reproduce the completed observation audit

No training, outcome selection or random probing is performed by these
commands. Use the repository's pinned Python environment and new output
directories / report names:

```text
python scripts/test_dgx_451k_outcome.py
python scripts/package_dgx_451k_outcome.py --mode extract --archive artifacts/greencert_451k_completed_outcome_20260907_v1.zip --archive-sha256 92e378a569b6837595ac6f520de6eb6cf004e5c14f52ee5b94be1e0d7e24090c --destination output/completed_451k
python scripts/audit_dgx_451k_outcome.py --root output/completed_451k/root --bundle output/completed_451k/amendment --reveal-sha256 25f204627419b5bbdfdbb902e1d0925f08f5a796aea51e8edbe48d2cb01040c3 --amendment-sha256 d8f56172db878c1a0986c6cfcaf384166b0a33a37284497e53514ec7a9d3e97a --publication-commit 62ba2e3fac7bdda3784f87e5cb1772f5c5fc1238 --report output/completed_451k/audit.json
```

The 8,699,804-byte artifact contains 219 members: the unchanged amendment
bundle, physical checkpoint, complete observation ledger, source-host
delegation and final reveal, plus a post-observation transport manifest.
The independent audit reconstructs first-index argmax decisions from
hexadecimal float64 logits and finds the event with a separate streak
algorithm. Windows and ARM reports agree byte for byte. The regression
suite checks 2,555 short event paths and rejects 30 malformed or coherently
modified temporary fixtures. The original records are never modified.

Full certificate-input reproduction requires the separate 11.75 GB
recorded graph; the small outcome artifact is not a substitute for it.
See [Recorded-window replay](RECORDED_WINDOW_REPLAY.md) and
[Full recorded graph transport](FULL_RECORDED_GRAPH_TRANSPORT.md).

## Execution and provenance

The amendment was committed at
`62ba2e3fac7bdda3784f87e5cb1772f5c5fc1238`; remote availability was checked
before delegation and execution. Both GitHub CI runs for that commit
completed successfully: 34161278329 and 34161278151. The source-host
reservation blocked competing local observers. One remote observer ran;
there was no current-model retry. The registered algorithm and event did
not change. The runtime changed from Windows/AMD64 to Linux/aarch64 on
DGX, including the platform-specific Python/PyTorch build. See the unchanged
[execution amendment](DGX_OUTCOME_AMENDMENT.md).

These are cooperative protocol controls, not tamper-proof execution
attestation. The post-run review identified three implementation limits:

1. The bridge checked its frozen observer sources but did not compare the
   entire replay-loader dependency list before importing it. The additional
   strict source-chain audit rejected the retained v1 loader because the
   completed numerical replay pins v3. A separate source-only audit finds
   exactly one AST difference: a reference-margin import changes to the
   portable adapter. Every function/class body is identical, and the
   `load_context` routine used by the observer does not reference either
   changed margin symbol. All 86 source pins agree with the explicitly
   documented legacy-loader variant; its 37-source bridge closure has no
   static internal dependency gap. This establishes current compatibility,
   not retroactive pre-execution source authentication. The strict refusal
   and the separate compatibility report are both retained in `results/`.
2. The reveal hash binds all observation rows and the start record, but not
   the engine-ready, intent or delegation files. The complete transport now
   preserves those exact bytes. Their timestamp order is checked, not
   independently timestamp-attested. The commit argument to the logit audit
   is an external provenance reference, not a proof of public availability.
3. The logger synchronizes file contents but does not explicitly synchronize
   the containing directory after reservation/link publication. The run
   completed normally; power-loss durability was not tested.

These findings are retained with the completed result. The frozen observer
and all original payloads remain unchanged. The next prospective experiment
must close these control gaps before its seal; it must not reuse this run
as a fresh observation or retrospectively relabel its provenance.
