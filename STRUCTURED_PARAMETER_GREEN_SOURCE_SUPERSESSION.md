# Structured-parameter source supersession record

The two structured-parameter audit protocols sealed
`scripts/structured_parameter_green.py` at SHA-256
`0E9561B61F4E76E368A272B28398C04156447B6D3318662F946BDA3164514D86`.
That exact byte sequence remains available as
`scripts/structured_parameter_green_sealed_v1.py`; the matching sealed test
source is retained as `scripts/test_structured_parameter_green_sealed_v1.py`.
The former is the implementation used when the independent verifiers replay
the sealed quadratic closures.

A later post-seal maintenance commit changed one validation expression in the
live module. The sealed implementation converted Python binary64 bounds to a
default-dtype PyTorch tensor before checking finiteness, so finite values above
the float32 range were rejected as infinity. The current source, SHA-256
`69BA0C19E6A8A34CDAF293F0DE0D58959EEF7B09BC5E53C17B0E3E17DCDABA47`,
uses `math.isfinite` and therefore implements the stated binary64 interface.
The maintained test file adds the corresponding regression. No operator
formula, root formula, protocol choice, or stored result changed.

`scripts/structured_parameter_green_source_bridge.py` enforces both hashes,
replays every stored direct and Gram closure attempt through both versions, and
requires exact equality. It also checks the one intended semantic extension:
a finite binary64 coefficient above the float32 range is rejected by the sealed
version and accepted by the maintained version. The original protocol and
result hashes are left untouched; the verifier resolves their dependency to the
exact sealed snapshot and reports the supersession explicitly.
