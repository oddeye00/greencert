# Structured parameter-Green Transformer audit protocol

Frozen before the first structured-operator probe on 2026-08-30.

## Evidence boundary

This is a post-v1.0.1, outcome-blind method audit. The theorem and audit were
designed after the released full-state results were known. It is therefore not
a prospective confirmation and may not change any frozen count in v1.0.1.
Revealed future trajectories and outcome files are forbidden inputs.

## Fixed cohort and policy

- Cohort: all 15 Green-evaluable corrected-path Transformer operators from the
  sealed relinearized panel.
- CASE_SET_SHA256:
  `A34AEBB6651B05C4FE18A5379D1778838B276C4709D530936285A600FE2030FB`
- Structured operator: (T=P_\theta K B), with
  (Bq=(-\eta q,\eta q)).
- Probe prefixes: 4, 8, 16.
- At each prefix: try the direct-image bound first; apply the transpose only if
  direct closure does not issue.
- Structured Green family failure budget: (10^{-6}), divided across the 15
  operators and three prefixes. Direct and Gram bounds reuse the same Gaussian
  projection event at a prefix.
- Inherited output family budget: (10^{-6}).
- MASTER_NONCE:
  `d6aaf814b0d13ac0f9305a39dfe94ed3bd2c779791381f1583d2ac0e2b5da391`
- No hyperparameter, prefix, route, case, or success criterion may change after
  a structured probe is observed.

## Promotion gate

The structured result may be promoted as a practical theorem improvement only
if all 15 inherited brackets are retained and total logical Green sweeps are
strictly below the matched full-state staged total. Otherwise it remains a
valid theorem with a negative or mixed systems result. No new coverage claim is
permitted because future outcomes were already known at method-development
time.

## Sealed dependencies

- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_THEOREM.md
  `310EEABFF92FBF024556E7E2E532ACE86DF4BD53284A6CAA49BEEAFB2E4C443F`
- DEPENDENCY:results/transformer_direct_image_green_panel_audit.json
  `931CBF5750510C49DEB92F16F77E8CCA355C7969A18BCF4EFA1A0701335ED705`
- DEPENDENCY:results/transformer_v3_relinearized_prefix_panel_audit.json
  `08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B`
- DEPENDENCY:scripts/audit_structured_parameter_green_transformer.py
  `67CAF938FCE231474B898F98DA20CA990A949BBC86DBCD47BC65EE1F8D26B66F`
- DEPENDENCY:scripts/audit_transformer_direct_image_green_panel.py
  `A46FFE4F68AA15E75EA37755231227481233A8EFD9C35D2066CBB1141A095BB7`
- DEPENDENCY:scripts/audit_transformer_relinearized_prefix_panel.py
  `BE35D0771CF49B53B2D0721AA4BF3035EE9A9BF2F2DFA1BABB2B9B37A47A2B58`
- DEPENDENCY:scripts/batched_green_operator.py
  `C30C1DAF0E8A8494B518CD12E6328146B1A20DD297033192C78408C2E46F54BF`
- DEPENDENCY:scripts/direct_image_green_bound.py
  `EC8B9DECA1FE6E17B4C03AA145A8829BA141BE8A5D223CDA86F82305B360F778`
- DEPENDENCY:scripts/prefix_gram_enclosure.py
  `758142A941D4039E72014C9352AFDE3CA01DD39EE5918D115017905E924B2D78`
- DEPENDENCY:scripts/streaming_variational_centerline.py
  `CEA0ADE5FC1255969B7A93CB1D8525EBBEE9020709CEB5399DB9A090E450228A`
- DEPENDENCY:scripts/structured_parameter_green.py
  `0E9561B61F4E76E368A272B28398C04156447B6D3318662F946BDA3164514D86`
- DEPENDENCY:scripts/test_structured_parameter_green.py
  `8D53A19E247FDFD4E67FF74B87A9E8194C6601ED8D799058F09F230CCF5F1EB8`
- DEPENDENCY:scripts/transformer_green_operator.py
  `F529F42D8DC6B01C94D5F5496AC60BB8F63B4BC030FF85449F106362C8CE20B4`
- DEPENDENCY:scripts/transformer_optimizer_probe.py
  `3C7665274245731141F4E0F451652F486CB65D7A16B0DCF233BAA66E8BFD5C79`
- DEPENDENCY:scripts/transformer_v3_certificate.py
  `5050DA95BAE58B02CB15AFAAF4802F9968A70562A26934B9F5E13FD224E71CBF`

