# Anchor-fixed structured parameter-Green audit protocol

Frozen on 2026-08-30 before the first anchor-restricted operator probe.  The
outcomes of the earlier unrestricted structured-operator audit were known when
this theorem was derived; this is therefore a post-release, outcome-blind
method audit, not a prospective confirmation.

## Fixed question and policy

The implemented causal Green operator returns errors

\[
(e_1,\ldots,e_H),
\]

while the nonlinear remainder at update (j) depends on (e_j).  Because the
realized anchor has (e_0=0), its update-zero nonlinear forcing is identically
zero.  This audit restricts the structured operator to the remaining
(H-1) forcing blocks.  No curvature-profile optimization is used: every
retained profile weight is one.

- Cohort: the same 15 Green-evaluable corrected-path Transformer operators.
- CASE_SET_SHA256:
  `A34AEBB6651B05C4FE18A5379D1778838B276C4709D530936285A600FE2030FB`
- Operator: (P_\theta K_H BQ_0), where (Q_0) prepends a zero forcing block.
- Prefixes: 4, 8, 16; direct image first, matched power-one Gram fallback.
- New Green family budget: (10^{-6}), divided over all operators and stages.
- Inherited output family budget: (10^{-6}).
- MASTER_NONCE:
  `d784fcf9c34ecb9372c4c20a492838406017c3a7cad9f0c42eff766b10ced7be`
- Revealed future trajectories and outcome files are forbidden.
- Promotion requires all 15 inherited brackets and a strict total logical
  Green-sweep reduction from the unrestricted structured baseline of 96.
  Otherwise the result is reported as negative or mixed.
- The audit may not change any prospective coverage count.

## Sealed dependencies

- DEPENDENCY:scripts/audit_anchor_fixed_structured_parameter_green_transformer.py
  `1FEEA6AEC1F47124C2BDDC1656CDC15939EA1066F52D83CC98B4EEF153FCDE49`
- DEPENDENCY:scripts/structured_parameter_green_v2.py
  `BA63311FDA933E0DDF92EC9298DBF7AF84C715A604E637A1A9EAFAB8FAB8E470`
- DEPENDENCY:scripts/test_structured_parameter_green_v2.py
  `E27FF12B9463B49C766290BED06444D993AF51185DD826211987F89118507219`
- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md
  `ED47B8D3201A25CAEF358341B4ACD0692CFFFA77504FC5E41C77949F75DE3589`
- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_THEOREM_V1_INDEXING_NOTE.md
  `7AB38225C7480C544B1BDCCA02F9E45B9042C62E1AD2DFB3E3CC0CB684BD8922`
- DEPENDENCY:results/structured_parameter_green_transformer_audit.json
  `78C3855CD5464414C92C23175D7B017A683D02FC4A1FD97AA47E9003F9412A3F`
- DEPENDENCY:scripts/audit_structured_parameter_green_transformer.py
  `6377CFEF7330A0E78889E45D1D7348B480EFDD29BD9C3816B3784FF17CC89960`
- DEPENDENCY:scripts/structured_parameter_green.py
  `0E9561B61F4E76E368A272B28398C04156447B6D3318662F946BDA3164514D86`
- DEPENDENCY:scripts/test_structured_parameter_green.py
  `8D53A19E247FDFD4E67FF74B87A9E8194C6601ED8D799058F09F230CCF5F1EB8`
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
- DEPENDENCY:scripts/transformer_green_operator.py
  `F529F42D8DC6B01C94D5F5496AC60BB8F63B4BC030FF85449F106362C8CE20B4`
- DEPENDENCY:scripts/transformer_optimizer_probe.py
  `3C7665274245731141F4E0F451652F486CB65D7A16B0DCF233BAA66E8BFD5C79`
- DEPENDENCY:scripts/transformer_v3_certificate.py
  `5050DA95BAE58B02CB15AFAAF4802F9968A70562A26934B9F5E13FD224E71CBF`
- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_THEOREM.md
  `310EEABFF92FBF024556E7E2E532ACE86DF4BD53284A6CAA49BEEAFB2E4C443F`
- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md
  `4FED1A8D11B716F53329EC3C466F1AAF6C6B95F47908072D1CFD475FA2015303`
- DEPENDENCY:STRUCTURED_PARAMETER_GREEN_AUDIT_ABORTED_V1.md
  `44C4EE2AE86EC641430F64BBCF2D80AE20ED18D02DED55A5F07CE65D9A2FAB4F`
- DEPENDENCY:results/transformer_v3_relinearized_prefix_panel_audit.json
  `08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B`
- DEPENDENCY:results/transformer_direct_image_green_panel_audit.json
  `931CBF5750510C49DEB92F16F77E8CCA355C7969A18BCF4EFA1A0701335ED705`
