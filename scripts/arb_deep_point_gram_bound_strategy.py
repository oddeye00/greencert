"""The structured point-Gram construction with an explicit norm strategy.

All activation/parameter Gram identities are unchanged and delegated to the
existing Arb geometry routines. The new argument chooses a proved PSD gain
bound, not a neural approximation. It must be included in the caller's
method seal. No global monkeypatch or live-worker source change is used.
"""
import numpy as np
import torch
from flint import arb, arb_mat

from arb_deep_point_gram import (kron_identity, select, identity, congruence,
                               attention_matrices, feedforward_matrices)
from arb_deep_point_gram_structured import normalized_gram, residual_gram
from arb_matrix_bounds import upper_float
from arb_transformer_objective import unflatten_arb, _linear


def block_gain(jacobian, tokens, width, gain_bound):
    # The feedforward activation Jacobian is block diagonal by token.
    gains = []
    for t in range(tokens):
        rows = list(range(t * width, (t + 1) * width))
        block = select(jacobian, rows, rows)
        gains.append(gain_bound(block * block.transpose()))
    return upper_float(max(gains))


def network_point_gains(parameter, pair, spec, config, *, normalization_eps,
                        gain_bound, local_gain_bound=None, return_matrices=False):
    if local_gain_bound is None:
        local_gain_bound = gain_bound
    if config.normalization not in ("none", "layernorm") or config.depth < 1:
        raise ValueError("unsupported model")
    if config.normalization == "layernorm" and len(normalization_eps) != config.depth:
        raise ValueError("actual normalization epsilons required")
    pair = np.asarray(pair.detach().cpu().numpy() if isinstance(pair, torch.Tensor) else pair).reshape(-1)
    if len(pair) != 2 or not np.issubdtype(pair.dtype, np.integer) or not ((pair >= 0) & (pair < config.modulus)).all():
        raise ValueError("valid modular pair required")
    values = unflatten_arb([v if isinstance(v, arb) else arb(float(v)) for v in parameter], spec).values
    tokens = [int(pair[0]), int(pair[1]), config.modulus]
    width = config.model_dim
    hidden = arb_mat(3, width, [values["token_embedding.weight"][token, k] + values["position_embedding"][t, k]
                               for t, token in enumerate(tokens) for k in range(width)])
    covariance = arb_mat(3, 3, [arb(int(tokens[t] == tokens[s]) + int(t == s))
                               for t in range(3) for s in range(3)])
    gram = kron_identity(covariance, width)
    gains, local, records = {}, {}, {}

    def record(name, g, c):
        gains[name] = upper_float(gain_bound(g))
        if return_matrices:
            records[name] = {"gram": g, "center": c}

    record("embedding", gram, hidden)
    for b in range(config.depth):
        prefix = f"blocks.{b}."
        normed = config.normalization == "layernorm"
        if normed:
            n, cross, gn = normalized_gram(hidden, gram, values[prefix + "norm1.weight"],
                values[prefix + "norm1.bias"], normalization_eps[b][0])
            record(prefix + "norm1.output", gn, n)
        else:
            n, cross, gn = hidden, gram, gram
        ap = prefix + ("self_attn." if normed else "attention.")
        attended, ja, ga = attention_matrices(n, values[ap + "in_proj_weight"], values[ap + "in_proj_bias"],
            values[ap + "out_proj.weight"], values[ap + "out_proj.bias"], config.heads)
        local[ap] = (upper_float(local_gain_bound(ja * ja.transpose())), upper_float(local_gain_bound(ga)))
        gram = residual_gram(gram, cross, gn, ja, ga)
        hidden = hidden + attended
        record(prefix + "attention.output", gram, hidden)
        if normed:
            n, cross, gn = normalized_gram(hidden, gram, values[prefix + "norm2.weight"],
                values[prefix + "norm2.bias"], normalization_eps[b][1])
            record(prefix + "norm2.output", gn, n)
        else:
            n, cross, gn = hidden, gram, gram
        feed, jf, gf = feedforward_matrices(n, values[prefix + "linear1.weight"], values[prefix + "linear1.bias"],
            values[prefix + "linear2.weight"], values[prefix + "linear2.bias"])
        local[prefix + "feedforward."] = (block_gain(jf, 3, width, local_gain_bound), upper_float(local_gain_bound(gf)))
        gram = residual_gram(gram, cross, gn, jf, gf)
        hidden = hidden + feed
        record(prefix + "output", gram, hidden)
    last = select(gram, list(range(2 * width, 3 * width)), list(range(2 * width, 3 * width)))
    readout = values["readout.weight"]
    output_gram = congruence(readout, last) + identity(config.modulus) * sum(
        (hidden[2, k] ** 2 for k in range(width)), arb(0))
    output = _linear(select(hidden, [2], list(range(width))), readout, None)
    record("logits", output_gram, output)
    return gains, local, records
