"""Outward forward logits/objective for deep pre-LayerNorm Transformers.

This extends only forward evaluation. It neither encloses optimizer gradients
nor validates the floating-point training program.
"""
from __future__ import annotations
import numpy as np
import torch
from flint import arb, arb_mat
from arb_transformer_objective import ArbParameterMap, _as_arb, _linear, _gelu, _softmax_row, unflatten_arb


def layernorm(hidden, gamma, beta, epsilon):
    """Biased per-token variance, exactly matching the mathematical torch map."""
    if not float(epsilon)>0:
        raise ValueError("LayerNorm epsilon must be positive")
    width=hidden.ncols()
    output=arb_mat(hidden.nrows(),width)
    for row in range(hidden.nrows()):
        mean=sum((hidden[row,k] for k in range(width)),arb(0))/width
        centered=[hidden[row,k]-mean for k in range(width)]
        variance=sum((x*x for x in centered),arb(0))/width
        denominator=(variance+arb(float(epsilon))).sqrt()
        for k,value in enumerate(centered):
            output[row,k]=value/denominator*gamma[0,k]+beta[0,k]
    return output


def _attention(
    hidden: arb_mat,
    parameter: ArbParameterMap,
    *,
    examples: int,
    model_dim: int,
    heads: int,
    prefix: str,
) -> arb_mat:
    values = parameter.values
    qkv = _linear(
        hidden,
        values[prefix+"in_proj_weight"],
        values[prefix+"in_proj_bias"],
    )
    head_dim = model_dim // heads
    if head_dim * heads != model_dim:
        raise ValueError("model dimension must divide the number of heads")
    joined = arb_mat(examples * 3, model_dim)
    scale = arb(head_dim).sqrt()
    for example in range(examples):
        base_row = 3 * example
        for head in range(heads):
            q = arb_mat(3, head_dim)
            k = arb_mat(3, head_dim)
            v = arb_mat(3, head_dim)
            base_col = head * head_dim
            for token in range(3):
                for col in range(head_dim):
                    q[token, col] = qkv[base_row + token, base_col + col]
                    k[token, col] = qkv[
                        base_row + token, model_dim + base_col + col
                    ]
                    v[token, col] = qkv[
                        base_row + token, 2 * model_dim + base_col + col
                    ]
            scores = (q * k.transpose()) / scale
            probabilities = arb_mat(3, 3)
            for query in range(3):
                allowed = [scores[query, key] for key in range(query + 1)]
                row = _softmax_row(allowed)
                for key, probability in enumerate(row):
                    probabilities[query, key] = probability
            attended = probabilities * v
            for token in range(3):
                for col in range(head_dim):
                    joined[base_row + token, base_col + col] = attended[token, col]
    return _linear(
        joined,
        values[prefix+"out_proj.weight"],
        values[prefix+"out_proj.bias"],
    )


def arb_deep_logits(parameter,pairs,spec,config,*,normalization_eps):
    if config.normalization not in ("none","layernorm") or config.depth<1:
        raise ValueError("unsupported architecture")
    if config.normalization=="layernorm" and len(normalization_eps)!=config.depth:
        raise ValueError("supply the two actual LayerNorm epsilons per block")
    flat=tuple(_as_arb(value) for value in parameter)
    mapped=unflatten_arb(flat,spec);values=mapped.values
    pairs=np.asarray(pairs.detach().cpu().numpy() if isinstance(pairs,torch.Tensor) else pairs,dtype=np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)<1:
        raise ValueError("nonempty pairs with two modular tokens required")
    if not bool(((pairs>=0)&(pairs<config.modulus)).all()):
        raise ValueError("invalid modular token")
    examples=len(pairs);width=config.model_dim
    hidden=arb_mat(3*examples,width)
    for i,(left,right) in enumerate(pairs):
        for t,token in enumerate((int(left),int(right),config.modulus)):
            for k in range(width):
                hidden[3*i+t,k]=values["token_embedding.weight"][token,k]+values["position_embedding"][t,k]
    for block in range(config.depth):
        prefix=f"blocks.{block}."
        use_norm=config.normalization=="layernorm"
        attended_input=layernorm(hidden,values[prefix+"norm1.weight"],values[prefix+"norm1.bias"],
                                normalization_eps[block][0]) if use_norm else hidden
        attended=_attention(attended_input,mapped,examples=examples,model_dim=width,heads=config.heads,
                            prefix=prefix+("self_attn." if use_norm else "attention."))
        hidden=hidden+attended
        feed_input=layernorm(hidden,values[prefix+"norm2.weight"],values[prefix+"norm2.bias"],
                            normalization_eps[block][1]) if use_norm else hidden
        pre=_linear(feed_input,values[prefix+"linear1.weight"],values[prefix+"linear1.bias"])
        activated=arb_mat(pre.nrows(),pre.ncols(),[_gelu(value) for value in pre.entries()])
        hidden=hidden+_linear(activated,values[prefix+"linear2.weight"],values[prefix+"linear2.bias"])
    last=arb_mat(examples,width)
    for i in range(examples):
        for k in range(width):last[i,k]=hidden[3*i+2,k]
    return _linear(last,values["readout.weight"],None)


def arb_deep_objective(parameter,pairs,labels,spec,config,*,normalization_eps):
    if config.loss!="cross_entropy":
        raise ValueError("cross-entropy objective required")
    flat=tuple(_as_arb(value) for value in parameter)
    output=arb_deep_logits(flat,pairs,spec,config,normalization_eps=normalization_eps)
    labels=np.asarray(labels.detach().cpu().numpy() if isinstance(labels,torch.Tensor) else labels,dtype=np.int64)
    if labels.shape!=(output.nrows(),) or not bool(((labels>=0)&(labels<config.modulus)).all()):
        raise ValueError("invalid labels")
    loss=arb(0)
    for i,label in enumerate(labels):
        row=[output[i,k] for k in range(output.ncols())]
        shift=max(float(value.mid()) for value in row)
        normalizer=arb(shift)+sum(((v-shift).exp() for v in row),arb(0)).log()
        loss+=normalizer-row[int(label)]
    regularizer=sum((v*v for v in flat),arb(0))*arb(config.weight_decay)/2
    return loss/output.nrows()+regularizer
