"""Outward data-loss HVPs with the quadratic penalty factored analytically.

For F(theta)=mean loss(theta)+lambda||theta||^2/2, H_F v is the
mean data-loss HVP plus lambda*v. Only the latter term moves outside the
example loop. The nonlinear objective, trainable coordinates and directions
are unchanged. Directions remain Arb intervals, including exact VJP b-a.
"""
import numpy as np
from flint import arb,arb_mat
from arb_reverse import leaf
from arb_reverse_transformer import _linear,_layernorm,_gelu,attention,cross_entropy


def data_direction_hvp(parameter,pairs,labels,spec,config,*,normalization_eps,direction):
    if config.depth<1 or config.normalization not in ("none","layernorm") or config.loss!="cross_entropy":
        raise ValueError("unsupported objective")
    if config.normalization=="layernorm" and (len(normalization_eps)!=config.depth or
                                              any(len(eps)!=2 for eps in normalization_eps)):
        raise ValueError("actual normalization epsilons required")
    parameter=np.asarray(parameter,dtype=np.float64).reshape(-1)
    if len(parameter)!=sum(spec.sizes) or not np.isfinite(parameter).all():
        raise ValueError("finite dyadic parameter center required")
    direction=[v if isinstance(v,arb) else arb(float(v)) for v in direction]
    if len(direction)!=len(parameter) or not all(v.is_finite() for v in direction):
        raise ValueError("one finite tangent enclosure per coordinate required")
    pairs=np.asarray(pairs.detach().cpu().numpy() if hasattr(pairs,"detach") else pairs)
    labels=np.asarray(labels.detach().cpu().numpy() if hasattr(labels,"detach") else labels)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)<1 or not np.issubdtype(pairs.dtype,np.integer):
        raise ValueError("nonempty integer token pairs required")
    if not ((pairs>=0)&(pairs<config.modulus)).all():raise ValueError("invalid modular tokens")
    if labels.shape!=(len(pairs),) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError("aligned integer labels required")
    values={};offset=0
    for name,shape,size in zip(spec.names,spec.shapes,spec.sizes):
        if len(shape) not in (1,2):raise ValueError("unsupported parameter shape")
        dims=(1,int(shape[0])) if len(shape)==1 else tuple(map(int,shape))
        values[name]=leaf(arb_mat(*dims,[arb(float(x)) for x in parameter[offset:offset+size]]),
                          arb_mat(*dims,direction[offset:offset+size]))
        offset+=size
    tokens=[token for pair in pairs for token in (int(pair[0]),int(pair[1]),config.modulus)]
    width=config.model_dim;n=len(pairs)
    hidden=values["token_embedding.weight"].gather(tokens,range(width))
    hidden=hidden+values["position_embedding"].gather(list(range(3))*n,range(width))
    for block in range(config.depth):
        prefix=f"blocks.{block}.";normed=config.normalization=="layernorm"
        x=_layernorm(hidden,values[prefix+"norm1.weight"],values[prefix+"norm1.bias"],normalization_eps[block][0]) if normed else hidden
        ap=prefix+("self_attn." if normed else "attention.")
        qkv=_linear(x,values[ap+"in_proj_weight"],values[ap+"in_proj_bias"])
        attended=attention(qkv,examples=n,width=width,heads=config.heads)
        hidden=hidden+_linear(attended,values[ap+"out_proj.weight"],values[ap+"out_proj.bias"])
        x=_layernorm(hidden,values[prefix+"norm2.weight"],values[prefix+"norm2.bias"],normalization_eps[block][1]) if normed else hidden
        hidden=hidden+_linear(_gelu(_linear(x,values[prefix+"linear1.weight"],values[prefix+"linear1.bias"])),
                             values[prefix+"linear2.weight"],values[prefix+"linear2.bias"])
    output=_linear(hidden.gather(list(range(2,3*n,3)),range(width)),values["readout.weight"])
    objective=cross_entropy(output,labels)
    objective.backward()
    products=[]
    for name in spec.names:
        adjoint=values[name].adjoint
        if adjoint is None:raise AssertionError("unvisited parameter leaf")
        products.extend(adjoint.tangent.entries())
    return products


def factored_chunked_hvp(parameter,pairs,labels,spec,config,*,normalization_eps,
                          direction,chunk_size=8,progress=None):
    pairs=np.asarray(pairs.detach().cpu().numpy() if hasattr(pairs,"detach") else pairs)
    labels=np.asarray(labels.detach().cpu().numpy() if hasattr(labels,"detach") else labels)
    if len(pairs)<1 or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError("nonempty examples and positive chunk size required")
    direction=[v if isinstance(v,arb) else arb(float(v)) for v in direction]
    penalty=arb(float(config.weight_decay))
    if not penalty.is_finite():raise ValueError("finite quadratic penalty required")
    total=[arb(0) for _ in range(sum(spec.sizes))]
    for start in range(0,len(pairs),chunk_size):
        end=min(len(pairs),start+chunk_size)
        values=data_direction_hvp(parameter,pairs[start:end],labels[start:end],spec,config,
                                  normalization_eps=normalization_eps,direction=direction)
        for k,v in enumerate(values):total[k]+=(end-start)*v
        if progress is not None:progress(start,end)
    return [v/len(pairs)+penalty*d for v,d in zip(total,direction)]
