"""Exact structured LayerNorm products and residual Gram product reuse.

For normalized input n(x), cache C=Jn G and Gn_total=Jn G Jn^T+G_affine.
The residual x+a(n(x),w) has Gram
  G + Q + Q^T + Ja Gn_total Ja^T + G_attention,  Q=Ja C.
This is the same Gram as the dense implementation. Each token's LayerNorm
Jacobian is diagonal plus two rank-one terms, so Jn G needs no dense Jn.
Every operation remains in Arb; no floating-point error model is introduced.
"""
import numpy as np
import torch
from flint import arb,arb_mat

from arb_deep_point_gram import (kron_identity,select,identity,congruence,
                               attention_matrices,feedforward_matrices)
from arb_matrix_bounds import psd_gain_upper,upper_float
from arb_transformer_objective import unflatten_arb,_linear


def normalization_geometry(x,gamma,beta,epsilon):
    tokens,width=x.nrows(),x.ncols()
    output=arb_mat(tokens,width);normalized=arb_mat(tokens,width);factors=[]
    for t in range(tokens):
        mean=sum((x[t,k] for k in range(width)),arb(0))/width
        c=[x[t,k]-mean for k in range(width)]
        r=(sum((v*v for v in c),arb(0))/width+arb(float(epsilon))).sqrt()
        if not bool(r>0):raise ValueError("normalization domain unresolved")
        diag=[gamma[0,k]/r for k in range(width)]
        u=[gamma[0,k]/(width*r) for k in range(width)]
        v=[gamma[0,k]*c[k]/(width*r**3) for k in range(width)]
        factors.append((c,diag,u,v))
        for k in range(width):
            normalized[t,k]=c[k]/r
            output[t,k]=normalized[t,k]*gamma[0,k]+beta[0,k]
    gram=arb_mat(tokens*width,tokens*width)
    for t in range(tokens):
        for s in range(tokens):
            for k in range(width):
                gram[t*width+k,s*width+k]=normalized[t,k]*normalized[s,k]+1
    return output,factors,gram


def normalization_left(factors,matrix):
    tokens=len(factors);width=len(factors[0][0])
    if matrix.nrows()!=tokens*width:raise ValueError("normalization product dimension mismatch")
    result=arb_mat(matrix.nrows(),matrix.ncols())
    for t,(c,diag,u,v) in enumerate(factors):
        block=select(matrix,list(range(t*width,(t+1)*width)),list(range(matrix.ncols())))
        summary=arb_mat(2,width,[arb(1)]*width+c)*block
        for i in range(width):
            for j in range(matrix.ncols()):
                result[t*width+i,j]=diag[i]*block[i,j]-u[i]*summary[0,j]-v[i]*summary[1,j]
    return result


def normalized_gram(hidden,gram,gamma,beta,epsilon):
    output,factors,affine=normalization_geometry(hidden,gamma,beta,epsilon)
    cross=normalization_left(factors,gram)
    normalized=normalization_left(factors,cross.transpose()).transpose()+affine
    return output,cross,normalized


def residual_gram(gram,cross,normalized,jacobian,local):
    q=jacobian*cross
    return gram+q+q.transpose()+congruence(jacobian,normalized)+local


def block_jacobian_gain(jacobian,tokens,width):
    # feedforward_matrices constructs a block-diagonal activation Jacobian.
    gains=[]
    for t in range(tokens):
        rows=list(range(t*width,(t+1)*width))
        block=select(jacobian,rows,rows)
        gains.append(psd_gain_upper(block*block.transpose()))
    return upper_float(max(gains))


def network_point_gains(parameter,pair,spec,config,*,normalization_eps,return_matrices=False):
    if config.normalization not in ("none","layernorm") or config.depth<1:
        raise ValueError("unsupported model")
    if config.normalization=="layernorm" and len(normalization_eps)!=config.depth:
        raise ValueError("actual normalization epsilons required")
    pair=np.asarray(pair.detach().cpu().numpy() if isinstance(pair,torch.Tensor) else pair).reshape(-1)
    if len(pair)!=2 or not np.issubdtype(pair.dtype,np.integer) or not ((pair>=0)&(pair<config.modulus)).all():
        raise ValueError("valid modular pair required")
    values=unflatten_arb([v if isinstance(v,arb) else arb(float(v)) for v in parameter],spec).values
    tokens=[int(pair[0]),int(pair[1]),config.modulus];width=config.model_dim
    hidden=arb_mat(3,width,[values["token_embedding.weight"][token,k]+values["position_embedding"][t,k]
                           for t,token in enumerate(tokens) for k in range(width)])
    covariance=arb_mat(3,3,[arb(int(tokens[t]==tokens[s])+int(t==s)) for t in range(3) for s in range(3)])
    gram=kron_identity(covariance,width);gains={};local={};records={}
    def record(name,g,c):
        gains[name]=upper_float(psd_gain_upper(g))
        if return_matrices:records[name]={"gram":g,"center":c}
    record("embedding",gram,hidden)
    for b in range(config.depth):
        prefix=f"blocks.{b}.";normed=config.normalization=="layernorm"
        if normed:
            n,cross,gn=normalized_gram(hidden,gram,values[prefix+"norm1.weight"],
                                     values[prefix+"norm1.bias"],normalization_eps[b][0])
            record(prefix+"norm1.output",gn,n)
        else:n,cross,gn=hidden,gram,gram
        ap=prefix+("self_attn." if normed else "attention.")
        attended,ja,ga=attention_matrices(n,values[ap+"in_proj_weight"],values[ap+"in_proj_bias"],
                                        values[ap+"out_proj.weight"],values[ap+"out_proj.bias"],config.heads)
        local[ap]=(upper_float(psd_gain_upper(ja*ja.transpose())),upper_float(psd_gain_upper(ga)))
        gram=residual_gram(gram,cross,gn,ja,ga)
        hidden=hidden+attended;record(prefix+"attention.output",gram,hidden)
        if normed:
            n,cross,gn=normalized_gram(hidden,gram,values[prefix+"norm2.weight"],
                                     values[prefix+"norm2.bias"],normalization_eps[b][1])
            record(prefix+"norm2.output",gn,n)
        else:n,cross,gn=hidden,gram,gram
        feed,jf,gf=feedforward_matrices(n,values[prefix+"linear1.weight"],values[prefix+"linear1.bias"],
                                       values[prefix+"linear2.weight"],values[prefix+"linear2.bias"])
        local[prefix+"feedforward."]=(block_jacobian_gain(jf,3,width),upper_float(psd_gain_upper(gf)))
        gram=residual_gram(gram,cross,gn,jf,gf)
        hidden=hidden+feed;record(prefix+"output",gram,hidden)
    last=select(gram,list(range(2*width,3*width)),list(range(2*width,3*width)))
    readout=values["readout.weight"]
    output_gram=congruence(readout,last)+identity(config.modulus)*sum((hidden[2,k]**2 for k in range(width)),arb(0))
    output=_linear(select(hidden,[2],list(range(width))),readout,None)
    record("logits",output_gram,output)
    return gains,local,records
