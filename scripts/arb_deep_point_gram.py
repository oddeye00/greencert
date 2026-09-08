"""Outward activation-space point Grams; no full parameter Jacobian.

Each output is an inclusion for the exact network at the supplied dyadic
weights. Shared parameters retain their cross-token covariance. This module
supplies point gains only; whole-ball higher derivative transport is separate.
"""
import numpy as np
import torch
from flint import arb,arb_mat
from arb_transformer_objective import unflatten_arb,_linear,_gelu,_softmax_row
from arb_matrix_bounds import psd_gain_upper,upper_float


def identity(size):
    value=arb_mat(size,size)
    for i in range(size):value[i,i]=1
    return value


def block_diag(blocks):
    rows=sum(b.nrows() for b in blocks);cols=sum(b.ncols() for b in blocks)
    output=arb_mat(rows,cols);ri=ci=0
    for block in blocks:
        for i in range(block.nrows()):
            for j in range(block.ncols()):output[ri+i,ci+j]=block[i,j]
        ri+=block.nrows();ci+=block.ncols()
    return output


def kron_identity(matrix,width):
    result=arb_mat(matrix.nrows()*width,matrix.ncols()*width)
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            for k in range(width):result[i*width+k,j*width+k]=matrix[i,j]
    return result


def select(matrix,rows,cols):
    return arb_mat(len(rows),len(cols),[matrix[i,j] for i in rows for j in cols])


def congruence(jacobian,gram):
    return jacobian*gram*jacobian.transpose()


def plus_ones(matrix):
    return arb_mat(matrix.nrows(),matrix.ncols(),[v+1 for v in matrix.entries()])


def layernorm_matrices(x,gamma,beta,epsilon):
    tokens,width=x.nrows(),x.ncols()
    normalized=arb_mat(tokens,width);output=arb_mat(tokens,width);blocks=[]
    for t in range(tokens):
        mean=sum((x[t,k] for k in range(width)),arb(0))/width
        c=[x[t,k]-mean for k in range(width)]
        r=(sum((v*v for v in c),arb(0))/width+arb(float(epsilon))).sqrt()
        if not bool(r>0):raise ValueError("normalization domain unresolved")
        block=arb_mat(width,width)
        for i in range(width):
            normalized[t,i]=c[i]/r
            output[t,i]=normalized[t,i]*gamma[0,i]+beta[0,i]
            for j in range(width):
                block[i,j]=gamma[0,i]*((arb(int(i==j))-arb(1)/width)/r-c[i]*c[j]/(width*r**3))
        blocks.append(block)
    gram=arb_mat(tokens*width,tokens*width)
    for t in range(tokens):
        for u in range(tokens):
            for d in range(width):gram[t*width+d,u*width+d]=normalized[t,d]*normalized[u,d]+1
    return output,block_diag(blocks),gram


def feedforward_matrices(x,w1,b1,w2,b2):
    tokens,width=x.nrows(),x.ncols()
    pre=_linear(x,w1,b1)
    activation=arb_mat(pre.nrows(),pre.ncols(),[_gelu(v) for v in pre.entries()])
    root2=arb(2).sqrt();root2pi=(2*arb.pi()).sqrt()
    weighted=[];jacobians=[]
    for t in range(tokens):
        derivative=[(1+(pre[t,h]/root2).erf())/2+pre[t,h]*(-pre[t,h]**2/2).exp()/root2pi
                    for h in range(w1.nrows())]
        current=arb_mat(width,w1.nrows(),[w2[i,h]*derivative[h] for i in range(width) for h in range(w1.nrows())])
        weighted.append(current);jacobians.append(current*w1)
    xcov=plus_ones(x*x.transpose());zcov=plus_ones(activation*activation.transpose())
    gram=arb_mat(tokens*width,tokens*width)
    for t in range(tokens):
        for u in range(tokens):
            covariance=(weighted[t]*weighted[u].transpose())*xcov[t,u]
            for i in range(width):
                for j in range(width):
                    gram[t*width+i,u*width+j]=covariance[i,j]+(zcov[t,u] if i==j else 0)
    return _linear(activation,w2,b2),block_diag(jacobians),gram


def attention_matrices(x,in_weight,in_bias,out_weight,out_bias,heads):
    tokens,width=x.nrows(),x.ncols();hd=width//heads
    if heads<1 or hd*heads!=width:raise ValueError("head width mismatch")
    qkv=_linear(x,in_weight,in_bias)
    covariance=x*x.transpose()
    if in_bias is not None:covariance=plus_ones(covariance)
    covariance=kron_identity(covariance,hd)
    jacobian=arb_mat(tokens*width,tokens*width)
    gram=arb_mat(tokens*width,tokens*width)
    joined=arb_mat(tokens,width)
    for head in range(heads):
        cols=list(range(head*hd,(head+1)*hd))
        q,k,v=[select(qkv,list(range(tokens)),[c+offset*width for c in cols]) for offset in range(3)]
        scores=(q*k.transpose())/arb(hd).sqrt()
        p=arb_mat(tokens,tokens)
        for t in range(tokens):
            for u,prob in enumerate(_softmax_row([scores[t,j] for j in range(t+1)])):p[t,u]=prob
        attended=p*v
        for t in range(tokens):
            for h,c in enumerate(cols):joined[t,c]=attended[t,h]
        jqblocks=[];jk=arb_mat(tokens*hd,tokens*hd)
        for t in range(tokens):
            sj=arb_mat(tokens,tokens,[(p[t,i] if i==j else 0)-p[t,i]*p[t,j]
                                     for i in range(tokens) for j in range(tokens)])
            dv=(v.transpose()*sj)/arb(hd).sqrt()
            jqblocks.append(dv*k)
            for i in range(hd):
                for u in range(tokens):
                    for j in range(hd):jk[t*hd+i,u*hd+j]=dv[i,u]*q[t,j]
        jq=block_diag(jqblocks);jv=kron_identity(p,hd)
        lifted=[block_diag([select(in_weight,[c+off*width for c in cols],list(range(width)))]*tokens)
                for off in range(3)]
        jh=jq*lifted[0]+jk*lifted[1]+jv*lifted[2]
        gh=congruence(jq,covariance)+congruence(jk,covariance)+congruence(jv,covariance)
        row_indices=[t*width+c for t in range(tokens) for c in cols]
        for i,r in enumerate(row_indices):
            for j in range(tokens*width):jacobian[r,j]=jh[i,j]
            for j,c in enumerate(row_indices):gram[r,c]=gh[i,j]
    projection=block_diag([out_weight]*tokens)
    gram=congruence(projection,gram)
    output_cov=joined*joined.transpose()
    if out_bias is not None:output_cov=plus_ones(output_cov)
    gram+=kron_identity(output_cov,width)
    return _linear(joined,out_weight,out_bias),projection*jacobian,gram


def network_point_gains(parameter,pair,spec,config,*,normalization_eps,return_matrices=False):
    values=unflatten_arb([v if isinstance(v,arb) else arb(float(v)) for v in parameter],spec).values
    pair=np.asarray(pair.detach().cpu().numpy() if isinstance(pair,torch.Tensor) else pair).reshape(-1)
    if len(pair)!=2 or not np.issubdtype(pair.dtype,np.integer) or not bool(((pair>=0)&(pair<config.modulus)).all()):
        raise ValueError("one valid modular token pair required")
    if config.normalization not in ("none","layernorm") or config.depth<1:
        raise ValueError("unsupported model")
    if config.normalization=="layernorm" and len(normalization_eps)!=config.depth:
        raise ValueError("actual normalization epsilons required")
    tokens=[int(pair[0]),int(pair[1]),config.modulus];width=config.model_dim
    hidden=arb_mat(3,width,[values["token_embedding.weight"][token,d]+values["position_embedding"][t,d]
                         for t,token in enumerate(tokens) for d in range(width)])
    covariance=arb_mat(3,3,[arb(int(tokens[t]==tokens[u])+int(t==u)) for t in range(3) for u in range(3)])
    gram=kron_identity(covariance,width);eye=identity(3*width)
    gains={};local={};records={}
    def record(name,g,c):
        gains[name]=upper_float(psd_gain_upper(g))
        if return_matrices:records[name]={"gram":g,"center":c}
    record("embedding",gram,hidden)
    for block in range(config.depth):
        prefix=f"blocks.{block}.";norm=config.normalization=="layernorm"
        if norm:
            n1,jn1,gn1=layernorm_matrices(hidden,values[prefix+"norm1.weight"],values[prefix+"norm1.bias"],normalization_eps[block][0])
            record(prefix+"norm1.output",congruence(jn1,gram)+gn1,n1)
        else:n1,jn1,gn1=hidden,eye,arb_mat(3*width,3*width)
        ap=prefix+("self_attn." if norm else "attention.")
        attended,ja,ga=attention_matrices(n1,values[ap+"in_proj_weight"],values[ap+"in_proj_bias"],
                                         values[ap+"out_proj.weight"],values[ap+"out_proj.bias"],config.heads)
        local[ap]=(upper_float(psd_gain_upper(ja*ja.transpose())),upper_float(psd_gain_upper(ga)))
        gram=congruence(eye+ja*jn1,gram)+congruence(ja,gn1)+ga
        hidden=hidden+attended;record(prefix+"attention.output",gram,hidden)
        if norm:
            n2,jn2,gn2=layernorm_matrices(hidden,values[prefix+"norm2.weight"],values[prefix+"norm2.bias"],normalization_eps[block][1])
            record(prefix+"norm2.output",congruence(jn2,gram)+gn2,n2)
        else:n2,jn2,gn2=hidden,eye,arb_mat(3*width,3*width)
        feed,jf,gf=feedforward_matrices(n2,values[prefix+"linear1.weight"],values[prefix+"linear1.bias"],
                                       values[prefix+"linear2.weight"],values[prefix+"linear2.bias"])
        local[prefix+"feedforward."]=(upper_float(psd_gain_upper(jf*jf.transpose())),upper_float(psd_gain_upper(gf)))
        gram=congruence(eye+jf*jn2,gram)+congruence(jf,gn2)+gf
        hidden=hidden+feed;record(prefix+"output",gram,hidden)
    weight=values["readout.weight"]
    last_gram=select(gram,list(range(2*width,3*width)),list(range(2*width,3*width)))
    output_gram=congruence(weight,last_gram)
    hidden_square=sum((hidden[2,k]**2 for k in range(width)),arb(0))
    output_gram+=identity(config.modulus)*hidden_square
    output=_linear(select(hidden,[2],list(range(width))),weight,None)
    record("logits",output_gram,output)
    return gains,local,records
