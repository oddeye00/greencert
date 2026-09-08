"""Outward full-gradient/HVP backend for arbitrary supported Transformer depth.

Uses the same smooth three-token causal model as the existing evaluators.
Supported building blocks: affine maps, GELU, residual addition, optional
pre-LayerNorm, softmax attention, cross entropy, and all-parameter L2.
Width/depth are data, not hard-coded derivative formulas. Stochastic layers
and optimizer-specific certification beyond the evaluated objective are not
silently approximated by this module.
"""
import math
import numpy as np
import torch
from flint import arb, arb_mat
from arb_reverse import Node, Dual, leaf, constant, dual_zeros, dual_gather, hadamard


def _put_block(destination, source, rows, cols):
    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            destination.value[r,c]=source.value[i,j]
            if destination.tangent is not None:
                destination.tangent[r,c]=source.tangent[i,j]


def softmax_dual(scores, *, causal=False):
    value=arb_mat(scores.value.nrows(),scores.value.ncols())
    tangent=None if scores.tangent is None else arb_mat(value.nrows(),value.ncols())
    for i in range(value.nrows()):
        allowed=i+1 if causal else value.ncols()
        shift=max(float(scores.value[i,j].mid()) for j in range(allowed))
        exponentials=[(scores.value[i,j]-shift).exp() for j in range(allowed)]
        total=sum(exponentials,arb(0))
        probs=[v/total for v in exponentials]
        mean=arb(0) if tangent is None else sum((probs[j]*scores.tangent[i,j] for j in range(allowed)),arb(0))
        for j,p in enumerate(probs):
            value[i,j]=p
            if tangent is not None:
                tangent[i,j]=p*(scores.tangent[i,j]-mean)
    return Dual(value,tangent)


def _softmax_vjp(probabilities, adjoint):
    weighted=probabilities.multiply(adjoint)
    mean_value=[sum((weighted.value[i,j] for j in range(weighted.value.ncols())),arb(0))
                for i in range(weighted.value.nrows())]
    mean_tangent=None if weighted.tangent is None else [
        sum((weighted.tangent[i,j] for j in range(weighted.value.ncols())),arb(0))
        for i in range(weighted.value.nrows())]
    n,d=adjoint.value.nrows(),adjoint.value.ncols()
    repeated=Dual(arb_mat(n,d,[mean_value[i] for i in range(n) for _ in range(d)]),
        None if mean_tangent is None else arb_mat(n,d,[mean_tangent[i] for i in range(n) for _ in range(d)]))
    return probabilities.multiply(adjoint-repeated)


def attention(qkv, *, examples, width, heads):
    if qkv.shape != (3*examples,3*width) or heads < 1 or width % heads:
        raise ValueError("attention shape mismatch")
    head_dim=width//heads;scale=1/arb(head_dim).sqrt()
    result=dual_zeros(3*examples,width,qkv.dual.tangent is not None)
    records=[]
    for example in range(examples):
        rr=list(range(3*example,3*example+3))
        for head in range(heads):
            cc=list(range(head*head_dim,(head+1)*head_dim))
            q=dual_gather(qkv.dual,rr,cc)
            k=dual_gather(qkv.dual,rr,[c+width for c in cc])
            v=dual_gather(qkv.dual,rr,[c+2*width for c in cc])
            scores=q.matmul(k.transpose()).scale(scale)
            p=softmax_dual(scores,causal=True)
            _put_block(result,p.matmul(v),rr,cc)
            records.append((rr,cc,q,k,v,p))
    def back(g):
        grad=dual_zeros(*qkv.shape,qkv.dual.tangent is not None)
        for rr,cc,q,k,v,p in records:
            local=dual_gather(g,rr,cc)
            grad_p=local.matmul(v.transpose())
            grad_scores=_softmax_vjp(p,grad_p)
            _put_block(grad,grad_scores.matmul(k).scale(scale),rr,cc)
            _put_block(grad,grad_scores.transpose().matmul(q).scale(scale),rr,[c+width for c in cc])
            _put_block(grad,p.transpose().matmul(local),rr,[c+2*width for c in cc])
        return grad
    return Node(result,((qkv,back),))


def _linear(hidden,weight,bias=None):
    result=hidden.matmul(weight.transpose())
    if bias is not None:
        result=result+bias.gather([0]*hidden.shape[0],range(bias.shape[1]))
    return result


def _gelu(hidden):
    root2=arb(2).sqrt();root2pi=(2*arb.pi()).sqrt()
    def fn(x):
        phi=(-x*x/2).exp()/root2pi
        cdf=(1+(x/root2).erf())/2
        return x*cdf,cdf+x*phi,(2-x*x)*phi
    return hidden.unary(fn)


def _layernorm(hidden,gamma,beta,epsilon):
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("positive finite LayerNorm epsilon required")
    n,d=hidden.shape;mixed=hidden.dual.tangent is not None
    ones_col=constant(arb_mat(d,1,[arb(1)]*d),mixed=mixed)
    ones_row=constant(arb_mat(1,d,[arb(1)]*d),mixed=mixed)
    centered=hidden-hidden.matmul(ones_col).scale(arb(1)/d).matmul(ones_row)
    variance=centered.multiply(centered).matmul(ones_col).scale(arb(1)/d)
    variance=variance+constant(arb_mat(n,1,[arb(float(epsilon))]*n),mixed=mixed)
    def rsqrt(x):
        if not bool(x>0):
            raise ValueError("LayerNorm variance not provably positive")
        inv=1/x.sqrt()
        return inv,-inv/(2*x),3*inv/(4*x*x)
    normalized=centered.multiply(variance.unary(rsqrt).matmul(ones_row))
    return normalized.multiply(gamma.gather([0]*n,range(d)))+beta.gather([0]*n,range(d))


def cross_entropy(logits, labels):
    n,d=logits.shape
    if len(labels)!=n or any(int(label)<0 or int(label)>=d for label in labels):
        raise ValueError("invalid labels")
    probabilities=softmax_dual(logits.dual)
    derivative=probabilities.copy()
    value=arb(0)
    for i,label in enumerate(labels):
        shift=max(float(logits.dual.value[i,j].mid()) for j in range(d))
        value+=arb(shift)+sum(((logits.dual.value[i,j]-shift).exp() for j in range(d)),arb(0)).log()-logits.dual.value[i,int(label)]
        derivative.value[i,int(label)]-=1
    derivative=derivative.scale(arb(1)/n)
    tangent=None if logits.dual.tangent is None else arb_mat(1,1,[
        sum(hadamard(derivative.value,logits.dual.tangent).entries(),arb(0))])
    def back(g):
        repeated=Dual(arb_mat(n,d,[g.value[0,0]]*(n*d)),
                      None if g.tangent is None else arb_mat(n,d,[g.tangent[0,0]]*(n*d)))
        return derivative.multiply(repeated)
    return Node(Dual(arb_mat(1,1,[value/n]),tangent),((logits,back),))


def objective_graph(parameter,pairs,labels,spec,config,*,normalization_eps,direction=None):
    if config.depth<1 or config.normalization not in ("none","layernorm") or config.loss!="cross_entropy":
        raise ValueError("unsupported model; no approximate fallback")
    if config.normalization=="layernorm" and (len(normalization_eps)!=config.depth or
                                             any(len(eps)!=2 for eps in normalization_eps)):
        raise ValueError("both actual LayerNorm epsilons required at every block")
    parameter=np.asarray(parameter,dtype=np.float64).reshape(-1)
    direction=None if direction is None else np.asarray(direction,dtype=np.float64).reshape(-1)
    if len(parameter)!=sum(spec.sizes) or not np.isfinite(parameter).all():
        raise ValueError("invalid parameter vector")
    if direction is not None and (direction.shape!=parameter.shape or not np.isfinite(direction).all()):
        raise ValueError("invalid tangent vector")
    pairs=np.asarray(pairs.detach().cpu().numpy() if isinstance(pairs,torch.Tensor) else pairs)
    labels=np.asarray(labels.detach().cpu().numpy() if isinstance(labels,torch.Tensor) else labels)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)<1 or not np.issubdtype(pairs.dtype,np.integer):
        raise ValueError("nonempty integer token pairs required")
    if not bool(((pairs>=0)&(pairs<config.modulus)).all()):
        raise ValueError("invalid modular tokens")
    if labels.shape!=(len(pairs),) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError("integer labels required")
    values={};offset=0;mixed=direction is not None
    for name,shape,size in zip(spec.names,spec.shapes,spec.sizes):
        if len(shape) not in (1,2):raise ValueError("unsupported parameter shape")
        dims=(1,int(shape[0])) if len(shape)==1 else tuple(map(int,shape))
        v=arb_mat(*dims,[arb(float(x)) for x in parameter[offset:offset+size]])
        t=None if direction is None else arb_mat(*dims,[arb(float(x)) for x in direction[offset:offset+size]])
        values[name]=leaf(v,t);offset+=size
    tokens=[token for pair in pairs for token in (int(pair[0]),int(pair[1]),config.modulus)]
    width=config.model_dim;n=len(pairs)
    hidden=values["token_embedding.weight"].gather(tokens,range(width))
    hidden=hidden+values["position_embedding"].gather(list(range(3))*n,range(width))
    for block in range(config.depth):
        prefix=f"blocks.{block}.";norm=config.normalization=="layernorm"
        first=_layernorm(hidden,values[prefix+"norm1.weight"],values[prefix+"norm1.bias"],normalization_eps[block][0]) if norm else hidden
        ap=prefix+("self_attn." if norm else "attention.")
        qkv=_linear(first,values[ap+"in_proj_weight"],values[ap+"in_proj_bias"])
        attended=attention(qkv,examples=n,width=width,heads=config.heads)
        hidden=hidden+_linear(attended,values[ap+"out_proj.weight"],values[ap+"out_proj.bias"])
        second=_layernorm(hidden,values[prefix+"norm2.weight"],values[prefix+"norm2.bias"],normalization_eps[block][1]) if norm else hidden
        pre=_linear(second,values[prefix+"linear1.weight"],values[prefix+"linear1.bias"])
        hidden=hidden+_linear(_gelu(pre),values[prefix+"linear2.weight"],values[prefix+"linear2.bias"])
    output=_linear(hidden.gather(list(range(2,3*n,3)),range(width)),values["readout.weight"])
    objective=cross_entropy(output,labels)
    for node in values.values():
        objective=objective+node.multiply(node).sum().scale(arb(float(config.weight_decay))/2)
    return objective,values,output


def arb_objective_gradient_hvp(parameter,pairs,labels,spec,config,*,normalization_eps,direction=None):
    objective,values,_=objective_graph(parameter,pairs,labels,spec,config,
                                     normalization_eps=normalization_eps,direction=direction)
    objective.backward()
    gradients=[];products=[]
    for name in spec.names:
        adjoint=values[name].adjoint
        if adjoint is None:
            raise AssertionError("unvisited parameter leaf")
        gradients.extend(adjoint.value.entries())
        if direction is not None:
            products.extend(adjoint.tangent.entries())
    return {"objective":objective.dual.value[0,0],"gradient":gradients,
            "hvp":None if direction is None else products,
            "directional_objective":None if direction is None else objective.dual.tangent[0,0]}
