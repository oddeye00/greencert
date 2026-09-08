"""Outward whole-ball neural derivative majorants through order three.

The center network is evaluated with Arb. Only nonnegative scalar majorants
are transported off the center; a parameter box is never substituted for the
Euclidean ball. Point Grams retain shared-parameter covariance, while mixed
activation/weight majorants control its change. This is the outward version
of the existing sub-block transport, not a new derivative-bound theorem.

Supported adapter: three-token, smooth GELU causal Transformers, any positive
depth, optional pre-LayerNorm, learned affine normalization, no dropout. All
trainable parameters participate. Optional caches are a low-level proof
interface: callers must authenticate their anchor, pair, architecture and
normalization epsilons. Omit caches to compute every ingredient here.
"""
from dataclasses import dataclass
from math import factorial, nextafter, inf

import numpy as np
from flint import arb, arb_mat

from arb_deep_point_gram import network_point_gains, select as matrix_select
from arb_matrix_bounds import frobenius_upper, spectral_upper, upper_float
from arb_transformer_objective import unflatten_arb, _linear, _gelu, _softmax_row


def nonnegative(value):
    value = value if isinstance(value, arb) else arb(float(value))
    if not value.is_finite() or not bool(value >= 0):
        raise ValueError("finite nonnegative majorant required")
    return value.upper()


def bound(value):
    """Discard roundoff uncertainty upward, never through a float midpoint."""
    if not value.is_finite():
        raise OverflowError("unresolved or nonfinite analytic bound")
    result = value.upper()
    if not bool(result >= 0):
        raise ValueError("negative analytic bound")
    return result


def minimum(*values):
    return min(bound(v) for v in values)


def norm(center):
    matrices = center if isinstance(center, tuple) else (center,)
    return bound(sum((frobenius_upper(m)**2 for m in matrices), arb(0)).sqrt())


@dataclass
class Jet:
    center: arb_mat
    uncapped_distance: arb
    first: arb
    second: arb
    third: arb
    value_cap: arb | None = None

    def __post_init__(self):
        for key in ("uncapped_distance", "first", "second", "third"):
            setattr(self, key, nonnegative(getattr(self, key)))
        if self.value_cap is not None:
            self.value_cap = nonnegative(self.value_cap)

    @property
    def distance(self):
        return (self.uncapped_distance if self.value_cap is None else
                minimum(self.uncapped_distance, self.value_cap + norm(self.center)))

    @property
    def value(self):
        result = bound(norm(self.center) + self.distance)
        return result if self.value_cap is None else minimum(result, self.value_cap)

    def scalars(self):
        return {key: upper_float(getattr(self, key))
                for key in ("value", "distance", "first", "second", "third")}


def add(left, right):
    return Jet(left.center + right.center,
               bound(left.distance + right.distance),
               bound(left.first + right.first), bound(left.second + right.second),
               bound(left.third + right.third))


def point_tighten(source, gain, radius):
    gain = nonnegative(gain)
    return Jet(source.center,
               minimum(source.distance, gain*radius + source.second*radius**2/2),
               minimum(source.first, gain + source.second*radius),
               source.second, source.third, source.value_cap)


def linear(source, weight, bias, radius, weight_gain):
    bias_gain = arb(source.center.nrows()).sqrt() if bias is not None else arb(0)
    operator = bound(nonnegative(weight_gain) + radius)
    return Jet(_linear(source.center, weight, bias),
               bound(operator*source.distance + radius*(norm(source.center)+bias_gain)),
               bound(operator*source.first + source.value + bias_gain),
               bound(operator*source.second + 2*source.first),
               bound(operator*source.third + 3*source.second))


def radial_bounds(center, displacement, epsilon):
    """Spherical-lift bounds on the WHOLE activation ball, with lower r."""
    displacement = nonnegative(displacement)
    epsilon = nonnegative(epsilon)
    if not bool(epsilon > 0):
        raise ValueError("strictly positive actual LayerNorm epsilon required")
    width = center.ncols()
    centered = arb_mat(center.nrows(), width)
    lowers = []
    for t in range(center.nrows()):
        mean = sum((center[t,k] for k in range(width)), arb(0))/width
        for k in range(width):
            centered[t,k] = center[t,k]-mean
        # Absolute-value lower endpoints avoid squaring an interval across 0.
        square_lower = sum((max(arb(0), abs(centered[t,k]).lower())**2
                            for k in range(width)), arb(0)).lower()
        lowers.append(max(arb(0), square_lower).sqrt().lower())
    radial_lower = max(arb(0), (min(lowers)-displacement).lower())
    rmin = (radial_lower**2/width + epsilon).sqrt().lower()
    if not bool(rmin > 0):
        raise ValueError("normalization domain could not be resolved")
    constants = (bound(1/rmin), bound(2/(arb(3*width).sqrt()*rmin**2)),
                 bound(3/(width*rmin**3)))
    return centered, rmin, constants


def layernorm(source, gamma, beta, radius, epsilon):
    centered, rmin, (a,b,c) = radial_bounds(source.center, source.distance, epsilon)
    tokens, width = centered.nrows(), centered.ncols()
    normalized = arb_mat(tokens, width)
    for t in range(tokens):
        denom = (sum((centered[t,k]**2 for k in range(width)), arb(0))/width
                 + arb(float(epsilon))).sqrt()
        if not bool(denom > 0):
            raise ValueError("center normalization domain unresolved")
        for k in range(width):
            normalized[t,k] = centered[t,k]/denom
    normal = Jet(normalized, bound(a*source.distance), bound(a*source.first),
                 bound(a*source.second+b*source.first**2),
                 bound(a*source.third+3*b*source.first*source.second+c*source.first**3),
                 bound(arb(tokens*width).sqrt()))
    gamma_gain = max(abs(v).upper() for v in gamma.entries())
    gain = bound(gamma_gain + radius)
    output = arb_mat(tokens,width,[normalized[t,k]*gamma[0,k]+beta[0,k]
                                  for t in range(tokens) for k in range(width)])
    ordinary = Jet(output,
        bound(gain*normal.distance+radius*(norm(normalized)+arb(tokens).sqrt())),
        bound(gain*normal.first+normal.value+arb(tokens).sqrt()),
        bound(gain*normal.second+2*normal.first), bound(gain*normal.third+3*normal.second))
    # Exact feature-wise 2x2 Grams of the shared learned (gamma,beta) block.
    eigenvalues = []
    for k in range(width):
        aa = sum((normalized[t,k]**2 for t in range(tokens)), arb(0))
        bb = sum((normalized[t,k] for t in range(tokens)), arb(0))
        # Frobenius-style endpoint bound on the nonnegative radicand avoids
        # interval cancellation when aa is close to the token count.
        disc = bound(abs(aa-tokens).upper()**2 + 4*abs(bb).upper()**2)
        eigenvalues.append(bound((aa.upper()+tokens+disc.sqrt())/2))
    lw = bound(max(eigenvalues).sqrt())
    _, _, (a0,_,_) = radial_bounds(source.center, arb(0), epsilon)
    lx = bound(gamma_gain*a0)
    bxx, bxw, cxxx, cxxw = bound(gain*b), a, bound(gain*c), b
    xg = bound(lx+bxx*source.distance+bxw*radius)
    wg = bound(lw+bxw*source.distance)
    distance = bound(lx*source.distance+lw*radius+bxx*source.distance**2/2
                     +bxw*source.distance*radius)
    first = bound(xg*source.first+wg)
    second = bound(xg*source.second+bxx*source.first**2+2*bxw*source.first)
    third = bound(xg*source.third+3*bxx*source.first*source.second+3*bxw*source.second
                  +cxxx*source.first**3+3*cxxw*source.first**2)
    result = Jet(output, minimum(ordinary.distance,distance), minimum(ordinary.first,first),
                 minimum(ordinary.second,second), minimum(ordinary.third,third))
    return result, {"rmin_lower": nextafter(float(rmin), -inf), "normalization_constants": [upper_float(v) for v in (a,b,c)],
                    "learned_parameter_point_gain": upper_float(lw)}


INDICES = tuple((a,b) for total in range(4) for a in range(total+1) for b in (total-a,))


def polynomial(value=0):
    result = {key: arb(0) for key in INDICES}
    result[0,0] = nonnegative(value)
    return result


def multiply(left, right):
    result = polynomial()
    for (a,b), x in left.items():
        for (c,d), y in right.items():
            if a+b+c+d <= 3:
                result[a+c,b+d] += x*y
    return {k: bound(v) for k,v in result.items()}


@dataclass
class Mixed:
    center: arb_mat | tuple
    distance: arb
    coefficients: dict


def mixed_input(center, distance):
    p = polynomial(bound(norm(center)+distance))
    p[1,0] = arb(1)
    return Mixed(center, distance, p)


def mixed_reshape(source, transform):
    return Mixed(transform(source.center), source.distance, dict(source.coefficients))


def mixed_linear(source, weight, bias, radius, weight_gain):
    wp = polynomial(bound(nonnegative(weight_gain)+radius))
    wp[0,1] = arb(1)
    p = multiply(wp, source.coefficients)
    bias_gain = arb(source.center.nrows()).sqrt() if bias is not None else arb(0)
    p[0,1] = bound(p[0,1]+bias_gain)
    distance = bound(wp[0,0]*source.distance+radius*(norm(source.center)+bias_gain))
    center = _linear(source.center,weight,bias)
    p[0,0] = bound(norm(center)+distance)
    return Mixed(center,distance,p)


def mixed_compose(source, center, constants, cap=None):
    a,b,c = constants
    delta = dict(source.coefficients)
    delta[0,0] = arb(0)
    square = multiply(delta,delta)
    cube = multiply(delta,square)
    p = {k: bound(a*delta[k]+b*square[k]/2+c*cube[k]/6) for k in INDICES}
    distance = bound(a*source.distance)
    p[0,0] = bound(norm(center)+distance)
    if cap is not None:
        p[0,0] = minimum(p[0,0],cap)
        distance = minimum(distance,cap+norm(center))
    return Mixed(center,distance,p)


def mixed_product(left,right,scale):
    if not isinstance(left.center,tuple) or len(left.center)!=len(right.center):
        raise ValueError("matching head-matrix tuples required")
    center = tuple((a*b)*scale for a,b in zip(left.center,right.center))
    distance = bound(scale*(norm(left.center)*right.distance+norm(right.center)*left.distance
                            +left.distance*right.distance))
    p = {k: bound(scale*v) for k,v in multiply(left.coefficients,right.coefficients).items()}
    p[0,0] = bound(norm(center)+distance)
    return Mixed(center,distance,p)


def split_heads(matrix, heads):
    tokens,width=matrix.nrows(),matrix.ncols()
    if heads < 1 or width%heads:
        raise ValueError("invalid head dimension")
    hd=width//heads
    return tuple(matrix_select(matrix,list(range(tokens)),list(range(h*hd,(h+1)*hd)))
                 for h in range(heads))


def join_heads(matrices):
    tokens,hd=matrices[0].nrows(),matrices[0].ncols()
    return arb_mat(tokens,hd*len(matrices),[m[t,k] for t in range(tokens)
                                          for m in matrices for k in range(hd)])


def causal_probabilities(scores):
    result=[]
    for score in scores:
        m=arb_mat(score.nrows(),score.ncols())
        for t in range(score.nrows()):
            for u,v in enumerate(_softmax_row([score[t,k] for k in range(t+1)])):
                m[t,u]=v
        result.append(m)
    return tuple(result)


def attention_mixed(center,distance,iw,ib,ow,ob,heads,radius,weight_gains):
    source=mixed_input(center,distance)
    tokens,width=center.nrows(),center.ncols()
    components=[]
    for index in range(3):
        rows=list(range(index*width,(index+1)*width))
        w=matrix_select(iw,rows,list(range(width)))
        b=None if ib is None else matrix_select(ib,[0],rows)
        jet=mixed_linear(source,w,b,radius,weight_gains[index])
        components.append(mixed_reshape(jet,lambda m:split_heads(m,heads)))
    q,k,v=components
    scores=mixed_product(q,mixed_reshape(k,lambda ms:tuple(m.transpose() for m in ms)),
                         1/arb(width//heads).sqrt())
    probabilities=mixed_compose(scores,causal_probabilities(scores.center),
         (arb(1)/2,arb(6).sqrt()/9,arb(1)/2),bound(arb(heads*tokens).sqrt()))
    attended=mixed_reshape(mixed_product(probabilities,v,arb(1)),join_heads)
    return mixed_linear(attended,ow,ob,radius,weight_gains[3])


def feedforward_mixed(center,distance,w1,b1,w2,b2,radius,weight_gains):
    hidden=mixed_linear(mixed_input(center,distance),w1,b1,radius,weight_gains[0])
    activation=arb_mat(hidden.center.nrows(),hidden.center.ncols(),[_gelu(v) for v in hidden.center.entries()])
    hidden=mixed_compose(hidden,activation,(arb(113)/100,arb(4)/5,arb(2)))
    return mixed_linear(hidden,w2,b2,radius,weight_gains[1])


def transported(source,mixed,gains,radius):
    m={f"{a}{b}":bound(value*factorial(a)*factorial(b))
       for (a,b),value in mixed.coefficients.items() if a+b}
    lx,lw=map(nonnegative,gains)
    bxx,bxw,bww=m["20"],m["11"],m["02"]
    xg=minimum(m["10"],lx+bxx*source.distance+bxw*radius)
    wg=minimum(m["01"],lw+bxw*source.distance+bww*radius)
    d=minimum(mixed.distance,lx*source.distance+lw*radius+bxx*source.distance**2/2
              +bxw*source.distance*radius+bww*radius**2/2)
    first=bound(xg*source.first+wg)
    second=bound(xg*source.second+bxx*source.first**2+2*bxw*source.first+bww)
    third=bound(xg*source.third+3*bxx*source.first*source.second+3*bxw*source.second
                +m["30"]*source.first**3+3*m["21"]*source.first**2+3*m["12"]*source.first+m["03"])
    return Jet(mixed.center,d,first,second,third)


def verified_weight_norms(parameter,spec,config):
    values=unflatten_arb([v if isinstance(v,arb) else arb(float(v)) for v in parameter],spec).values
    matrices={}
    width=config.model_dim
    for block in range(config.depth):
        p=f"blocks.{block}."
        ap=p+("self_attn." if config.normalization=="layernorm" else "attention.")
        for index in range(3):
            matrices[ap+f"in_proj_weight_slice{index}"]=matrix_select(values[ap+"in_proj_weight"],
                list(range(index*width,(index+1)*width)),list(range(width)))
        for key in (ap+"out_proj.weight",p+"linear1.weight",p+"linear2.weight"):
            matrices[key]=values[key]
    matrices["readout.weight"]=values["readout.weight"]
    return {key:spectral_upper(matrix) for key,matrix in matrices.items()}


def neural_envelope(parameter,pair,spec,config,radius,*,normalization_eps,
                    point_gains=None,local_gains=None,weight_norms=None):
    """All bounds apply on ||theta-parameter||_2 <= radius, for one pair."""
    radius=nonnegative(radius)
    if config.normalization not in ("none","layernorm") or config.depth<1:
        raise ValueError("unsupported architecture")
    if getattr(config,"dropout",0)!=0:
        raise ValueError("stochastic dropout is not covered")
    if config.model_dim<1 or config.heads<1 or config.model_dim%config.heads:
        raise ValueError("invalid attention dimensions")
    if config.normalization=="layernorm" and len(normalization_eps)!=config.depth:
        raise ValueError("actual per-layer epsilon values required")
    pair=np.asarray(pair).reshape(-1)
    if len(pair)!=2 or not np.issubdtype(pair.dtype,np.integer) or not ((pair>=0)&(pair<config.modulus)).all():
        raise ValueError("one valid modular token pair required")
    if (point_gains is None)!=(local_gains is None):
        raise ValueError("supply both point and local enclosures or neither")
    if point_gains is None:
        point_gains,local_gains,_=network_point_gains(parameter,pair,spec,config,normalization_eps=normalization_eps)
    if weight_norms is None:
        weight_norms=verified_weight_norms(parameter,spec,config)
    def weight(key):
        record=weight_norms[key]
        if not record.get("uses_verified_reconstruction_residual",False):
            raise ValueError("unverified weight-norm cache")
        return nonnegative(record["upper"])
    values=unflatten_arb([v if isinstance(v,arb) else arb(float(v)) for v in parameter],spec).values
    width=config.model_dim
    tokens=[int(pair[0]),int(pair[1]),config.modulus]
    embedding=arb_mat(3,width,[values["token_embedding.weight"][t,k]+values["position_embedding"][j,k]
                              for j,t in enumerate(tokens) for k in range(width)])
    gain=bound(arb(6).sqrt())
    hidden=point_tighten(Jet(embedding,bound(gain*radius),gain,arb(0),arb(0)),point_gains["embedding"],radius)
    trace=[{"stage":"embedding",**hidden.scalars()}]
    for block in range(config.depth):
        prefix=f"blocks.{block}."
        normed=config.normalization=="layernorm"
        attended_input=hidden
        if normed:
            attended_input,info=layernorm(hidden,values[prefix+"norm1.weight"],values[prefix+"norm1.bias"],
                                          radius,normalization_eps[block][0])
            attended_input=point_tighten(attended_input,point_gains[prefix+"norm1.output"],radius)
            trace.append({"stage":prefix+"norm1.output",**info,**attended_input.scalars()})
        ap=prefix+("self_attn." if normed else "attention.")
        mixed=attention_mixed(attended_input.center,attended_input.distance,values[ap+"in_proj_weight"],
            values[ap+"in_proj_bias"],values[ap+"out_proj.weight"],values[ap+"out_proj.bias"],config.heads,radius,
            [weight(ap+f"in_proj_weight_slice{k}") for k in range(3)]+[weight(ap+"out_proj.weight")])
        attended=transported(attended_input,mixed,local_gains[ap],radius)
        hidden=point_tighten(add(hidden,attended),point_gains[prefix+"attention.output"],radius)
        trace.append({"stage":prefix+"attention.output",**hidden.scalars()})
        feed_input=hidden
        if normed:
            feed_input,info=layernorm(hidden,values[prefix+"norm2.weight"],values[prefix+"norm2.bias"],
                                     radius,normalization_eps[block][1])
            feed_input=point_tighten(feed_input,point_gains[prefix+"norm2.output"],radius)
            trace.append({"stage":prefix+"norm2.output",**info,**feed_input.scalars()})
        mixed=feedforward_mixed(feed_input.center,feed_input.distance,values[prefix+"linear1.weight"],
             values[prefix+"linear1.bias"],values[prefix+"linear2.weight"],values[prefix+"linear2.bias"],radius,
             [weight(prefix+"linear1.weight"),weight(prefix+"linear2.weight")])
        feed=transported(feed_input,mixed,local_gains[prefix+"feedforward."],radius)
        hidden=point_tighten(add(hidden,feed),point_gains[prefix+"output"],radius)
        trace.append({"stage":prefix+"output",**hidden.scalars()})
    last=Jet(matrix_select(hidden.center,[2],list(range(width))),hidden.distance,hidden.first,hidden.second,hidden.third)
    output=linear(last,values["readout.weight"],None,radius,weight("readout.weight"))
    output=point_tighten(output,point_gains["logits"],radius)
    trace.append({"stage":"logits",**output.scalars()})
    return output,trace


def saturation_constants(logits,label,displacement):
    values=list(logits)
    if len(values)<2 or not isinstance(label,(int,np.integer)) or not 0<=label<len(values):
        raise ValueError("invalid classification label")
    if not all(v.is_finite() for v in values):
        raise ValueError("nonfinite logits")
    displacement=nonnegative(displacement)
    total=sum(((v-values[label]).exp() for k,v in enumerate(values) if k!=label),arb(0))
    a=bound(arb(2).sqrt())*displacement+total.log()
    # sigmoid is increasing, so use an upper endpoint BEFORE applying it.
    q=(1/(1+(-a.upper()).exp())).upper()
    q=min(arb(1),q)
    root2=arb(2).sqrt()
    # q+(1-q)q^3 is increasing on [0,1]. Endpoint arithmetic stays outward.
    return {"wrong_probability_upper":q,"loss_first":bound(root2*q),
            "loss_second":minimum(arb(1)/2,2*q),
            "loss_third":minimum(arb(6).sqrt()/9,2*root2*(q+(1-q)*q**3))}


def objective_drift(jet,constants,learning_rate):
    eta=nonnegative(learning_rate)
    third=(constants["loss_third"]*jet.first**3
           +3*constants["loss_second"]*jet.first*jet.second+constants["loss_first"]*jet.third)
    return bound(arb(2).sqrt()*eta*third)
