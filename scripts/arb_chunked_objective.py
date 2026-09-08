"""Memory-bounded outward objective/gradient/HVP over every training example.

Each chunk uses the identical dyadic parameters and direction. Multiplying
each mean by its exact integer sample count and dividing once by the total
preserves the full objective, including its once-weighted L2 term. Chunking
does not truncate parameters, samples, or derivative coordinates.
"""
import numpy as np
from flint import arb
from arb_reverse_transformer import arb_objective_gradient_hvp


def chunked_objective_gradient_hvp(parameter,pairs,labels,spec,config,*,normalization_eps,
                                    direction=None,chunk_size=8,progress=None):
    if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError("positive integer chunk size required")
    pairs=np.asarray(pairs.detach().cpu().numpy() if hasattr(pairs,"detach") else pairs)
    labels=np.asarray(labels.detach().cpu().numpy() if hasattr(labels,"detach") else labels)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)<1 or labels.shape!=(len(pairs),):
        raise ValueError("nonempty aligned examples/labels required")
    n=sum(spec.sizes);count=len(pairs)
    gradient=[arb(0) for _ in range(n)]
    hvp=None if direction is None else [arb(0) for _ in range(n)]
    objective=arb(0);directional=arb(0) if direction is not None else None;chunks=[]
    for start in range(0,count,chunk_size):
        end=min(count,start+chunk_size);size=end-start
        values=arb_objective_gradient_hvp(parameter,pairs[start:end],labels[start:end],spec,config,
            normalization_eps=normalization_eps,direction=direction)
        objective+=values["objective"]*size
        for k,v in enumerate(values["gradient"]):gradient[k]+=size*v
        if hvp is not None:
            directional+=values["directional_objective"]*size
            for k,v in enumerate(values["hvp"]):hvp[k]+=size*v
        chunks.append([start,end])
        if progress is not None:progress(start,end)
        del values
    return {"objective":objective/count,"gradient":[v/count for v in gradient],
        "hvp":None if hvp is None else [v/count for v in hvp],
        "directional_objective":None if directional is None else directional/count,
        "chunks":chunks,"examples":count,"parameters":n}
