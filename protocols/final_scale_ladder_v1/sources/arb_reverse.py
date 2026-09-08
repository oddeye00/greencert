"""Small outward reverse-over-forward matrix AD engine.

All primal, tangent, adjoint, and adjoint-tangent operations use Arb.
An optional dyadic leaf tangent yields a full Hessian-vector enclosure after
one reverse pass of a scalar objective. No randomized norm bound is needed.
This is classical forward-over-reverse differentiation with interval scalars.
"""
from __future__ import annotations
from dataclasses import dataclass
from flint import arb, arb_mat


def zeros_like(value):
    return arb_mat(value.nrows(), value.ncols())


def hadamard(a, b):
    if (a.nrows(), a.ncols()) != (b.nrows(), b.ncols()):
        raise ValueError("elementwise shape mismatch")
    return arb_mat(a.nrows(), a.ncols(), [x*y for x,y in zip(a.entries(),b.entries())])


@dataclass
class Dual:
    value: arb_mat
    tangent: arb_mat | None = None

    def __add__(self, other):
        if (self.tangent is None) != (other.tangent is None):
            raise ValueError("mixed tangent modes")
        return Dual(self.value+other.value,
                    None if self.tangent is None else self.tangent+other.tangent)

    def __neg__(self):
        return self.scale(-1)

    def __sub__(self, other):
        return self + (-other)

    def scale(self, coefficient):
        return Dual(self.value*coefficient,
                    None if self.tangent is None else self.tangent*coefficient)

    def multiply(self, other):
        if (self.tangent is None) != (other.tangent is None):
            raise ValueError("mixed tangent modes")
        return Dual(hadamard(self.value,other.value), None if self.tangent is None else
                    hadamard(self.tangent,other.value)+hadamard(self.value,other.tangent))

    def matmul(self, other):
        if (self.tangent is None) != (other.tangent is None):
            raise ValueError("mixed tangent modes")
        return Dual(self.value*other.value, None if self.tangent is None else
                    self.tangent*other.value+self.value*other.tangent)

    def transpose(self):
        return Dual(self.value.transpose(), None if self.tangent is None else self.tangent.transpose())

    def copy(self):
        return Dual(arb_mat(self.value), None if self.tangent is None else arb_mat(self.tangent))

    def mapped(self, fn):
        """fn(x) returns (f(x), f'(x)); enclose a componentwise dual function."""
        pairs=[fn(v) for v in self.value.entries()]
        result=arb_mat(self.value.nrows(),self.value.ncols(),[v[0] for v in pairs])
        tangent=None if self.tangent is None else arb_mat(
            self.value.nrows(),self.value.ncols(),[p[1]*d for p,d in zip(pairs,self.tangent.entries())])
        return Dual(result,tangent)


def dual_zeros(rows, cols, mixed):
    return Dual(arb_mat(rows,cols), arb_mat(rows,cols) if mixed else None)


def dual_gather(value, rows, cols):
    def take(matrix):
        return arb_mat(len(rows),len(cols),[matrix[i,j] for i in rows for j in cols])
    return Dual(take(value.value), None if value.tangent is None else take(value.tangent))


def dual_scatter(value, rows, cols, nrows, ncols):
    def put(matrix):
        out=arb_mat(nrows,ncols)
        for i,r in enumerate(rows):
            for j,c in enumerate(cols):
                out[r,c]+=matrix[i,j]
        return out
    return Dual(put(value.value),None if value.tangent is None else put(value.tangent))


class Node:
    def __init__(self, dual, parents=()):
        self.dual=dual
        self.parents=tuple(parents)
        self.adjoint=None

    @property
    def shape(self):
        return self.dual.value.nrows(),self.dual.value.ncols()

    def __add__(self, other):
        return Node(self.dual+other.dual,((self,lambda g:g),(other,lambda g:g)))

    def __sub__(self, other):
        return self+other.scale(-1)

    def scale(self, coefficient):
        coefficient=coefficient if isinstance(coefficient,arb) else arb(coefficient)
        return Node(self.dual.scale(coefficient),((self,lambda g:g.scale(coefficient)),))

    def multiply(self, other):
        return Node(self.dual.multiply(other.dual),
                    ((self,lambda g:g.multiply(other.dual)),(other,lambda g:g.multiply(self.dual))))

    def matmul(self, other):
        return Node(self.dual.matmul(other.dual),
                    ((self,lambda g:g.matmul(other.dual.transpose())),
                     (other,lambda g:self.dual.transpose().matmul(g))))

    def transpose(self):
        return Node(self.dual.transpose(),((self,lambda g:g.transpose()),))

    def gather(self, rows, cols):
        rows,cols=tuple(rows),tuple(cols)
        return Node(dual_gather(self.dual,rows,cols),
                    ((self,lambda g:dual_scatter(g,rows,cols,*self.shape)),))

    def sum(self):
        v=sum(self.dual.value.entries(),arb(0))
        d=None if self.dual.tangent is None else arb_mat(1,1,[sum(self.dual.tangent.entries(),arb(0))])
        def back(g):
            count=self.shape[0]*self.shape[1]
            return Dual(arb_mat(*self.shape,[g.value[0,0]]*count),
                None if g.tangent is None else arb_mat(*self.shape,[g.tangent[0,0]]*count))
        return Node(Dual(arb_mat(1,1,[v]),d),((self,back),))

    def unary(self, fn):
        """fn returns f, f', f''; second derivative needed for HVP mode."""
        rows=[fn(v) for v in self.dual.value.entries()]
        value=arb_mat(*self.shape,[v[0] for v in rows])
        first=arb_mat(*self.shape,[v[1] for v in rows])
        second=arb_mat(*self.shape,[v[2] for v in rows])
        tangent=None if self.dual.tangent is None else hadamard(first,self.dual.tangent)
        derivative=Dual(first,None if tangent is None else hadamard(second,self.dual.tangent))
        return Node(Dual(value,tangent),((self,lambda g:g.multiply(derivative)),))

    def backward(self):
        if self.shape != (1,1):
            raise ValueError("reverse seed must be a scalar")
        # Iterative traversal avoids a Python recursion bound on deep stacks.
        seen=set();order=[];stack=[(self,False)]
        while stack:
            node,done=stack.pop()
            if done:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node));stack.append((node,True))
                stack.extend((parent,False) for parent,_ in node.parents)
        for node in order:
            node.adjoint=None
        self.adjoint=Dual(arb_mat(1,1,[arb(1)]),
                          None if self.dual.tangent is None else arb_mat(1,1))
        for node in reversed(order):
            if node.adjoint is None:
                continue
            for parent,vjp in node.parents:
                contribution=vjp(node.adjoint)
                parent.adjoint=contribution if parent.adjoint is None else parent.adjoint+contribution


def leaf(value, tangent=None):
    return Node(Dual(value,tangent))


def constant(matrix, *, mixed):
    return Node(Dual(matrix,zeros_like(matrix) if mixed else None))
