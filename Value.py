import math

class value:

    def __init__(self, data, children=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.prev = set(children)
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        if not isinstance(other, value):
            other = value(other)
        out = value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, value):
            other = value(other)
        out = value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = value(self.data ** other, (self,), f"**{other}")
    
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self *-1

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
            out = value(math.exp(self.data), (self,), "exp")

            def _backward():
                self.grad += out.data * out.grad
            out._backward = _backward

            return out

    def tanh(self):
            t = math.tanh(self.data)
            out = value(t, (self,), "tanh")

            def _backward():
                self.grad += (1.0 - t ** 2) * out.grad
            out._backward = _backward

            return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()