import math

class value:

    def __init__(self, data, children=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.prev = set(children)
        self.op = op

    def __repr__(self):
        return f"value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        if not isinstance(other, value):
            other = value(other)
        out = value(self.data + other.data, (self, other), "+")
        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, value):
            other = value(other)
        out = value(self.data * other.data, (self, other), "*")
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        return value(self.data ** other, (self,), f"**{other}")

    def __neg__(self):
        return self *-1

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
      return value(math.exp(self.data), (self,), "exp")

    def tanh(self):
        t = math.tanh(self.data)
        return value(t, (self,), "tanh")

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self ** -1)
    