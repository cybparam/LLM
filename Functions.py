import math
import numpy as np

learn = True

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

class tokenizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.punct = [",", ".", "!", "?", ";", ":"]

    def tokenize(self, user):
        if self.lowercase:
            user = user.lower()

        tokens = ["<BOS>"]
        cword = []

        for char in user:
            if char.isspace():
                if cword:
                    tokens.append("".join(cword))
                    cword = []
            elif char in self.punct:
                if cword:
                    tokens.append("".join(cword))
                    cword = []
                tokens.append(char)
            else:
                cword.append(char)
        if cword:
            tokens.append("".join(cword))

        tokens.append("<EOS>")
        return tokens

    def detokenize(self, ait):
        ignore = ["<EOS>", "<BOS>"]
        ai = []
        for token in ait:
            if token in ignore:
                continue
            if ai and token not in self.punct:
                ai.append(" ")
            ai.append(token)
        return "".join(ai)

class vocab:

    def __init__(self):
        self.hook = []
        self.stoi = {"<EOS>": 0, "<BOS>": 1, "<UKN>": 2, "<PAD>":3}
        self.itos = {0: "<EOS>", 1:"<BOS>", 2: "<UKN>", 3: "<PAD>"}

    def encode(self, tokens):
        output = []
        context = []
        for _ in tokens:
            if _ in self.stoi and _ not in ("<PAD>", "<UKN>", "<EOS>", "<BOS>"):
                context.append(self.stoi[_])
        for token in tokens:
            if token in self.stoi:
                output.append(self.stoi[token])
            else:
                if learn:
                    l = len(self.stoi)
                    self.stoi[token] = l
                    self.itos[l] = token
                    output.append(self.stoi[token])
                    for fn in self.hook:
                        fn(l, token, context)
                    context.append(l)
                else:
                    output.append(self.stoi["<UKN>"])
                    
        return output

    def decode(self, output):
        ait= []
        for token in output:
            ait.append(self.itos.get(token, "<UKN>"))
        return ait

    def reghook(self, fn):
        self.hook.append(fn)

class embeddingtable:

    def __init__(self, vocabsize, embeddim):
        self.vocabsize = vocabsize
        self.embeddim = embeddim
        self.weight = np.random.randn(self.vocabsize, self.embeddim) * 0.02

    def forward(self, tokenid):
        self.previd = tokenid
        return self.weight[tokenid]

    def addhook(self, nid, token, context):
        if context:
            nrow = np.mean(self.weight[context], axis=0)
            noise = np.random.randn(1, self.embeddim) * 0.005
            nrow = nrow + noise
        else:
            nrow = np.random.randn(1, self.embeddim) * 0.02
        self.weight = np.vstack([self.weight, nrow])
        self.vocabsize += 1

class linear:

    def __init__(self, indim, outdim):
        self.indim = indim
        self.outdim = outdim
        self.weight = np.random.randn(indim, outdim) * 0.02
        self.bias = np.zeros((1, outdim))

    def forward(self, x):
        self.prevx = x
        return (x @ self.weight) + self.bias

def softmax(logits, temp=1):
    logits = logits / temp
    t = np.exp(logits - np.max(logits, axis =  -1, keepdims = True))
    return t / np.sum(t, axis = -1, keepdims = True)

class layernorm:

    def __init__(self, dim):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.esp = 0.00001

    def out(self, x):
        var = np.var(x, axis = -1, keepdims = True) + self.esp
        m = np.mean(x, axis = -1, keepdims = True)
        x = (x - m) / np.sqrt(var)
        return (self.gamma * x) + self.beta


def gelu(x):
    curve = np.tanh(math.sqrt(2/np.pi) * (x + (0.044715 * (x**3))))
    return 0.5 * x * (1 + curve)
