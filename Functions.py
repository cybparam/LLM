import math
import numpy as np

learn = False

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

    def backward(self, dy):
        self.dweight = np.zeros_like(self.weight)
        return np.add.at(self.dweight, self.previd, dy)

    def addhook(self, nid, token, context):
        if context:
            nrow = np.mean(self.weight[context], axis=0)
            noise = np.random.randn(1, self.embeddim) * 0.005
            nrow = nrow + noise
        else:
            nrow = np.random.randn(1, self.embeddim) * 0.02
        self.weight = np.vstack([self.weight, nrow])
        self.vocabsize += 1

    def save(self, path):
        np.savez(path, weight=self.weight)

class posembeddings:

    def __init__(self, len, dim):
        self.weight = np.random.randn(len, dim) * 0.02

    def forward(self, tokens):
        self.t = tokens.shape[0]
        return tokens + self.weight[:self.t]    

    def backward(self, dy):
        self.dweight = np.zeros_like(self.weight)
        self.dweight[:self.t] = dy
        return dy

    def save(self, path):
        np.savez(path, weight=self.weight)

class linear:

    def __init__(self, indim, outdim):
        self.indim = indim
        self.outdim = outdim
        self.weight = np.random.randn(indim, outdim) * 0.02
        self.bias = np.ones((1, outdim)) * 0.005

    def forward(self, x):
        self.prevx = x
        return (x @ self.weight) + self.bias

    def backward(self, dy):
        self.dweight = self.prevx.T @ dy
        self.dbias = np.sum(dy, axis = 0, keepdims = True)
        dx = dy @ self.weight.T
        return dx

    def addhook(self, nid, token, context):
        ncol = np.random.randn(self.indim, 1) * 0.02
        self.weight = np.hstack([self.weight, ncol])
        self.bias = np.hstack([self.bias, np.zeros((1, 1))])
        self.outdim += 1

    def save(self, path):
        np.savez(path, weight=self.weight, bias=self.bias)
        
def softmax(logits, temp=1):
    logits = logits / temp
    t = np.exp(logits - np.max(logits, axis =  -1, keepdims = True))
    return t / np.sum(t, axis = -1, keepdims = True)

class layernorm:

    def __init__(self, dim):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.dim = dim
        self.esp = 0.00001

    def forward(self, x):
        var = np.var(x, axis = -1, keepdims = True) + self.esp
        m = np.mean(x, axis = -1, keepdims = True)
        x = (x - m) / np.sqrt(var)
        self.x = x
        self.var = np.sqrt(var)
        return (self.gamma * x) + self.beta

    def backward(self, dy):
        self.dbeta = np.sum(dy, axis = 0, keepdims = True)
        self.dgamma = np.sum(dy*self.x, axis = 0, keepdims = True)
        dx = dy * self.gamma
        t1 = 1 / (self.dim * self.var)
        t5 = np.sum(dx, axis = -1, keepdims = True)
        t4 = (self.x * (np.sum(dx * self.x, axis = -1, keepdims = True)))
        t2 = t5 + t4
        t3 = self.dim * dx
        return t1 * (t3 - t2)

    def save(self, path):
        np.savez(path, gamma=self.gamma, beta=self.beta)
    
class GELU:

    def forward(self, x):
        self.prevx = x
        curve = np.tanh(np.sqrt(2/np.pi) * (x + (0.044715 * (x**3))))
        return 0.5 * x * (1 + curve)

    def backward(self, dy):
        u = np.sqrt(2/np.pi) * (self.prevx + (0.044715 * (self.prevx**3)))
        du = np.sqrt(2/np.pi) * (1 + (3 * (0.044715 * (self.prevx**2))))
        c = np.tanh(u)
        grad = (0.5 * (1 + c)) + (0.5 * self.prevx * (1 - c**2) * du)
        return dy*grad


def crossentropy(probs, targets):
    prob = probs[np.arange(len(targets)), targets]
    loss =  np.mean(-np.log(prob + 10**-9))
    probc = probs.copy()
    probc[np.arange(len(targets)), targets] -= 1.0
    dlogits =  probc / len(targets)
    return loss, dlogits

class adamw:

    def __init__(self, layers, lr):
        self.t = 0
        self.lr = lr
        self.layers = layers

    def updatepara(self, m, v, w, dw):
        if np.shape(m) != w.shape:
            m, v = 0, 0
        b1, b2, e, l = 0.9, 0.999, 10**-8, 0.01
        m = b1 * m + (1 - b1)*dw
        v = b2 * v + (1 - b2)*(dw**2)
        mh = m / (1 - (b1**self.t))
        vh = v / (1 - (b2**self.t))
        w = w - self.lr * ((mh/(np.sqrt(vh) + e)) + l* w)
        return w, m, v

    def step(self):
        self.t += 1
        for layer in self.layers:
            if not hasattr(layer, "mw"):
                layer.mw, layer.vw = 0, 0
                layer.mb, layer.vb = 0, 0
            if hasattr(layer, "weight"):
                layer.weight, layer.mw, layer.vw = self.updatepara(layer.mw, layer.vw, layer.weight, layer.dweight)
            if hasattr(layer, "bias"):
                layer.bias, layer.mb, layer.vb = self.updatepara(layer.mb, layer.vb, layer.bias, layer.dbias)
            if hasattr(layer, "beta"):
                layer.beta, layer.mb, layer.vb = self.updatepara(layer.mb, layer.vb, layer.beta, layer.dbeta)
            if hasattr(layer, "gamma"):
                layer.gamma, layer.mw, layer.vw = self.updatepara(layer.mw, layer.vw, layer.gamma, layer.dgamma)

class selfattention:

    def __init__(self, dim):
        self.wq, self.wk, self.wv = linear(dim, dim), linear(dim, dim), linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        self.q, self.k, self.v, = self.wq.forward(x), self.wk.forward(x), self.wv.forward(x)
        scores = (self.q @ self.k.T) /math.sqrt(self.dim)
        mask = np.triu(np.ones(scores.shape) * -1e9, k=1)
        self.probs = softmax(scores + mask)
        return self.probs @ self.v
    
    def backward(self, dy):
        dprobs = dy @ self.v.T
        dv = self.probs.T @ dy
        dscores = (self.probs * (dprobs - np.sum(self.probs * dprobs, axis = 1, keepdims = True))) / math.sqrt(self.dim)
        dq = dscores @ self.k
        dk = dscores.T @ self.q
        dxq, dxk, dxv = self.wq.backward(dq), self.wk.backward(dk), self.wv.backward(dv) 
        return dxq + dxk + dxv

class residualhighway:

    def __init__(self, dim):
        self.ln1 = layernorm(dim)
        self.ln2 = layernorm(dim)
        self.linup = linear(dim, dim * 4)
        self.lindown = linear(dim * 4, dim)
        self.atten = selfattention(dim)
        self.gelu = GELU()

    def ffn(self, h):
        return self.lindown.forward(self.gelu.forward(self.linup.forward(h)))

    def forward(self, tokens):
        h = tokens + self.atten.forward(self.ln1.forward(tokens))
        return h + self.ffn(self.ln2.forward(h))

    def backward(self, dy):
        dy = self.ln2.backward(self.linup.backward(self.gelu.backward(self.lindown.backward(dy)))) + dy
        dy= self.ln1.backward(self.atten.backward(dy)) + dy
        return dy