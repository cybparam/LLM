import math

learn = True

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
        out = value(self.data + (-other.data), (self, other), "-")
        return out

    def __mul__(self, other):
        if not isinstance(other, value):
            other = value(other)
        out = value(self.data * other.data, (self, other), "*")
        return out

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
        self.itos = {0: "<EOS>", 1:"<BOS>"}

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

    def register_hook(self, fn):
        self.hook.append(fn)

tok, voc  = tokenizer(), vocab()
def runai(user):
    tokens = tok.tokenize(user)
    output = voc.encode(tokens)
    ait = voc.decode(output)
    ai = tok.detokenize(ait)
    return ai

user = input("SAY HI!: ").lower()
ai = runai(user)
print(ai)