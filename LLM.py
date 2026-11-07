import Functions as fn
import numpy as np

user = input("User: ").lower()

tok = fn.tokenizer()
voc = fn.vocab()
emb = fn.embeddingtable(len(voc.stoi), 8)
posemb = fn.posembeddings(1024, 8)
lin1 = fn.linear(8, 16)
lin2 = fn.linear(16, emb.vocabsize)
ln = fn.layernorm(8)
gelu = fn.GELU()
atten = fn.selfattention(8)
reshigh = fn.residualhighway(8)
voc.reghook(emb.addhook)
voc.reghook(lin2.addhook)
optimizer = fn.adamw([emb, posemb, ln, atten.wq, atten.wk, atten.wv, lin1, lin2], 0.05)


def forward(tokens):
    return lin2.forward(gelu.forward(lin1.forward(atten.forward(ln.forward(posemb.forward(emb.forward(tokens[:-1])))))))

def backward(dy):
    return emb.backward(posemb.backward(ln.backward(atten.backward(lin1.backward(gelu.backward(lin2.backward(dy)))))))

tokens = voc.encode(tok.tokenize(user))

for step in range(1000):
    prob = fn.softmax(forward(tokens))
    loss, dy = fn.crossentropy(prob, tokens[1:])
    dx = backward(dy)
    optimizer.step()

# Original ai output code
def aif(prob):
    ai = tok.detokenize(voc.decode([np.random.choice(len(prob[-1]), p = prob[-1])]))
    return ai
#------------------------

user = input("User: ").lower()
tokens = voc.encode(tok.tokenize(user))
ai = []

while True:
    prob = fn.softmax(forward(tokens))
    ait = np.random.choice(len(prob[-1]), p = prob[-1])

    if ait == 0:
        break

    ai.append(ait)
    tokens.insert(-1, ait)

print(tok.detokenize(voc.decode(ai)))