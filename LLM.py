import Functions as fn
import numpy as np

user = input("User: ").lower()

tok = fn.tokenizer()
voc = fn.vocab()
emb = fn.embeddingtable(len(voc.stoi), 8)
lin1 = fn.linear(8, 16)
lin2 = fn.linear(16, emb.vocabsize)
ln = fn.layernorm(8)
gelu = fn.GELU()
voc.reghook(emb.addhook)
voc.reghook(lin2.addhook)
optimizer = fn.adamw([emb, ln, lin1, lin2], 0.05)

tokens = voc.encode(tok.tokenize(user))

for step in range(100):
    prob = fn.softmax(lin2.forward(gelu.forward(lin1.forward(ln.forward(emb.forward(tokens[:-1]))))))
    loss, dy = fn.crossentropy(prob, tokens[1:])
    dx = emb.backward(ln.backward(lin1.backward(gelu.backward(lin2.backward(dy)))))
    optimizer.step()

def aif(prob):
    ai = tok.detokenize(voc.decode([np.random.choice(len(prob[-1]), p = prob[-1])]))
    return ai

user = input("User: ").lower()
tokens = voc.encode(tok.tokenize(user))
ai = []

while True:
    prob = fn.softmax(lin2.forward(gelu.forward(lin1.forward(ln.forward(emb.forward(tokens[:-1]))))))
    ait = np.random.choice(len(prob[-1]), p = prob[-1])

    if ait == 0:
        break

    ai.append(ait)
    tokens.insert(-1, ait)

print(tok.detokenize(voc.decode(ai)))