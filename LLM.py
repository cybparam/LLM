import Functions as fn
import numpy as np
import time

tok, voc = fn.tokenizer(), fn.vocab()
emb = fn.embeddingtable(len(voc.stoi), 8)
lin = fn.linear(8, 16)
ln = fn.layernorm(8)
voc.reghook(emb.addhook)

def runai(user):
    tokens = tok.tokenize(user)
    embeddings = emb.forward(voc.encode(tokens))
    lin1 = fn.gelu(lin.forward(ln.out(embeddings)))
    lin2 = fn.linear(16, emb.vocabsize)
    softmax = fn.softmax(lin2.forward(lin1))
    ai = np.random.choice(len(softmax[-1]), p = softmax[-1])
    return voc.decode([ai])

user = input("SAY HI!: ").lower()
for i in range(50):
    ai = runai(user)
    user += " "
    user += tok.detokenize(ai)
    if 0 in ai:
        break
    print(tok.detokenize(ai), end = " ")