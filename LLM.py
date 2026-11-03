import Functions as fn
import numpy as np

tok, voc = fn.tokenizer(), fn.vocab()
emb = fn.embeddingtable(len(voc.stoi), 8)
lin = fn.linear(8, 16)
ln = fn.layernorm(8)
voc.reghook(emb.addhook)

def runai(user):
    tokens = tok.tokenize(user)
    output = voc.encode(tokens)
    vectors = emb.forward(output)
    out = ln.out(vectors)
    print(out.shape, np.mean(out[0]), np.var(out[0]))
    print(fn.gelu(np.array([[-3.0, 0.0, 3.0]])))
    prob = fn.softmax(lin.forward(vectors))
    print(prob.shape, np.sum(prob[0]))
    ait = voc.decode(output)
    ai = tok.detokenize(ait)
    return ai

user = input("SAY HI!: ").lower()
ai = runai(user)
print(ai)