import Functions as fn
import numpy as np
import json

tok = fn.tokenizer()
voc = fn.vocab()

try:
    with open("Training_data/vocabulary.json", "r") as file:
        vocabulary = json.load(file)
        voc.stoi = {k: int(v) for k, v in vocabulary.items()}
        voc.itos = {int(v): k for k, v in vocabulary.items()}
except FileNotFoundError:
    print("No Training Data Found. Cannot run the model")
    exit()

emb = fn.embeddingtable(len(voc.stoi), 8)
posemb = fn.posembeddings(1024, 8)
lin = fn.linear(8, emb.vocabsize)
ln = fn.layernorm(8)
reshigh = fn.residualhighway(8)
voc.reghook(emb.addhook)
voc.reghook(lin.addhook)
layers = [emb, posemb, ln, reshigh.atten.wq, reshigh.atten.wk, reshigh.atten.wv, reshigh.linup, reshigh.lindown, reshigh.ln1, reshigh.ln2, lin]
optimizer = fn.adamw(layers, 0.05)

try:
    for i, layer in enumerate(layers):
        data = np.load(f"Training_data/layer_{i}.npz")
        if hasattr(layer, "weight"):
            layer.weight = data["weight"]
        if hasattr(layer, "bias"):
            layer.bias = data["bias"]
        if hasattr(layer, "beta"):
            layer.gamma = data["gamma"]
            layer.beta = data["beta"]
except FileNotFoundError:
    print("Training Data not found! initialising data...")
    exit()

def forward(tokens):
    return lin.forward(ln.forward(reshigh.forward(posemb.forward(emb.forward(tokens[:-1])))))

print("XX Type //exit to exit chat XX")
while True:
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