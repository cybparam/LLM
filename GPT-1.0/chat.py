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

emb = fn.embeddingtable(len(voc.stoi), 128)
posemb = fn.posembeddings(500, 128)
lin = fn.linear(128, emb.vocabsize)
ln = fn.layernorm(128)
blocks = [fn.residualhighway(128) for _ in range(5)]
voc.reghook(emb.addhook)
voc.reghook(lin.addhook)
layer1 = []

for reshigh in blocks:
    layer1.append(reshigh.atten.wq)
    layer1.append(reshigh.atten.wk)
    layer1.append(reshigh.atten.wv)
    layer1.append(reshigh.linup)
    layer1.append(reshigh.lindown)
    layer1.append(reshigh.ln1)
    layer1.append(reshigh.ln2)

layer2 = [emb, posemb, ln, lin]
layers = layer2 + layer1


try:
    for i, layer in enumerate(layers):
        data = np.load(f"Training_data/current/layer_{i}.npz")
        if hasattr(layer, "weight"):
            layer.weight = data["weight"]
        if hasattr(layer, "bias"):
            layer.bias = data["bias"]
        if hasattr(layer, "beta"):
            layer.gamma = data["gamma"]
            layer.beta = data["beta"]
except FileNotFoundError:
    print("Training Data not found!")
    exit()

def forward(tokens):
    vectors = posemb.forward(emb.forward(tokens[:-1]))
    for b in blocks:
        vectors = b.forward(vectors)
    return lin.forward(ln.forward(vectors))

print("XX Type //exit to exit chat XX")
while True:
    user = input("User: ").lower()
    
    if "//exit" in user:
        break

    tokens = voc.encode(tok.tokenize(user))
    tokens.insert(-1, 4)
    ai = []
    print("AI: ", end=" ")
    while True:
        prob = fn.softmax(forward(tokens), 0.5)
        ait = np.random.choice(len(prob[-1]), p = prob[-1])
        if ait == 0 or ait == 4:
            break
        print(tok.detokenize(voc.decode([ait])), end = " ", flush=True)
        ai.append(ait)
        tokens.insert(-1, ait)
        if len(tokens) > 500:
            break
    print()