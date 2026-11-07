import Functions as fn
import numpy as np
import random
import json

fn.learn = True

tok = fn.tokenizer()
voc = fn.vocab()

try:
    with open("Training_data/vocabulary.json", "r") as file:
        vocabulary = json.load(file)
        voc.stoi = {k: int(v) for k, v in vocabulary.items()}
        voc.itos = {int(v): k for k, v in vocabulary.items()}
except FileNotFoundError:
    print("Vocabulary did not found!!, initialising base Vocabulary...")

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

try:
    with open("Training_data/token_memory.json", "r") as file:
        memory = json.load(file)
except FileNotFoundError:
    print("No Token History Found!")
    memory = []

def forward(tokens):
    return lin.forward(ln.forward(reshigh.forward(posemb.forward(emb.forward(tokens[:-1])))))

def backward(dy):
    return emb.backward(posemb.backward(reshigh.backward(ln.backward(lin.backward(dy)))))

print("XX Type //exit to exit chat XX")
while True:
    user = input("User: ").lower()

    if "//exit" in user:
        break
    else:
        token_list = [voc.encode(tok.tokenize(user))]
        memory.append(token_list[0])

        for i in range(random.randint(1, 3)):
            token_list.append(random.choice(memory))

        for tokens in token_list:
            for step in range(500):
                prob = fn.softmax(forward(tokens))
                loss, dy = fn.crossentropy(prob, tokens[1:])
                dx = backward(dy)
                optimizer.step()

with open("Training_data/vocabulary.json", "w") as file:
    json.dump(voc.stoi, file)

with open("Training_data/token_memory.json", "w") as file:
    json.dump(memory, file)

for i, layer in enumerate(layers):
    layer.save(f"Training_data/layer_{i}.npz")
