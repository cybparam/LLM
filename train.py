import Functions as fn
import numpy as np
import random
import json
import time
import sys

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

try:
    data_set = []
    with open("Training/train.txt", "r", encoding="utf-8") as dataset:
        for line in dataset:
            turns = [t.strip() for t in line.split("__eou__") if t.strip()]
            conversation = " <SEP> ".join(turns)
            tokens = voc.encode(tok.tokenize(conversation))
            for i in range(0, len(tokens) - 1, 499):
                chunk = tokens[i:i + 500]
                if len(chunk) >= 3:
                    data_set.append(chunk)
except FileNotFoundError:
    print("DataSet Not Found")
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
optimizer = fn.adamw(layers, 0.005)

try:
    for i, layer in enumerate(layers):
        data = np.load(f"Training_data/layer_{i}.npz")
        if hasattr(layer, "weight"):
            layer.weight = data["weight"]
        if hasattr(layer, "bias"):
            layer.bias = data["bias"]
        if hasattr(layer, "gamma"):
            layer.gamma = data["gamma"]
        if hasattr(layer, "beta"):
            layer.beta = data["beta"]
except FileNotFoundError:
    print("Training Data not found! initialising data...")

try:
    with open("Training_data/step.txt", "r") as file:
        steps = file.read()
        optimizer.t = int(steps)
except FileNotFoundError:
    print("Starting with step 0")
try:
    for i, layer in enumerate(layers):
        data = np.load(f"Training_data/Training_Progress/layer_{i}.npz")
        if hasattr(layer, "weight"):
            layer.mw = data["mw"]
            layer.vw = data["vw"]
        if hasattr(layer, "bias"):
            layer.mb = data["mb"]
            layer.vb = data["vb"]
        if hasattr(layer, "gamma"):
            layer.mw = data["mw"]
            layer.vw = data["vw"]
        if hasattr(layer, "beta"):
            layer.mb = data["mb"]
            layer.vb = data["vb"]
except FileNotFoundError:
    print("Training Progress not found! Starting All Over...")

def forward(tokens):
    vectors = posemb.forward(emb.forward(tokens[:-1]))
    for b in blocks:
        vectors = b.forward(vectors)
    return lin.forward(ln.forward(vectors))

def backward(dy):
    grad = ln.backward(lin.backward(dy))
    for b in reversed(blocks):
        grad = b.backward(grad)
    return emb.backward(posemb.backward(grad))

def save(layers):
    with open("Training_data/vocabulary.json", "w") as file:
        json.dump(voc.stoi, file)

    for i, layer in enumerate(layers):
        layer.save(f"Training_data/layer_{i}.npz")

    with open("Training_data/step.txt", "w") as file:
        file.write(str(optimizer.t))

    optimizer.save(f"Training_data/Training_Progress/")

    print("Saved CheckPoint")

def progress_bar(current, total, loss, speed, start):
    width = 30
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "-" * (width - filled)
    spinner = "|/-\\"
    icon = spinner[current % len(spinner)]
    elapsed = time.time() - start
    eta = (total - current) / speed if speed > 0 else 0
    sys.stdout.write(f"\r{icon} [{bar}] {percent*100:5.1f}% "f"| {current}/{total} " f"| Loss: {loss:.5f} "f"| ETA: {eta/60:.1f} min")
    sys.stdout.flush()

print("Starting Training....")
for epoch in range(50):
    random.shuffle(data_set)
    totalloss = 0.0
    start = time.time()
    progress_bar(0, len(data_set), 0.0, 0.0, start)
    for i, tokens in enumerate(data_set):
        prob = fn.softmax(forward(tokens))
        loss, dy = fn.crossentropy(prob, tokens[1:])
        dx = backward(dy)
        optimizer.step()
        totalloss += loss
        if (i + 1) % 1 == 0 or i == len(data_set) - 1:
            elapsed = time.time() - start
            speed = (i + 1) / elapsed
            progress_bar(i + 1, len(data_set), totalloss / (i + 1), speed, start)
    totalloss = totalloss / len(data_set)
    print()
    print(f"Epoch: {epoch} | Loss: {totalloss}")
    save(layers)

print("Training Complete. Saving Data...")
save(layers)
print("All Set!")