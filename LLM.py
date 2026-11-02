from Functions import value, tokenizer, vocab, embeddingtable

tok, voc = tokenizer(), vocab()
emb = embeddingtable(len(voc.stoi), 8)
voc.reghook(emb.addhook)
def runai(user):
    tokens = tok.tokenize(user)
    output = voc.encode(tokens)
    vectors = emb.forward(output)
    print(emb.weight.shape, vectors.shape)
    ait = voc.decode(output)
    ai = tok.detokenize(ait)
    return ai

user = input("SAY HI!: ").lower()
ai = runai(user)
print(ai)