# GPT

A self-made LLM which can be fed datasets to get trained. So-called the baby brain AI :)

---

### First Test — Passed
Enter a sentence and the brain will predict a word. It will be gibberish and based on the vocabulary you entered, but yeah, it will reply.

### Second Test — Passed
Now enter content. The brain will learn that content in a loop that runs 100 times. Then enter the first few words of the input again, and the AI will reply with the remaining sentence.

### Before the Final Test
Added self-attention and positional embeddings, so repeated words won't cause any issues.

### Final Test — Passed
Now the model can remember what it was trained on.

### Residual Highway
Added a residual highway class with forward and backward functions to create multiple neural layers just by stacking them.

### Training Progress
The final brain is complete, and I've trained it for 10 epochs for now. Due to hardware limitations and the 2D tensor nature of the architecture, each epoch takes about 100 minutes to train. The training loss is still around 3.9.

Created a `validation.py` file to validate the brain's responses using the `validation.txt` file with `learn = false` and no optimizer, save, or backward passes.

The `chat.py` can now reply with somewhat meaningful conversation, but most replies still don't make sense.

---

### Project Paused

Stopping this project right now! Exams nearby!!

Will resume and upgrade this brain with 4D tensor processing to make it multi-head and capable of batch-wise processing.

The next plan is to upgrade the whole brain with PyTorch.

---

### Current Limitations & Flaws

- CPU-only computation; no GPU acceleration.
- Single-head self-attention architecture.
- 2D tensor-based processing.
- No batch-wise training or processing.
- Slow training, taking approximately 100 minutes per epoch.
- Limited contextual understanding.
- Inconsistent grammar and sentence generation.
- Poor conversational coherence.
- Limited training, with only 10 epochs completed.
- Model performance is constrained by the current architecture and available hardware.