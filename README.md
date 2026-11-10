# LLM
A Self Made LLM Which can Be Fed Data sets to get trained. So called the baby brain ai :)

#############
Passed the first test now enter a sentence and the brain will predeict a word. it will be gibberish and based in the vocab you entered but yeah it will reply you

#############
Passed the second test.
Now enter content the brain will learn that content in a loop which runs 100 times ad then enter first few words of the inputagain and ai will reply with remaining complete sentence

--Before the fial test run. Added self attention and positioanl embeddings so now the repeated words wont cause any issue

#############
Passed The final test run
Now the model can remember what was it trained on.

###############
added residual highway class with forward and backward functioon to create multiple neural layers by just stacking them

###############
So the final brain is complete and i've trained this on 10 epoch cycle for now. due to hardware limitation and 2D tensor nature of the architecture. Each epoch cycle take about 100 min to train still the training loss is of 3.9 till now. Created a validation.py file to validate the brain'sresponses using the validation.txt file with learn = false and no optimizer/save/backward passes. 

{Attaching All the code files with training data of 10th epoch with training progress to resume training}

Also If you find any bug, related to why the training is so slow like even on 10th epoch the loss is near 4 pls lmk.

The chat.py can now reply and do a soewhere meaningful conversation but mostly the replies dont make sense.

Pls lmk the bug in any files and any limitations in this code and training/chat or brain's overall architecture.

*************************
Stopping this project right now! Exams nearby!!

Will resume and upgrade this brain with capabilities of 4D tensors processing TO make this multihead and batch wise processing.this will eventually increase the training speed optimising the absolute limit of the hardware. 
The will also result in brain to learn more features like grammer, past context, and userinput's context
*************************

The next Plan is to upgradethe whole brain with pytorch......