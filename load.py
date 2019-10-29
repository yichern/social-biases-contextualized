import pytorch_pretrained_bert as bert

version = "gpt2-345"
tokenizer = bert.GPT2Tokenizer.from_pretrained(version)
model = bert.GPT2Model.from_pretrained(version)
