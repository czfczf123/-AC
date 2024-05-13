import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert/bert-base-chinese")

text="我是你的爸爸"
inputs =  tokenizer(text, return_tensors="pt")
print(inputs)
model = AutoModelForMaskedLM.from_pretrained("bert/bert-base-chinese")
res=model(**inputs,output_hidden_states=True).hidden_states
#res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
#print(model(**inputs,output_hidden_states=True))
res = torch.cat(res[-3:-2], -1)[0].cpu()

print(res.shape)
