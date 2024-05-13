import random

from TTS.tts.datasets.formatters import kespeech1

item=kespeech1("../datasets/KeSpeech")
print(len(item))
print(item[:4])

M=[]
NM=[]
for i in item:
    if i["dialect"] == "Mandarin":
        M.append(i)
    else:
        NM.append(i)
M=random.sample(M,len(M)//6)


print(len(M))
print(len(NM))
