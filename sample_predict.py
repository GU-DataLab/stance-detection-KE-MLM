from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# select mode path here
# see more at https://huggingface.co/kornosk
# pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden"
# pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-trump"
pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"
# pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)

id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}

##### Prediction Neutral #####
sentence = "Hello World."
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

##### Prediction Favor #####
sentence = "Go Go Biden!!!"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

##### Prediction Against #####
sentence = "Biden is the worst."
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

# please consider citing our paper if you feel this is useful :)