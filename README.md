# Stance Detection

This repository is for the paper - [Knowledge Enhance Masked Language Model for Stance Detection](https://www.aclweb.org/anthology/2021.naacl-main.376), NAACL 2021. 🚀

Code for log-odds-ratio with Dirichlet prior is at [log-odds-ratio](https://github.com/kornosk/log-odds-ratio) repository.

## Data Sets

This data sets are for research purposes only - [Download](https://portals.mdi.georgetown.edu/public/stance-detection-KE-MLM) 🔥

- Data format is CSV with only 3 columns: `"tweet_id","text","label"`
- Labels = `{0:"AGAINST", 1:"FAVOR", 2:"NONE"}`

The data set contains 2500 manually-stance-labeled tweets, 1250 for each candidate (Joe Biden and Donald Trump). These tweets were sampled from the unlabeled set that our research team collected English tweets related to the 2020 US Presidential election. Through the Twitter Streaming API, we collected data using election-related hashtags and keywords. Between January 2020 and September 2020, we collected over 5 million tweets, not including quotes and retweets. These unlabeled tweets were used to fine-tune all of our language models. The labeled data that we publicly provide were sampled from this 5M set and were labeled using Amazon Mechanical Turk.

The stance label distributions are shown in the table below. Please refer to [our paper](https://www.aclweb.org/anthology/2021.naacl-main.376.pdf) for more detail about the data sets.

|       | %SUPPORT | %OPPOSE | %NEUTRAL |
| ----- | :----: | :----: | :----: |
| Biden | 31.3 | 39.0 | 29.8 |
| Trump | 27.3 | 39.9 | 32.8 |

## Result
On each pre-trained language model, we trained for the downstream stance detection task for five times and report average scores in Table 2.

![image](https://user-images.githubusercontent.com/15230011/114804906-176f4c00-9d70-11eb-9122-b35c7803fd68.png)

## Pre-trained Models

All models are uploaded to my [Huggingface](https://huggingface.co/kornosk) 🤗 so you can load model with **just three lines of code**!!!

- [BERT-Political-Election-2020-Twitter-MLM](https://huggingface.co/kornosk/bert-political-election2020-twitter-mlm) - Feel free to fine-tune this to any downstream task 🎯
- [BERT-Election-2020-Twitter-Stance-Biden-f-BERT](https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden)
- [BERT-Election-2020-Twitter-Stance-Biden-KE-MLM](https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden-KE-MLM)
- [BERT-Election-2020-Twitter-Stance-Trump-f-BERT](https://huggingface.co/kornosk/bert-election2020-twitter-stance-trump)
- [BERT-Election-2020-Twitter-Stance-Trump-KE-MLM](https://huggingface.co/kornosk/bert-election2020-twitter-stance-trump-KE-MLM)

## Usage

We tested in `pytorch v1.8.1` and `transformers v4.5.1`.

Please see specific model pages above for more usage detail. Below is a sample use case. 

### 1. Choose and load model for stance detection

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# select mode path here
# see more at https://huggingface.co/kornosk
pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
```

### 2. Get a prediction (see more in `sample_predict.py`)
```python
id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}

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

# please consider citing our paper if you feel this is useful :)
```

## Citation
If you feel our paper and resources are useful, please consider citing our work! 🙏
```bibtex
@inproceedings{kawintiranon2021knowledge,
    title={Knowledge Enhanced Masked Language Model for Stance Detection},
    author={Kawintiranon, Kornraphop and Singh, Lisa},
    booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year={2021},
    publisher={Association for Computational Linguistics},
    url={https://www.aclweb.org/anthology/2021.naacl-main.376}
}
```

## Troubleshoot

### 1. Can't load the model
- From [this issue](https://github.com/GU-DataLab/stance-detection-KE-MLM/issues/1)
- Check the dependencies `pytorch==1.8.1` and `transformers==4.5.1`
- Try removing `tensorflow`
