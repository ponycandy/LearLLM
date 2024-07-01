from  SimpleTokenizerV1 import  SimpleTokenizerV1


import re
import requests

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
response = requests.get(url)
raw_text = response.text
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if
                item.strip()]

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)


vocab = {token:integer for integer,token in
         enumerate(all_words)}



tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn
said with pardonable pride."""
ids = tokenizer.encode(text)

tokenizer.decode(ids)

text = "Hello, do you like tea?"
# tokenizer.encode(text)
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
#添加未知和终止token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}


for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


