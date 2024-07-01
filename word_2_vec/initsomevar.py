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