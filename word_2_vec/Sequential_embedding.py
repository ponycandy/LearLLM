import torch
from SlidingwindowSampler import create_dataloader_v1


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader) #A
    first_batch = next(data_iter)
    print(first_batch)


output_dim = 256

vocab_size = 50257

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=128)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)

print("\nInputs shape:\n", inputs.shape)

