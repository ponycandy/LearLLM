import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]) #A
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) #B
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
class DummyTransformerBlock(nn.Module): #C
    def __init__(self, cfg):
    	super().__init__()
    def forward(self, x): #D
    	return x
class DummyLayerNorm(nn.Module): #E
    def __init__(self, normalized_shape, eps=1e-5): #F
    	super().__init__()
    def forward(self, x):
    	return x
    
class LayerNorm(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.eps = 1e-5
		self.scale = nn.Parameter(torch.ones(emb_dim))
		self.shift = nn.Parameter(torch.zeros(emb_dim))
	def forward(self, x):
		mean = x.mean(dim=-1, keepdim=True)
		var = x.var(dim=-1, keepdim=True, unbiased=False)
		norm_x = (x - mean) / torch.sqrt(var + self.eps)
		return self.scale * norm_x + self.shift