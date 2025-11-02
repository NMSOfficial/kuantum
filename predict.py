import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from model import TabAttention

# modeli yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_continuous = 33

model = TabAttention(
    categories=[],
    num_continuous=num_continuous,
    dim=32,
    depth=4,
    heads=8,
    dim_head=16,
    dim_out=3,
    attn_dropout=0.1,
    ff_dropout=0.1,
    lastmlp_dropout=0.0,
    cont_embeddings='MLP',
    attentiontype='col'
).to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/NMSKuantum.pt", map_location=device))
model.eval()

# yardımcı fonksiyonlar (seninkilerle aynı)
@torch.no_grad()
def encode_cont(model, x_cont):
    outs = []
    for i in range(model.num_continuous):
        v = model.simple_MLP[i](x_cont[:, i:i+1])
        outs.append(v.unsqueeze(1))
    return torch.cat(outs, dim=1) if outs else torch.empty((x_cont.size(0), 0, model.dim), device=x_cont.device)

@torch.no_grad()
def encode_categ(model, x_categ):
    if x_categ.numel() == 0:
        return torch.empty((x_categ.size(0), 0, model.dim), device=x_categ.device)
    x = x_categ + model.categories_offset.to(x_categ.device)
    emb = model.embeds(x)
    return emb

# 🔍 tek event tahmini
@torch.no_grad()
def predict_event(model, X_row):
    x_cont = torch.tensor(X_row, dtype=torch.float32, device=device).unsqueeze(0)
    x_categ = torch.empty((1, 0), dtype=torch.long, device=device)
    x_cont_enc = encode_cont(model, x_cont)
    x_categ_enc = encode_categ(model, x_categ)
    logits = model(x_categ, x_cont, x_categ_enc, x_cont_enc)
    pred = torch.argmax(logits, dim=-1).item()
    return pred
