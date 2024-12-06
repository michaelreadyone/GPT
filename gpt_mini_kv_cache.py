import torch
import torch.nn as nn
from torch.nn import functional as F
from icecream import ic

# hyperparameters
batch_size = 2 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 32
n_head = 2
n_layer = 2
dropout = 0.2

training = True

ic(f'training: {training}, n_head: {n_head}, n_layer: {n_layer}')

# torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str=device, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str=device):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class Head(nn.Module):
    """ One head of self-attention with ROPE """

    def __init__(self, head_size, head_idx):
        super().__init__()
        self.head_idx = head_idx
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.cache_k = None
        self.cache_v = None
        self.warm_up = True
        self.infer_len = 0

    def forward(self, x):
        B, T, C = x.shape
        if training == True:
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            
            freqs_complex = precompute_theta_pos_frequencies(self.head_size, T)
            
            k = apply_rotary_embeddings(k, freqs_complex)
            q = apply_rotary_embeddings(q, freqs_complex)
            

            wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
            wei = wei.masked_fill(torch.tril(torch.ones(T, T, device=device)) == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        else:
            if self.warm_up:
                k = self.key(x)
                q = self.query(x)
                v = self.value(x)
                self.infer_len = T
                
                freqs_complex = precompute_theta_pos_frequencies(self.head_size, T)
                
                k = apply_rotary_embeddings(k, freqs_complex)
                q = apply_rotary_embeddings(q, freqs_complex)
                

                wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
                wei = wei.masked_fill(torch.tril(torch.ones(T, T, device=device)) == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
                
                self.cache_k = k
                self.cache_v = v
                self.warm_up = False
            else:
                x = x[:,-1,:]
                k = self.key(x)
                q = self.query(x)
                v = self.value(x)
                self.infer_len = self.infer_len + 1
                
                freqs_complex = precompute_theta_pos_frequencies(self.head_size, self.infer_len)[-1].unsqueeze(0)
                
                k = apply_rotary_embeddings(k, freqs_complex)
                q = apply_rotary_embeddings(q, freqs_complex)
                

                cache_k = torch.cat((self.cache_k, k.unsqueeze(0)), dim=1)
                cache_k = cache_k[:,-block_size:,:]
                cache_v = torch.cat((self.cache_v, v.unsqueeze(0)), dim=1)
                cache_v = cache_v[:,-block_size:,:]
                self.cache_k = cache_k
                self.cache_v = cache_v

                att = q @ cache_k.transpose(-2,-1) * cache_k.shape[-1]**-0.5 # (B, 1, hs) @ (B, hs, T) -> (B, 1, T)
                att = F.softmax(att, dim=-1) # (B, 1, T)
                att = self.dropout(att)
                
                # logger.info(f'cache_k: {cache_k}')
                # logger.info(f'cache_v: {cache_v}')
                out = att @ cache_v # (B, 1, T) @ (B, T, hs) -> (B, 1, hs)
                      
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, i) for i in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel().to(device)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Model saving/loading paths
model_save_path = 'gpt_mini.pth'

if __name__ == '__main__':

    # Training loop with model saving
    for iter in range(max_iters):
        # Evaluate the loss on train and val sets every eval_interval steps
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save the model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, model_save_path)
            print(f"Model saved at step {iter}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()