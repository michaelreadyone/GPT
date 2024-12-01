import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np
import time
from datetime import datetime

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
training = True
ic(f'training: {training}, n_head: {n_head}, n_layer: {n_layer}')

torch.manual_seed(1337)

def plot_1d_bar(array, filename, title="Heatmap", cmap="viridis", xlabel="X-axis", ylabel="Y-axis"):

    # If the input is a PyTorch tensor, move it to CPU and convert to NumPy
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    
    # Check if the array is 1D
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    
    # Create the bar plot
    plt.figure(figsize=(15,12)) 
    indices = np.arange(len(array))  # Create indices for the bars
    plt.bar(indices, array)
    
    # Add labels, title, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save specified file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to free memory


def save_heatmap(array, filename, title="Heatmap", cmap="viridis", xlabel="X-axis", ylabel="Y-axis"):

    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    
    # Check if the array is 2D
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    # Create the heatmap
    plt.figure(figsize=(15,12)) 
    plt.imshow(array, cmap=cmap, interpolation="nearest")
    plt.colorbar()  # Add color bar
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Save the heatmap to the specified file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, head_idx, cache, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.cache = cache
        self.warm_up = True

    def forward(self, x, training = training):
        
        if self.head_idx == 0 and self.layer_idx == 0:
            ic(f"=== Att begins at head {self.head_idx} ===")
        B,T,C = x.shape
        
        if training == True:
            k = self.key(x) # (B,T,hs)
            q = self.query(x)
            v = self.value(x)
            
            att = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            att = F.softmax(att, dim=-1) # (B, T, T)
            # att = self.dropout(att)
            out = att @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            if self.head_idx == 0 and self.layer_idx == 0:
                # ic(x.shape)
                # ic(x)
                ic(k.shape)
                ic(k)
                # ic(v.shape)
                # ic(v)
            
            

            datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # ic(wei[-1,:,-1])

        else:
            ic(self.warm_up)
            if self.warm_up:
                k = self.key(x) # (B,T,hs)
                q = self.query(x)
                v = self.value(x)    
                att = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
                att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
                att = F.softmax(att, dim=-1) # (B, T, T)
                out = att @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
                
                k_cache = k
                v_cache = v
                if self.head_idx == 0 and self.layer_idx == 0:
                    ic(k_cache.shape)
                    ic(k_cache)
                    # ic(v_cache.shape)
                    # ic(v_cache)
                self.cache = torch.cat((k_cache.unsqueeze(0), v_cache.unsqueeze(0)))
                self.warm_up = False
            else:
        
                x_last = x[:,-1,:]
                q_last = self.query(x_last) # (B,1,hs)
                k_last = self.key(x_last)   # (B,1,hs)
                v_last = self.value(x_last) # (B,1,hs)
                q = self.key(x)
            
                k_cache = self.cache[0]
                k_cache = torch.cat((k_cache, k_last.unsqueeze(0)), dim=1)
                k_cache = k_cache[:,-block_size:,:]
                
                v_cache = self.cache[1]
                v_cache = torch.cat((v_cache, v_last.unsqueeze(0)), dim=1)
                v_cache = v_cache[:,-block_size:,:]
                
                self.cache = torch.cat((k_cache.unsqueeze(0), v_cache.unsqueeze(0)))


                att = q_last @ k_cache.transpose(-2,-1) * k_cache.shape[-1]**-0.5 # (B, 1, hs) @ (B, hs, T) -> (B, 1, T)
                att = F.softmax(att, dim=-1) # (B, 1, T)
                if self.head_idx == 0 and self.layer_idx == 0:
                    ic(k_cache.shape)
                    ic(k_cache)
                    # ic(v_cache.shape)
                    # ic(v_cache)
                # logger.info(f'k_cache: {k_cache}')
                # logger.info(f'v_cache: {v_cache}')
                out = att @ v_cache # (B, 1, T) @ (B, T, hs) -> (B, 1, hs)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, layer_idx):
        super().__init__()
        cache=[None for _ in range(num_heads)]
        self.heads = nn.ModuleList([Head(head_size, i, cache[i], layer_idx) for i in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, layer_idx):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.layer_idx = layer_idx
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, layer_idx)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # ic(f"+++ Block start at layer {self.layer_idx} +++")
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, layer_idx=i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        ic(f"*** GPT model called, idx shape: {idx.shape} ***")
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        ic(x.shape)
        ic(x)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        times_per_token = []
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # ic(idx.shape)
            start_time = time.time()
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            time_diff = time.time() - start_time
            # ic(time_diff)
            # ic(decode(idx_next[-1].tolist()))
            times_per_token.append(time_diff)
        return idx, times_per_token

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Model saving/loading paths
model_save_path = 'gpt_language_model.pth'

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

