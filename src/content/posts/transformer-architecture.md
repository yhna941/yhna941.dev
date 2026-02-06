---
title: "Transformer #2: 완전한 구조 분해 - Encoder & Decoder"
description: "Transformer의 전체 아키텍처를 처음부터 끝까지 구현하며 완전히 이해합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["transformer", "encoder", "decoder", "deep-learning", "pytorch"]
draft: false
---

# Transformer #2: 완전한 구조

**"Attention is All You Need"** 논문을 **처음부터 끝까지** 구현합니다.

이번 글:
- Transformer 전체 구조
- Encoder & Decoder
- Positional Encoding
- 완전한 구현

---

## Transformer 아키텍처

### 전체 구조

```
Input → Embedding → Positional Encoding
                          ↓
                    ┌─────────┐
                    │ Encoder │ × N
                    └─────────┘
                          ↓
Output → Embedding → Positional Encoding
                          ↓
                    ┌─────────┐
                    │ Decoder │ × N
                    └─────────┘
                          ↓
                    Linear + Softmax
                          ↓
                     Probabilities
```

**하이퍼파라미터 (Base model):**

```python
d_model = 512      # Model dimension
n_heads = 8        # Attention heads
d_ff = 2048        # Feed-forward dimension
n_layers = 6       # Encoder/Decoder layers
dropout = 0.1
max_seq_len = 5000
```

---

## Positional Encoding

### 문제

**Self-Attention은 순서 무시:**

```python
["I", "love", "you"]
["love", "I", "you"]
["you", "love", "I"]

→ 모두 같은 출력!
```

Attention은 **set operation**이지 **sequence operation**이 아님!

### 해결

**위치 정보 추가:**

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- $pos$: 위치 (0, 1, 2, ...)
- $i$: 차원 (0, 1, ..., $d_{model}/2$)

**왜 sin/cos?**

1. **범위 고정**: $[-1, 1]$
2. **상대 위치**: $PE_{pos+k}$는 $PE_{pos}$의 선형 결합
3. **외삽 가능**: 학습 시보다 긴 시퀀스도 OK

### 구현

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Compute div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len]
        return x

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

pe = PositionalEncoding(d_model=128, max_len=100)
encoding = pe.pe.squeeze(0).numpy()

plt.figure(figsize=(15, 5))
sns.heatmap(encoding.T, cmap='RdBu', center=0)
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.show()

# 특정 위치들의 인코딩
positions = [0, 10, 50, 99]
for pos in positions:
    plt.plot(encoding[pos], label=f'pos={pos}')
plt.legend()
plt.title('Positional Encoding for Different Positions')
plt.show()
```

---

## Feed-Forward Network

**각 위치마다 독립적인 MLP:**

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ (보통 $d_{ff} = 4 \times d_{model}$)
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_ff)
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        
        # (batch, seq_len, d_model)
        x = self.w_2(x)
        return x
```

---

## Layer Normalization

**각 샘플마다 정규화:**

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ (mean)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ (variance)

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

**Batch Norm vs Layer Norm:**

```
Batch Norm: 배치 내 같은 feature 정규화
- (batch, seq_len, d_model) → dim=0

Layer Norm: 각 샘플의 feature 정규화
- (batch, seq_len, d_model) → dim=-1

Transformer는 Layer Norm!
→ 시퀀스 길이가 다를 수 있음
```

---

## Encoder Layer

**구조:**

```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (Residual)
  ↓
Feed-Forward
  ↓
Add & Norm (Residual)
  ↓
Output
```

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout)
        
        # Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, 1, 1, seq_len)
        """
        # 1. Self-Attention + Residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

---

## Encoder

**N개의 Encoder Layer:**

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)  # Scale embedding
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        """
        src: (batch, src_len)
        src_mask: (batch, 1, 1, src_len)
        """
        # 1. Embedding (scaled)
        x = self.embedding(src) * self.scale
        
        # 2. Positional Encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. Encoder Layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

# 사용
encoder = Encoder(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    dropout=0.1
)

src = torch.randint(0, 10000, (32, 20))  # Batch 32, length 20
src_mask = create_padding_mask(src)

output = encoder(src, src_mask)
print(output.shape)  # (32, 20, 512)
```

---

## Decoder Layer

**구조:**

```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention (with Encoder output)
  ↓
Add & Norm
  ↓
Feed-Forward
  ↓
Add & Norm
  ↓
Output
```

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout)
        
        # Cross-Attention
        self.cross_attn = MultiHeadAttention(n_heads, d_model, dropout)
        
        # Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        x: (batch, tgt_len, d_model)
        encoder_output: (batch, src_len, d_model)
        src_mask: (batch, 1, 1, src_len)
        tgt_mask: (batch, 1, tgt_len, tgt_len)
        """
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Cross-Attention
        # Query from decoder, Key & Value from encoder
        cross_attn_output, _ = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

---

## Decoder

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        tgt: (batch, tgt_len)
        encoder_output: (batch, src_len, d_model)
        """
        # 1. Embedding
        x = self.embedding(tgt) * self.scale
        
        # 2. Positional Encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. Decoder Layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
```

---

## Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len
        )
        
        # Decoder
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len
        )
        
        # Final linear layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        # Encode
        encoder_output = self.encoder(src, src_mask)
        
        # Decode
        decoder_output = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(decoder_output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """Encoder only"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decoder only"""
        decoder_output = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        return self.fc_out(decoder_output)

# 사용
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    dropout=0.1
)

# Forward
src = torch.randint(0, 10000, (32, 20))
tgt = torch.randint(0, 10000, (32, 15))

src_mask = create_padding_mask(src)
tgt_mask = create_look_ahead_mask(15) & create_padding_mask(tgt)

output = model(src, tgt, src_mask, tgt_mask)
print(output.shape)  # (32, 15, 10000)
```

---

## Mask 생성

```python
def create_padding_mask(seq):
    """
    seq: (batch, seq_len)
    return: (batch, 1, 1, seq_len)
    """
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    """
    return: (1, 1, size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    return mask.unsqueeze(0).unsqueeze(0)

def create_target_mask(tgt):
    """
    Decoder mask: padding + look-ahead
    tgt: (batch, tgt_len)
    return: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)
    
    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(tgt_len)
    
    # Padding mask
    padding_mask = create_padding_mask(tgt)
    
    # Combine
    mask = look_ahead_mask & padding_mask
    return mask
```

---

## Training

### Loss Function

```python
import torch.nn.functional as F

def compute_loss(logits, targets, pad_idx=0):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    # Reshape
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Cross-entropy (ignore padding)
    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=pad_idx,
        reduction='mean'
    )
    
    return loss
```

### Label Smoothing

**Hard labels:**

```
"cat" → [0, 0, 1, 0, 0]  (one-hot)
```

**Smoothed labels:**

```
"cat" → [0.02, 0.02, 0.92, 0.02, 0.02]
```

더 robust!

```python
class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        """
        logits: (batch * seq_len, vocab_size)
        targets: (batch * seq_len,)
        """
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        # Mask padding
        mask = (targets != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)
        
        # KL divergence
        loss = self.criterion(log_probs, true_dist)
        return loss / mask.sum()
```

### Learning Rate Schedule

**Warmup + Decay:**

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})
$$

```python
class NoamScheduler:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def compute_lr(self):
        step = self.current_step
        warmup = self.warmup_steps
        
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (warmup ** -1.5)
        )

# 사용
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(d_model=512, warmup_steps=4000, optimizer=optimizer)
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Shift target (input vs output)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_target_mask(tgt_input)
        
        # Forward
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Train
for epoch in range(num_epochs):
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, device
    )
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
```

---

## Inference

### Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_len, start_token, end_token):
    """
    src: (1, src_len)
    """
    model.eval()
    
    # Encode once
    encoder_output = model.encode(src, src_mask)
    
    # Start with <start> token
    tgt = torch.tensor([[start_token]])
    
    for _ in range(max_len):
        # Create target mask
        tgt_mask = create_target_mask(tgt)
        
        # Decode
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Get last token prediction
        next_token = output[:, -1, :].argmax(dim=-1)
        
        # Append
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        
        # Stop if <end> token
        if next_token.item() == end_token:
            break
    
    return tgt
```

### Beam Search

```python
def beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5):
    """
    src: (1, src_len)
    """
    model.eval()
    
    # Encode
    encoder_output = model.encode(src, src_mask)
    
    # Initialize beam
    # Each beam: (sequence, score)
    beams = [(torch.tensor([[start_token]]), 0.0)]
    
    for _ in range(max_len):
        new_beams = []
        
        for seq, score in beams:
            # Stop if already ended
            if seq[0, -1].item() == end_token:
                new_beams.append((seq, score))
                continue
            
            # Decode
            tgt_mask = create_target_mask(seq)
            output = model.decode(seq, encoder_output, src_mask, tgt_mask)
            
            # Get top-k tokens
            log_probs = F.log_softmax(output[:, -1, :], dim=-1)
            top_probs, top_indices = log_probs.topk(beam_size)
            
            # Expand beam
            for prob, idx in zip(top_probs[0], top_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + prob.item()
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_size
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Stop if all beams ended
        if all(seq[0, -1].item() == end_token for seq, _ in beams):
            break
    
    # Return best sequence
    return beams[0][0]
```

---

## 요약

**Transformer 구조:**

1. **Embedding + Positional Encoding**
2. **Encoder**: Self-Attention + FFN (× N)
3. **Decoder**: Masked Self-Attention + Cross-Attention + FFN (× N)
4. **Output**: Linear + Softmax

**핵심 요소:**

- **Multi-Head Attention**: 여러 관점
- **Positional Encoding**: 위치 정보
- **Residual + LayerNorm**: 안정적 학습
- **Feed-Forward**: 비선형 변환

**하이퍼파라미터 (Base):**

```python
d_model = 512
n_heads = 8
d_ff = 2048
n_layers = 6
dropout = 0.1
```

**다음 글:**
- **BERT**: Encoder-only Transformer
- **GPT**: Decoder-only Transformer
- **Vision Transformer**: 이미지에 Transformer

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
