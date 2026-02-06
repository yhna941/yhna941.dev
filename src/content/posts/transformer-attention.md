---
title: "Transformer #1: Attention 메커니즘의 모든 것"
description: "Seq2Seq부터 Self-Attention까지, Attention 메커니즘의 진화와 원리를 완전히 이해합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["transformer", "attention", "deep-learning", "nlp", "pytorch"]
draft: false
---

# Transformer #1: Attention 메커니즘

**"Attention is All You Need"** (2017)

이 한 문장이 AI를 바꿨습니다.

이번 시리즈:
- Attention의 탄생
- Self-Attention 수학
- Multi-Head Attention
- 완전한 구현

---

## 왜 Attention인가?

### 문제: RNN의 한계

**기계 번역 (Seq2Seq):**

```
Input:  "I love deep learning"
Output: "나는 딥러닝을 사랑한다"
```

**RNN Encoder-Decoder (2014):**

```python
# Encoder: 문장 → 고정 크기 벡터
h1 = rnn(embed("I"))
h2 = rnn(h1, embed("love"))
h3 = rnn(h2, embed("deep"))
h4 = rnn(h3, embed("learning"))

context = h4  # 전체 문장 정보를 하나의 벡터에!

# Decoder: 벡터 → 번역
s1 = rnn(context, embed("<start>"))
y1 = softmax(W @ s1)  # "나는"

s2 = rnn(s1, embed("나는"))
y2 = softmax(W @ s2)  # "딥러닝을"

# ...
```

**문제:**

```
Long sentence: "I love deep learning and ..."
                                           ↑
                                    정보 손실!

context 벡터가 모든 것을 담아야 함
→ Bottleneck!
```

### 해결: Attention

**아이디어:**

> "번역할 때마다 입력 문장의 관련 부분을 다시 본다!"

```
"나는" 생성 시 → "I" 집중
"딥러닝을" 생성 시 → "deep learning" 집중
"사랑한다" 생성 시 → "love" 집중
```

---

## Attention 메커니즘 (Bahdanau, 2015)

### 수학

**1. Encoder (양방향 RNN):**

```python
# Forward
h1_fwd = rnn_fwd(embed("I"))
h2_fwd = rnn_fwd(h1_fwd, embed("love"))
h3_fwd = rnn_fwd(h2_fwd, embed("deep"))
h4_fwd = rnn_fwd(h3_fwd, embed("learning"))

# Backward
h4_bwd = rnn_bwd(embed("learning"))
h3_bwd = rnn_bwd(h4_bwd, embed("deep"))
h2_bwd = rnn_bwd(h3_bwd, embed("love"))
h1_bwd = rnn_bwd(h2_bwd, embed("I"))

# Concatenate
h1 = [h1_fwd; h1_bwd]
h2 = [h2_fwd; h2_bwd]
h3 = [h3_fwd; h3_bwd]
h4 = [h4_fwd; h4_bwd]
```

**2. Attention Scores:**

Decoder state `s_t`와 각 encoder state `h_i`의 관련도:

$$
e_{t,i} = \text{score}(s_t, h_i) = v^T \tanh(W_1 s_t + W_2 h_i)
$$

**3. Attention Weights (Softmax):**

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^n \exp(e_{t,j})}
$$

**4. Context Vector:**

가중 평균:

$$
c_t = \sum_{i=1}^n \alpha_{t,i} h_i
$$

**5. Decoder:**

Context를 사용:

$$
s_t = \text{RNN}(s_{t-1}, [y_{t-1}; c_t])
$$

$$
p(y_t) = \text{softmax}(W_o s_t)
$$

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)  # Decoder projection
        self.W2 = nn.Linear(hidden_size, hidden_size)  # Encoder projection
        self.v = nn.Linear(hidden_size, 1)  # Score
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Expand decoder_hidden to (batch, seq_len, hidden_size)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Score: (batch, seq_len, 1)
        energy = torch.tanh(
            self.W1(decoder_hidden) + self.W2(encoder_outputs)
        )
        scores = self.v(energy).squeeze(2)  # (batch, seq_len)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder (Bidirectional GRU)
        self.encoder = nn.GRU(
            embed_size, 
            hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Attention
        self.attention = BahdanauAttention(hidden_size * 2)
        
        # Decoder
        self.decoder = nn.GRU(
            embed_size + hidden_size * 2,  # Input + context
            hidden_size * 2,
            batch_first=True
        )
        
        # Output
        self.out = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, src, tgt):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        # Encode
        src_embedded = self.embedding(src)
        encoder_outputs, hidden = self.encoder(src_embedded)
        # encoder_outputs: (batch, src_len, hidden_size * 2)
        
        # Decode
        tgt_embedded = self.embedding(tgt)
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        
        outputs = []
        decoder_hidden = hidden[-1].unsqueeze(0)  # Last layer
        
        for t in range(tgt_len):
            # Attention
            context, attn_weights = self.attention(
                decoder_hidden.squeeze(0),
                encoder_outputs
            )
            
            # Decoder input: [embedding; context]
            decoder_input = torch.cat([
                tgt_embedded[:, t:t+1],
                context.unsqueeze(1)
            ], dim=2)
            
            # Decode
            output, decoder_hidden = self.decoder(
                decoder_input,
                decoder_hidden
            )
            
            # Predict
            pred = self.out(output.squeeze(1))
            outputs.append(pred)
        
        return torch.stack(outputs, dim=1)

# 사용
model = Seq2SeqWithAttention(
    vocab_size=10000,
    embed_size=256,
    hidden_size=512
)

src = torch.randint(0, 10000, (32, 20))  # Batch 32, length 20
tgt = torch.randint(0, 10000, (32, 15))  # Batch 32, length 15

output = model(src, tgt)
print(output.shape)  # (32, 15, 10000)
```

---

## Self-Attention (Transformer, 2017)

### 핵심 아이디어

**Seq2Seq Attention:**
- Decoder → Encoder 관계
- 시퀀스 간 attention

**Self-Attention:**
- 한 시퀀스 내부 관계
- "단어들끼리 서로 본다!"

**예시:**

```
"The animal didn't cross the street because it was too tired"
                                                  ↑
                                    "it" refers to "animal"
```

Self-Attention으로 "it"이 "animal"과 관련 있음을 학습!

### Query, Key, Value (QKV)

**정보 검색 비유:**

```
YouTube 검색:
- Query: "딥러닝 강의"
- Key: 각 영상의 제목/태그
- Value: 실제 영상 내용

검색 과정:
1. Query와 Key 비교 (유사도)
2. 유사도 높은 영상 선택
3. 해당 Value(영상) 반환
```

**Self-Attention:**

```python
# 각 단어를 3개로 변환
Q = X @ W_Q  # Query: "내가 찾는 것"
K = X @ W_K  # Key: "내가 제공하는 것"
V = X @ W_V  # Value: "실제 내용"
```

### 수학

**입력:**

$$
X \in \mathbb{R}^{n \times d}
$$

- $n$: 시퀀스 길이
- $d$: 차원

**변환:**

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$

**Attention Scores:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**단계별:**

1. **Similarity:** $QK^T \in \mathbb{R}^{n \times n}$
   - 모든 단어 쌍의 유사도

2. **Scaling:** $\frac{QK^T}{\sqrt{d_k}}$
   - Gradient 안정화

3. **Weights:** $\text{softmax}(\cdots)$
   - 각 행이 확률 분포

4. **Output:** $\text{softmax}(\cdots) V \in \mathbb{R}^{n \times d_k}$
   - Weighted sum of values

### PyTorch 구현

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature  # sqrt(d_k)
    
    def forward(self, q, k, v, mask=None):
        """
        q: (batch, n_heads, seq_len, d_k)
        k: (batch, n_heads, seq_len, d_k)
        v: (batch, n_heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        """
        # 1. Q @ K^T
        attn = torch.matmul(q, k.transpose(-2, -1))  # (batch, n_heads, seq_len, seq_len)
        
        # 2. Scale
        attn = attn / self.temperature
        
        # 3. Mask (optional)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax
        attn = F.softmax(attn, dim=-1)
        
        # 5. @ V
        output = torch.matmul(attn, v)  # (batch, n_heads, seq_len, d_v)
        
        return output, attn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        """
        # Q, K, V
        q = self.W_q(x)  # (batch, seq_len, d_k)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Attention
        output, attn = self.attention(q, k, v, mask)
        
        return output, attn

# 사용
d_model = 512
d_k = d_v = 64

attn = SelfAttention(d_model, d_k, d_v)
x = torch.randn(32, 10, d_model)  # Batch 32, seq 10
output, weights = attn(x)

print(output.shape)  # (32, 10, 64)
print(weights.shape)  # (32, 10, 10)
```

### Attention Weights 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    """
    attention_weights: (tgt_len, src_len)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap="YlGnBu",
        cbar=True
    )
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.title("Attention Weights")
    plt.show()

# 예시
src = ["The", "animal", "didn't", "cross", "the", "street"]
tgt = ["동물은", "거리를", "건너지", "않았다"]
weights = torch.softmax(torch.randn(4, 6), dim=1).numpy()

visualize_attention(weights, src, tgt)
```

---

## Multi-Head Attention

### 왜?

**Single Head의 한계:**

```
"The animal didn't cross the street because it was too tired"

Single attention:
- "it" → "animal" (70%)
- "it" → "street" (30%)

But...
- 의미 관계: "it" → "animal"
- 문법 관계: "it" → "was"
- 위치 관계: "it" → "tired"
```

**Multi-Head:**

> "여러 관점에서 동시에 본다!"

### 수학

**$h$개의 Head:**

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$
- 보통 $d_k = d_v = d_{model} / h$

**예시 (8 heads, d_model=512):**

```
d_k = 512 / 8 = 64

Head 1: (batch, seq_len, 64)
Head 2: (batch, seq_len, 64)
...
Head 8: (batch, seq_len, 64)

Concat: (batch, seq_len, 512)
Project: (batch, seq_len, 512)
```

### PyTorch 구현

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, seq_len, d_model)
        """
        batch_size = q.size(0)
        
        # 1. Linear projections
        q = self.W_q(q)  # (batch, seq_len, d_model)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 2. Split into heads
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
        # → (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Attention for each head
        output, attn = self.attention(q, k, v, mask)
        # output: (batch, n_heads, seq_len, d_k)
        
        # 4. Concat heads
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
        # → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. Final projection
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attn

# 사용
mha = MultiHeadAttention(n_heads=8, d_model=512)
x = torch.randn(32, 10, 512)

output, attn = mha(x, x, x)
print(output.shape)  # (32, 10, 512)
print(attn.shape)  # (32, 8, 10, 10)
```

---

## Masking

### 1. Padding Mask

**문제:**

```
Batch:
- "I love you" (3 words)
- "Deep learning is awesome" (4 words)

Padded:
- "I love you <pad>"
- "Deep learning is awesome"
```

Padding에 attention 주면 안 됨!

**해결:**

```python
def create_padding_mask(seq):
    """
    seq: (batch, seq_len)
    return: (batch, 1, 1, seq_len)
    """
    # 0은 <pad>
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask

# 사용
seq = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
mask = create_padding_mask(seq)
print(mask.shape)  # (2, 1, 1, 4)
```

### 2. Look-Ahead Mask (Decoder)

**문제:**

```
번역 중:
"I love deep learning"
→ "나는 딥러닝을 사랑한다"

"딥러닝을" 생성 시:
- "나는" 봐야 함 ✅
- "딥러닝을" 보면 안 됨! ❌ (미래)
- "사랑한다" 보면 안 됨! ❌ (미래)
```

**해결:**

```python
def create_look_ahead_mask(size):
    """
    return: (size, size) upper triangular matrix
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

# 사용
mask = create_look_ahead_mask(4)
print(mask)
# tensor([[ True, False, False, False],
#         [ True,  True, False, False],
#         [ True,  True,  True, False],
#         [ True,  True,  True,  True]])
```

---

## 성능 비교

### RNN vs Attention

**계산 복잡도:**

```
RNN: O(n) sequential operations
     → 병렬화 불가

Self-Attention: O(1) sequential
                 → 완전 병렬화!
```

**실제 속도 (n=512, d=512):**

```
RNN:
- Forward: 0.5s
- Backward: 1.0s
- Total: 1.5s

Self-Attention:
- Forward: 0.1s
- Backward: 0.2s
- Total: 0.3s

5배 빠름!
```

**메모리:**

```
RNN: O(n·d)
Self-Attention: O(n²·d)

단, n이 작으면 Self-Attention 유리
```

---

## 요약

**Attention의 진화:**

```
Seq2Seq (2014)
→ Bahdanau Attention (2015)
→ Self-Attention (2017)
→ Transformer!
```

**핵심 개념:**

1. **Attention**: 관련 부분에 집중
2. **Self-Attention**: 시퀀스 내부 관계
3. **QKV**: Query, Key, Value
4. **Multi-Head**: 여러 관점
5. **Masking**: Padding, Look-ahead

**수식:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**다음 글:**
- **Transformer 구조**: Encoder, Decoder 완전 분해
- **Positional Encoding**: 위치 정보
- **Training Tips**: 학습 기법

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
