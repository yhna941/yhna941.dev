---
title: "LLM Inference 최적화 #2: KV Caching 완전 정복"
description: "Transformer의 핵심 최적화 기법인 KV Caching을 메모리 레이아웃부터 실제 구현까지 완전히 이해해봅시다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "kv-cache", "attention", "inference"]
draft: false
---

# LLM Inference 최적화 #2: KV Caching

LLM inference에서 **KV Cache**는 필수입니다. 이게 없으면 토큰 하나 생성할 때마다 **모든 이전 토큰을 다시 계산**해야 하거든요.

GPT-3가 "Hello, how are you?"를 생성하는데 KV Cache 없으면 **100배 느립니다**.

어떻게 동작하는지, 어떻게 구현하는지 완전히 파헤쳐봅시다.

---

## 문제: Autoregressive 생성은 반복이 많다

### 토큰 하나씩 생성

LLM은 한 번에 한 토큰씩 생성합니다.

```
Input: "Translate to Korean:"
Step 1: → "안"
Step 2: "안" → "녕"
Step 3: "안녕" → "하"
Step 4: "안녕하" → "세"
Step 5: "안녕하세" → "요"
```

### Attention은 모든 이전 토큰을 본다

각 단계마다 **모든 이전 토큰**에 attention을 계산합니다.

```
Step 1: "Translate to Korean:" + "안"
  Query: "안"
  Key/Value: ["Translate", "to", "Korean", ":"]

Step 2: "Translate to Korean: 안" + "녕"
  Query: "녕"
  Key/Value: ["Translate", "to", "Korean", ":", "안"]

Step 3: "Translate to Korean: 안녕" + "하"
  Query: "하"
  Key/Value: ["Translate", "to", "Korean", ":", "안", "녕"]
```

### 문제: 중복 계산

"Translate to Korean:"의 Key/Value는 **매번 똑같은데** 계속 다시 계산합니다!

```python
# 매 스텝마다
for token in ["Translate", "to", "Korean", ":"]:
    K = linear_k(token_embedding)  # 똑같은 계산 반복!
    V = linear_v(token_embedding)
```

**엄청난 낭비입니다.**

---

## 해결책: KV Cache

아이디어는 간단합니다:

> **한 번 계산한 Key, Value를 저장해두고 재사용하자!**

### Before (No Cache)

```python
def generate_token(prompt_tokens, generated_tokens):
    all_tokens = prompt_tokens + generated_tokens
    
    # 매번 전부 계산
    K = compute_keys(all_tokens)      # 비싼 연산!
    V = compute_values(all_tokens)    # 비싼 연산!
    
    Q = compute_query(generated_tokens[-1])
    
    output = attention(Q, K, V)
    next_token = sample(output)
    return next_token
```

### After (With Cache)

```python
def generate_token(new_token, kv_cache):
    # 새 토큰의 K, V만 계산
    new_K = compute_key(new_token)
    new_V = compute_value(new_token)
    
    # 캐시에 추가
    kv_cache['K'].append(new_K)
    kv_cache['V'].append(new_V)
    
    # 저장된 K, V 사용
    K = kv_cache['K']  # 이미 계산된 것들!
    V = kv_cache['V']
    
    Q = compute_query(new_token)
    output = attention(Q, K, V)
    next_token = sample(output)
    return next_token
```

---

## 수식으로 이해하기

### Standard Attention

```
Q = X @ W_Q    # [seq_len, d_model] @ [d_model, d_k]
K = X @ W_K    # [seq_len, d_model] @ [d_model, d_k]
V = X @ W_V    # [seq_len, d_model] @ [d_v]

scores = Q @ K^T / sqrt(d_k)        # [seq_len, seq_len]
attn = softmax(scores, dim=-1)
output = attn @ V                   # [seq_len, d_v]
```

### Autoregressive Generation (No Cache)

**Step t**: 토큰 `x_t` 생성

```
# 전체 시퀀스 [x_1, x_2, ..., x_{t-1}] 처리
K_{1:t-1} = [x_1, ..., x_{t-1}] @ W_K
V_{1:t-1} = [x_1, ..., x_{t-1}] @ W_V

# 새 쿼리
Q_t = x_{t-1} @ W_Q

# Attention
output_t = attention(Q_t, K_{1:t-1}, V_{1:t-1})
x_t = sample(output_t)
```

**문제**: `K_{1:t-1}`과 `V_{1:t-1}`을 매번 다시 계산!

### With KV Cache

**Step 1** (초기화):
```
# Prompt 처리
K_cache = [x_1, ..., x_n] @ W_K
V_cache = [x_1, ..., x_n] @ W_V
```

**Step t** (생성):
```
# 새 토큰만 계산
K_new = x_{t-1} @ W_K
V_new = x_{t-1} @ W_V

# 캐시에 추가
K_cache = concat(K_cache, K_new)
V_cache = concat(V_cache, V_new)

# Attention (캐시 사용)
Q_t = x_{t-1} @ W_Q
output_t = attention(Q_t, K_cache, V_cache)
```

---

## 메모리 레이아웃

### Multi-Head Attention

실제로는 여러 개의 head가 있습니다.

```python
num_layers = 32
num_heads = 32
head_dim = 128
seq_len = 2048

# 각 레이어, 각 헤드마다 K, V
K_cache.shape = [num_layers, seq_len, num_heads, head_dim]
V_cache.shape = [num_layers, seq_len, num_heads, head_dim]
```

### 메모리 크기 계산

**LLaMA-7B 예시:**
- Layers: 32
- Heads: 32  
- Head dim: 128
- Sequence: 2048
- Data type: float16 (2 bytes)

```
KV_cache_size = 2 (K+V) × 32 (layers) × 2048 (seq) × 32 (heads) × 128 (dim) × 2 (bytes)
              = 1,073,741,824 bytes
              = 1 GB
```

**배치 크기 16이면 16GB!**

### 텐서 레이아웃 최적화

**Naive Layout** (느림):
```python
# [batch, num_layers, seq_len, num_heads, head_dim]
K_cache[b, l, s, h, d]
```

**문제**: 메모리 접근이 비효율적 (stride가 큼)

**Optimized Layout** (빠름):
```python
# [num_layers, batch, num_heads, seq_len, head_dim]
K_cache[l, b, h, s, d]
```

**이유**: 같은 head의 데이터가 연속으로 배치 → 캐시 히트율 ↑

---

## 구현 예제

### 1. 기본 KV Cache 클래스

```python
import torch

class KVCache:
    def __init__(self, batch_size, num_layers, num_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # 캐시 버퍼 미리 할당
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        # 현재 시퀀스 길이 추적
        self.seq_len = torch.zeros(batch_size, dtype=torch.long)
    
    def update(self, layer_idx, key_states, value_states):
        """
        새 K, V 추가
        
        Args:
            layer_idx: 레이어 인덱스
            key_states: [batch, num_heads, 1, head_dim] - 새 토큰의 key
            value_states: [batch, num_heads, 1, head_dim] - 새 토큰의 value
        """
        batch_size = key_states.shape[0]
        
        for b in range(batch_size):
            pos = self.seq_len[b]
            
            # 캐시에 쓰기
            self.k_cache[layer_idx, b, :, pos, :] = key_states[b, :, 0, :]
            self.v_cache[layer_idx, b, :, pos, :] = value_states[b, :, 0, :]
        
        # 길이 증가
        self.seq_len += 1
    
    def get(self, layer_idx):
        """
        현재까지의 K, V 반환
        
        Returns:
            K: [batch, num_heads, seq_len, head_dim]
            V: [batch, num_heads, seq_len, head_dim]
        """
        max_len = self.seq_len.max().item()
        
        K = self.k_cache[layer_idx, :, :, :max_len, :]
        V = self.v_cache[layer_idx, :, :, :max_len, :]
        
        return K, V
    
    def clear(self):
        """캐시 초기화"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len.zero_()
```

### 2. Attention with Cache

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, kv_cache=None, layer_idx=0):
        """
        Args:
            x: [batch, seq_len, d_model]
            kv_cache: KVCache 객체
            layer_idx: 현재 레이어 인덱스
        """
        batch_size, seq_len, d_model = x.shape
        
        # Query 계산 (항상 새로 계산)
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        if kv_cache is None:
            # 캐시 없음: 전체 계산
            K = self.W_k(x)
            V = self.W_v(x)
            
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 캐시 사용: 새 토큰만 계산
            new_K = self.W_k(x[:, -1:, :])  # 마지막 토큰만
            new_V = self.W_v(x[:, -1:, :])
            
            new_K = new_K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            new_V = new_V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 캐시 업데이트
            kv_cache.update(layer_idx, new_K, new_V)
            
            # 전체 K, V 가져오기
            K, V = kv_cache.get(layer_idx)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # [batch, num_heads, seq_len, head_dim] → [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        
        output = self.W_o(output)
        return output
```

### 3. 생성 루프

```python
def generate(model, prompt_tokens, max_new_tokens, kv_cache):
    """
    Args:
        model: Transformer 모델
        prompt_tokens: [batch, prompt_len]
        max_new_tokens: 생성할 토큰 수
        kv_cache: KVCache 객체
    """
    kv_cache.clear()
    generated = prompt_tokens.clone()
    
    # Prefill: 프롬프트 처리 (캐시 채우기)
    with torch.no_grad():
        logits = model(prompt_tokens, kv_cache=kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    generated = torch.cat([generated, next_token], dim=1)
    
    # Decode: 토큰 하나씩 생성
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # 마지막 토큰만 입력 (캐시 사용)
            logits = model(next_token, kv_cache=kv_cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # EOS 체크
        if (next_token == EOS_TOKEN).all():
            break
    
    return generated
```

---

## 최적화 기법

### 1. Flash Attention과의 통합

Flash Attention은 메모리 효율적인 attention 구현입니다.

```python
from flash_attn import flash_attn_func

def flash_attention_with_cache(Q, K_cache, V_cache, new_K, new_V):
    """
    Flash Attention + KV Cache
    
    Args:
        Q: [batch, num_heads, 1, head_dim] - 새 쿼리
        K_cache: [batch, num_heads, seq_len, head_dim] - 캐시된 key
        V_cache: [batch, num_heads, seq_len, head_dim] - 캐시된 value
        new_K: [batch, num_heads, 1, head_dim] - 새 key
        new_V: [batch, num_heads, 1, head_dim] - 새 value
    """
    # 캐시 업데이트
    K = torch.cat([K_cache, new_K], dim=2)
    V = torch.cat([V_cache, new_V], dim=2)
    
    # Flash Attention 실행
    output = flash_attn_func(
        Q, K, V,
        causal=True,
        softmax_scale=1.0 / (Q.shape[-1] ** 0.5)
    )
    
    return output, K, V
```

### 2. Multi-Query Attention (MQA)

메모리를 더 줄이는 방법: **Head 간 K, V 공유**

```python
# Standard Multi-Head
K.shape = [batch, num_heads, seq_len, head_dim]
V.shape = [batch, num_heads, seq_len, head_dim]

# Multi-Query Attention
K.shape = [batch, 1, seq_len, head_dim]  # 모든 head가 공유!
V.shape = [batch, 1, seq_len, head_dim]
Q.shape = [batch, num_heads, 1, head_dim]  # Query는 여전히 여러 개
```

**메모리 절약**: num_heads배 (예: 32배!)

**성능**: 약간 떨어짐 (1-2%)

**사용 예**: PaLM, StarCoder

### 3. Grouped-Query Attention (GQA)

MQA의 중간 타협: **Head를 그룹으로 묶어서 공유**

```python
num_heads = 32
num_kv_heads = 4  # 4개 그룹

# GQA
K.shape = [batch, num_kv_heads, seq_len, head_dim]
V.shape = [batch, num_kv_heads, seq_len, head_dim]
Q.shape = [batch, num_heads, 1, head_dim]

# Query head들을 KV head에 매핑
for i in range(num_heads):
    kv_idx = i // (num_heads // num_kv_heads)
    # Q[i]가 K[kv_idx], V[kv_idx] 사용
```

**메모리 절약**: num_heads / num_kv_heads배 (예: 8배)

**성능**: 거의 동일 (0.5% 차이)

**사용 예**: LLaMA-2, Mistral

---

## 실전 구현: HuggingFace Transformers

### Cache 클래스

HuggingFace는 `DynamicCache` 클래스를 제공합니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Cache 초기화
past_key_values = DynamicCache()

# 프롬프트 처리
prompt = "Translate to Korean: Hello"
inputs = tokenizer(prompt, return_tensors="pt")

# Prefill
outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Decode
generated = inputs.input_ids
for _ in range(50):
    # 새 토큰만 입력
    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        use_cache=True
    )
    
    # past_key_values가 자동으로 업데이트됨!
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    
    if next_token.item() == tokenizer.eos_token_id:
        break

print(tokenizer.decode(generated[0]))
```

### Static Cache (고정 크기)

메모리를 미리 할당해서 더 빠르게:

```python
from transformers import StaticCache

# 최대 길이 미리 지정
max_cache_len = 4096
past_key_values = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=max_cache_len,
    device="cuda",
    dtype=torch.float16
)

# 사용법은 동일
outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
```

---

## 벤치마크

### 속도 비교

**시나리오**: LLaMA-7B, 프롬프트 100 토큰, 생성 100 토큰

| 방식 | 시간 (초) | 속도 |
|------|----------|------|
| No Cache | 45.2 | 1x |
| KV Cache | 0.52 | **87x** |

### 메모리 비교

**배치 크기 1, 시퀀스 2048:**

| 방식 | 메모리 사용량 |
|------|--------------|
| Standard MHA | 1.0 GB |
| Multi-Query Attention (MQA) | 0.03 GB (32배 절약) |
| Grouped-Query (GQA, 4 groups) | 0.125 GB (8배 절약) |

---

## 고급 패턴

### 1. Prefix Caching

같은 프롬프트를 여러 번 쓸 때:

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}
    
    def get_or_compute(self, prefix_tokens, model):
        key = tuple(prefix_tokens.tolist())
        
        if key in self.cache:
            return self.cache[key].clone()
        
        # 계산
        with torch.no_grad():
            outputs = model(prefix_tokens, use_cache=True)
            kv_cache = outputs.past_key_values
        
        self.cache[key] = kv_cache
        return kv_cache.clone()

# 사용
prefix_cache = PrefixCache()

system_prompt = "You are a helpful assistant."
system_tokens = tokenizer(system_prompt, return_tensors="pt").input_ids

# 첫 요청
kv = prefix_cache.get_or_compute(system_tokens, model)  # 계산
generate(model, user_query_1, kv)

# 두 번째 요청
kv = prefix_cache.get_or_compute(system_tokens, model)  # 캐시 히트!
generate(model, user_query_2, kv)
```

### 2. Streaming with Cache

스트리밍 생성:

```python
def generate_stream(model, prompt_tokens, kv_cache):
    """토큰을 하나씩 yield"""
    kv_cache.clear()
    
    # Prefill
    with torch.no_grad():
        logits = model(prompt_tokens, kv_cache=kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    yield next_token.item()
    
    # Decode
    while True:
        with torch.no_grad():
            logits = model(next_token, kv_cache=kv_cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        token_id = next_token.item()
        if token_id == EOS_TOKEN:
            break
        
        yield token_id

# 사용
for token_id in generate_stream(model, prompt, cache):
    print(tokenizer.decode([token_id]), end='', flush=True)
```

---

## Pitfalls (주의사항)

### 1. 배치 처리 시 길이 차이

배치 내 시퀀스 길이가 다르면:

```python
batch = [
    [1, 2, 3, 4, 5],        # 길이 5
    [1, 2, 3],              # 길이 3
    [1, 2, 3, 4, 5, 6, 7],  # 길이 7
]
```

**문제**: KV Cache를 어떻게 관리?

**해결책 1**: Padding (메모리 낭비)
```python
# 최대 길이로 패딩
padded = pad_sequence(batch, max_len=7)
```

**해결책 2**: Separate caches (추천)
```python
caches = [KVCache() for _ in range(batch_size)]
```

### 2. Dynamic Batching

생성 중 배치가 바뀌면:

```python
# 처음: 4개 요청
batch_size = 4

# 중간에 2개 끝남
batch_size = 2  # 캐시 어떻게?
```

**해결책**: Request별로 캐시 관리 (vLLM 방식)

### 3. Out of Memory

긴 시퀀스:

```python
seq_len = 10000  # 길어!
# KV Cache = 10000 × ...  → OOM
```

**해결책**: Sliding window attention
```python
# 최근 2048 토큰만 유지
if seq_len > 2048:
    K_cache = K_cache[:, :, -2048:, :]
    V_cache = V_cache[:, :, -2048:, :]
```

---

## 요약

**KV Caching**은:

1. **Key와 Value를 저장**해서 재사용
2. **토큰당 계산량**: O(n²) → O(n)
3. **속도**: 50-100배 향상
4. **메모리**: 시퀀스 길이에 비례

**최적화 기법**:
- Multi-Query Attention (MQA): 32배 메모리 절약
- Grouped-Query Attention (GQA): 8배 절약, 성능 유지
- Flash Attention: 메모리 효율 + 속도
- Prefix Caching: 반복 프롬프트 최적화

**필수 기법**: 모든 LLM inference에 사용!

---

## 다음 글

**6편: LoRA Fine-tuning**
- 적은 파라미터로 모델 학습하기
- LoRA 수학적 원리
- 실전 구현 & QLoRA

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
