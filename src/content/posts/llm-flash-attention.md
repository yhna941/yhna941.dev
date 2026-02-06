---
title: "LLM Inference 최적화 #4: Flash Attention - 메모리도 줄이고 속도도 올리고"
description: "Attention의 메모리 복잡도를 O(N²)에서 O(N)으로 줄이는 Flash Attention의 원리와 실전 사용법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "flash-attention", "cuda", "memory"]
draft: false
---

# LLM Inference 최적화 #4: Flash Attention

Standard attention의 문제는 **메모리**입니다. 시퀀스 길이가 2배 늘면 메모리는 **4배** 증가합니다. O(N²) 복잡도죠.

**Flash Attention**은 이걸 O(N)으로 줄입니다. 어떻게? **메모리 계층을 이해**하고, **재계산**을 영리하게 씁니다.

결과:
- 메모리: **10-20배 절약**
- 속도: **2-4배 빠름**
- 긴 시퀀스: **가능해짐** (4K → 64K)

---

## 문제: Standard Attention의 메모리 폭탄

### Attention 수식

```
Q, K, V = [seq_len, d_model] 각각
S = QK^T / √d                    # [seq_len, seq_len]  ← 문제!
P = softmax(S)                   # [seq_len, seq_len]  ← 문제!
O = PV                           # [seq_len, d_model]
```

### 메모리 계산

**시퀀스 길이 2048, fp16:**

```
S: [2048, 2048] × 2 bytes = 8 MB
P: [2048, 2048] × 2 bytes = 8 MB
총: 16 MB (per head)
```

32 heads × 32 layers = **16 GB**

**배치 크기 16이면? 256 GB!**

### 문제의 근본

Attention matrix S와 P를 **전체 메모리에 저장**합니다.

```python
# Standard attention
S = Q @ K.T / sqrt(d)      # Materialize: [N, N]
P = softmax(S)             # Materialize: [N, N]
O = P @ V                  # Result: [N, d]
```

**N=4096이면 S, P는 각각 32MB (fp16)**

---

## GPU 메모리 계층

이해의 핵심은 **메모리 속도**입니다.

```
HBM (High Bandwidth Memory):
  - 크기: 40-80 GB
  - 속도: ~1.5 TB/s
  - 느림!

SRAM (On-chip):
  - 크기: ~20 MB (per SM)
  - 속도: ~19 TB/s
  - 빠름! (10배+)
```

**Standard attention은 HBM에 S, P를 쓰고 읽습니다** → 느림!

**Flash Attention은 SRAM만 씁니다** → 빠름!

---

## Flash Attention 핵심 아이디어

### 1. Tiling (타일링)

큰 행렬을 작은 **블록(tile)**로 나눕니다.

```
Q: [N, d] → blocks: [B_q, d]  (B_q = N / num_blocks)
K: [N, d] → blocks: [B_k, d]
V: [N, d] → blocks: [B_k, d]

한 번에 하나의 블록만 SRAM에 로드
```

### 2. Recomputation (재계산)

중간 결과(S, P)를 **저장 안 하고 재계산**합니다.

```
Forward: S, P 저장 안 함 (SRAM에서만 계산)
Backward: S, P 다시 계산 (Q, K, V에서)
```

**Trade-off:**
- 메모리: ↓↓↓ (S, P 안 저장)
- 계산: ↑ (재계산)
- 총 속도: ↑ (메모리 I/O가 병목이라 계산 증가는 괜찮음)

### 3. Online Softmax

Softmax를 **스트리밍**으로 계산합니다.

**Standard softmax:**
```python
# 전체 행 필요
S_row = [s_1, s_2, ..., s_N]
max_val = max(S_row)
exp_vals = [exp(s_i - max_val) for s_i in S_row]
sum_exp = sum(exp_vals)
P_row = [e / sum_exp for e in exp_vals]
```

**문제:** 전체 행을 메모리에 저장 필요

**Online softmax:**
```python
# 블록씩 처리
max_val = -inf
sum_exp = 0

for block in blocks:
    old_max = max_val
    max_val = max(max_val, max(block))
    
    # 이전 값들 rescale
    sum_exp = sum_exp * exp(old_max - max_val)
    
    # 현재 블록 추가
    sum_exp += sum(exp(block - max_val))

# 최종 정규화
P_row = exp_vals / sum_exp
```

블록 단위로 처리 가능!

---

## Flash Attention 알고리즘

### Pseudo-code

```python
def flash_attention(Q, K, V, block_size):
    N, d = Q.shape
    O = zeros(N, d)
    l = zeros(N)  # sum of exp (for softmax)
    m = fill(-inf, N)  # max value (for softmax)
    
    # Q를 블록으로 나눔
    for Q_block in split(Q, block_size):
        # O, l, m의 해당 블록
        O_block = zeros(block_size, d)
        l_block = zeros(block_size)
        m_block = fill(-inf, block_size)
        
        # K, V를 블록으로 순회
        for K_block, V_block in zip(split(K, block_size), split(V, block_size)):
            # Attention scores (SRAM에서만)
            S_block = Q_block @ K_block.T / sqrt(d)
            
            # Online softmax update
            m_new = max(m_block, max(S_block, axis=1))
            
            # Rescale 이전 값들
            scale = exp(m_block - m_new)
            O_block = O_block * scale[:, None]
            l_block = l_block * scale
            
            # 현재 블록 추가
            P_block = exp(S_block - m_new[:, None])
            O_block += P_block @ V_block
            l_block += sum(P_block, axis=1)
            
            m_block = m_new
        
        # 최종 정규화
        O_block = O_block / l_block[:, None]
        
        # 글로벌 출력에 쓰기
        write_to(O, O_block)
    
    return O
```

### 핵심 트릭

1. **S, P 저장 안 함**: SRAM에서만 계산
2. **블록 단위 처리**: 작은 블록만 SRAM에 로드
3. **Online softmax**: 블록씩 softmax 업데이트
4. **Rescaling**: 새 max 값에 맞춰 이전 값 조정

---

## CUDA 구현 핵심

### 1. 메모리 배치

```cuda
__global__ void flash_attention_kernel(
    const float* Q,  // [batch, heads, N, d]
    const float* K,
    const float* V,
    float* O,
    int N, int d, int block_size
) {
    // Shared memory (SRAM)
    __shared__ float Q_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ float V_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ float S_smem[BLOCK_SIZE][BLOCK_SIZE];
    
    // 각 스레드 블록이 Q의 한 블록 처리
    int q_block_idx = blockIdx.x;
    
    // Q 블록을 shared memory에 로드
    load_block_to_smem(Q, Q_smem, q_block_idx);
    
    // 출력 누적용
    float O_local[HEAD_DIM] = {0};
    float l_local = 0.0f;
    float m_local = -INFINITY;
    
    // K, V 블록들 순회
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        // K, V 블록 로드
        load_block_to_smem(K, K_smem, k_block_idx);
        load_block_to_smem(V, V_smem, k_block_idx);
        __syncthreads();
        
        // S = Q @ K^T (shared memory에서)
        compute_attention_scores(Q_smem, K_smem, S_smem);
        __syncthreads();
        
        // Online softmax & output update
        update_output_online(
            S_smem, V_smem,
            O_local, &l_local, &m_local
        );
    }
    
    // 최종 정규화 & 글로벌 메모리에 쓰기
    normalize_and_write(O, O_local, l_local);
}
```

### 2. Warp-level 최적화

```cuda
// Warp reduction for max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

---

## 실전 사용: PyTorch

### 설치

```bash
pip install flash-attn --no-build-isolation
```

### 기본 사용

```python
import torch
from flash_attn import flash_attn_func

# 입력 준비
batch_size = 4
num_heads = 32
seq_len = 2048
head_dim = 128

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# Flash Attention
output = flash_attn_func(
    Q, K, V,
    causal=True,  # Causal masking (GPT-style)
    softmax_scale=1.0 / (head_dim ** 0.5)
)

# output: [batch, seq_len, num_heads, head_dim]
```

### Transformer Layer에 통합

```python
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Flash Attention 사용
        self.attn = FlashSelfAttention(
            causal=True,
            softmax_scale=None,  # 자동 계산
            attention_dropout=0.1
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        
        # Attention
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        
        return x
```

### HuggingFace Transformers 통합

```python
from transformers import AutoModelForCausalLM

# Flash Attention 자동 사용 (최신 버전)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # 여기!
)

# 추론
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
```

---

## Flash Attention 2

2023년에 나온 개선 버전입니다.

### 주요 개선

**1. Work partitioning**
- GPU 워크로드를 더 효율적으로 분산
- Warp 단위 병렬화

**2. Non-matmul FLOPs 줄임**
- Softmax, rescaling 최적화

**3. Low-occupancy 개선**
- 작은 배치/헤드에서도 빠름

### 성능 비교

**LLaMA-7B, seq_len=2048:**

| 버전 | 속도 (ms) | 메모리 (GB) |
|------|----------|------------|
| Standard | 45 | 16 |
| Flash Attention 1 | 18 (2.5x) | 2 (8x) |
| Flash Attention 2 | 12 (3.75x) | 2 (8x) |

---

## 한계와 트레이드오프

### 1. Recomputation 오버헤드

Forward는 빠른데, **Backward는 계산 2배**입니다.

```
Standard:
  Forward: S, P 저장
  Backward: S, P 읽어서 gradient 계산

Flash Attention:
  Forward: S, P 저장 안 함
  Backward: S, P 재계산 + gradient 계산
```

**하지만:** 메모리 I/O 절약이 더 커서 **전체적으론 빠름**

### 2. 긴 시퀀스에서만 빛남

짧은 시퀀스(< 512)에서는 오버헤드가 클 수 있습니다.

```python
# 시퀀스 길이별 속도업
seq_len=256:   1.2x
seq_len=512:   1.5x
seq_len=1024:  2.0x
seq_len=2048:  3.0x
seq_len=4096:  4.0x
```

### 3. FP16/BF16만 지원

FP32는 지원 안 됩니다. (CUDA 최적화 때문)

---

## 고급 기법

### 1. Flash Attention + KV Cache

```python
from flash_attn import flash_attn_with_kvcache

# Prefill (전체 시퀀스)
cache_k = torch.empty(batch, seqlen_k, num_heads, head_dim, device='cuda', dtype=torch.float16)
cache_v = torch.empty_like(cache_k)

output = flash_attn_with_kvcache(
    q, k, v,
    cache_k, cache_v,
    cache_seqlens=None,  # 처음
    causal=True
)

# Decode (새 토큰)
new_q = q[:, -1:, :, :]
new_k = k[:, -1:, :, :]
new_v = v[:, -1:, :, :]

output = flash_attn_with_kvcache(
    new_q, new_k, new_v,
    cache_k, cache_v,
    cache_seqlens=prev_seqlen,  # 이전 길이
    causal=True
)
```

### 2. Multi-Query Attention (MQA)

```python
# Q: [batch, seq, num_heads, head_dim]
# K, V: [batch, seq, 1, head_dim]  ← 1개 head

output = flash_attn_func(
    Q, K, V,
    causal=True
)
# 자동으로 K, V를 num_heads만큼 broadcast
```

### 3. Grouped-Query Attention (GQA)

```python
# Q: [batch, seq, 32, 128]  ← 32 heads
# K, V: [batch, seq, 4, 128]  ← 4 groups

output = flash_attn_func(
    Q, K, V,
    causal=True
)
# Q의 8개 head당 K, V의 1개 group 사용
```

---

## 벤치마크

### A100 80GB, LLaMA-7B

**Forward pass (ms):**

| Seq Len | Standard | Flash v1 | Flash v2 |
|---------|----------|----------|----------|
| 512 | 8.2 | 6.5 | 5.1 |
| 1024 | 18.5 | 9.2 | 7.3 |
| 2048 | 45.3 | 18.1 | 12.4 |
| 4096 | 125.7 | 42.8 | 28.6 |
| 8192 | OOM | 98.3 | 65.2 |

**메모리 (GB):**

| Seq Len | Standard | Flash |
|---------|----------|-------|
| 2048 | 16 | 2 |
| 4096 | 64 | 4 |
| 8192 | OOM | 8 |

---

## 실전 예제: 긴 문서 처리

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flash Attention 2 사용
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 긴 문서 (16K 토큰)
long_document = """
[여기에 16,000 단어짜리 문서]
"""

prompt = f"""Summarize this document:

{long_document}

Summary:"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to("cuda")

# 추론 (Flash Attention 덕분에 가능!)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

Standard attention이면 OOM! Flash Attention은 가능!

---

## 다른 최적화 기법과 비교

| 기법 | 메모리 | 속도 | 정확도 |
|------|--------|------|--------|
| Standard Attention | 1x | 1x | 100% |
| Flash Attention | 0.1x | 2-4x | 100% |
| Sparse Attention | 0.3x | 1.5x | 98% |
| Linear Attention | 0.05x | 3x | 90% |
| Flash + Sparse | 0.05x | 5x | 98% |

**Flash Attention이 최고**: 속도, 메모리, 정확도 모두 우수!

---

## 미래: Flash Attention 3

연구 중인 방향:

**1. Asymmetric Attention**
- Query와 Key/Value를 다른 정밀도로

**2. Hierarchical Attention**
- 긴 시퀀스를 계층적으로

**3. Hardware-aware 튜닝**
- H100, Hopper 아키텍처 최적화

---

## 요약

**Flash Attention**은:

1. **O(N²) → O(N)** 메모리 복잡도
2. **Tiling + Recomputation** 전략
3. **Online Softmax** 스트리밍 계산
4. **SRAM 활용** (HBM 회피)

**결과:**
- 메모리: **8-10배 절약**
- 속도: **2-4배 향상**
- 긴 시퀀스: **가능** (4K → 64K)

**사용처:**
- 모든 Transformer 모델
- 긴 문서 처리
- 고해상도 이미지 (Vision Transformer)
- 메모리 제약 환경

**핵심**: 알고리즘 + 하드웨어 이해 = 극적 성능 향상!

---

## 다음 글

**8편: Speculative Decoding**
- 추론 속도 2-3배 향상
- 작은 모델로 큰 모델 가속
- 무손실 최적화

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
