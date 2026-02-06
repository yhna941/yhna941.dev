---
title: "LLM Inference 최적화 #8: Tensor Parallelism - 거대 모델을 여러 GPU에 나누기"
description: "하나의 레이어를 여러 GPU에 분산시키는 Tensor Parallelism으로 메모리 한계를 극복하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "parallelism", "distributed", "multi-gpu"]
draft: false
---

# LLM Inference 최적화 #8: Tensor Parallelism

GPT-3 175B는 **350GB** 메모리가 필요합니다 (FP16). 단일 GPU로는 불가능하죠.

**Tensor Parallelism**은 이를 해결합니다:
- **하나의 레이어를 여러 GPU에 분산**
- 각 GPU가 **일부만 계산**
- 통신으로 결과 결합
- **메모리: N개 GPU로 1/N**

결과:
- 175B 모델 → 8개 A100으로 실행
- 통신 오버헤드 최소화
- Linear scaling (이론적)

---

## 문제: 모델이 GPU에 안 들어감

### 메모리 계산

**GPT-3 175B (FP16):**
```
Parameters: 175B × 2 bytes = 350 GB
+ Activations: ~50 GB
+ KV Cache (batch 32): ~100 GB
= 500 GB 총 필요
```

**A100 80GB로는 불가능!**

### 기존 해결책들

**1. Data Parallelism (안 됨)**
```
각 GPU가 전체 모델 복사 → 메모리 동일
```

**2. Model Parallelism (순차)**
```
GPU 1: Layer 1-10
GPU 2: Layer 11-20
GPU 3: Layer 21-30
...

문제: GPU 활용률 낮음 (순차 실행)
```

---

## Tensor Parallelism

### 핵심 아이디어

> **하나의 행렬 연산을 여러 GPU에 나눈다**

```
Standard:
Y = XW    # W: [4096, 4096]

Tensor Parallel (2 GPUs):
W = [W1 | W2]  # Split column-wise

GPU 0: Y1 = X @ W1  # [batch, 2048]
GPU 1: Y2 = X @ W2  # [batch, 2048]

Y = [Y1 | Y2]  # Concat
```

### 수식

**Linear layer:**
```
Y = XW + b
  where W: [d_in, d_out]
```

**Split W column-wise (n GPUs):**
```
W = [W_1, W_2, ..., W_n]
  where W_i: [d_in, d_out/n]

GPU i: Y_i = XW_i
All: Y = [Y_1, Y_2, ..., Y_n]
```

**메모리:** 각 GPU는 `d_out/n`만 저장!

---

## Transformer에 적용

### 1. Self-Attention

**MLP (Feed-Forward):**
```
h = activation(X @ W1)
Y = h @ W2

Standard:
  W1: [d_model, 4*d_model]
  W2: [4*d_model, d_model]

Tensor Parallel (column split):
  W1 = [W1_1, W1_2, ..., W1_n]
  W2 = [W2_1; W2_2; ...; W2_n]  (row split!)
```

**구현:**
```python
class ParallelMLP(nn.Module):
    def __init__(self, d_model, d_ff, world_size):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # W1: Column parallel
        self.fc1 = ColumnParallelLinear(
            d_model,
            d_ff // world_size,
            gather_output=False  # Keep split
        )
        
        # W2: Row parallel
        self.fc2 = RowParallelLinear(
            d_ff // world_size,
            d_model,
            input_is_parallel=True  # Already split
        )
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        
        # W1 (column parallel)
        h = self.fc1(x)  # [batch, seq, d_ff/world_size]
        h = F.gelu(h)
        
        # W2 (row parallel)
        y = self.fc2(h)  # [batch, seq, d_model]
        
        return y
```

### 2. Multi-Head Attention

**Q, K, V를 head 단위로 분산:**

```python
# Standard
num_heads = 32
head_dim = 128

# Tensor Parallel (4 GPUs)
num_heads_per_gpu = 32 // 4 = 8

class ParallelAttention(nn.Module):
    def __init__(self, d_model, num_heads, world_size):
        self.num_heads = num_heads // world_size
        self.head_dim = d_model // num_heads
        
        # Q, K, V: Column parallel
        self.qkv = ColumnParallelLinear(
            d_model,
            3 * d_model // world_size,
            gather_output=False
        )
        
        # Output: Row parallel
        self.out = RowParallelLinear(
            d_model // world_size,
            d_model,
            input_is_parallel=True
        )
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        
        # QKV projection (각 GPU는 8 heads만)
        qkv = self.qkv(x)  # [batch, seq, 3*d_model/world_size]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = k.view(batch, seq, self.num_heads, self.head_dim)
        v = v.view(batch, seq, self.num_heads, self.head_dim)
        
        # Attention (local to each GPU)
        attn_out = self.attention(q, k, v)
        
        # Output projection (all-reduce)
        out = self.out(attn_out)  # [batch, seq, d_model]
        
        return out
```

---

## 통신 패턴

### Column Parallel

```python
class ColumnParallelLinear(nn.Module):
    """
    Y = XW
    W를 column-wise로 분할
    
    Input: X (replicated)
    Output: Y_i (split)
    Communication: None
    """
    def forward(self, x):
        # x: [batch, in_features] on all GPUs
        
        # Local matmul
        output = F.linear(x, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather across GPUs
            output = gather_from_model_parallel_region(output)
        
        return output
```

### Row Parallel

```python
class RowParallelLinear(nn.Module):
    """
    Y = XW
    W를 row-wise로 분할
    
    Input: X_i (split)
    Output: Y (replicated)
    Communication: All-reduce
    """
    def forward(self, x):
        # x: [batch, in_features/world_size] on each GPU
        
        # Local matmul
        output_parallel = F.linear(x, self.weight, self.bias)
        
        # All-reduce across GPUs
        output = reduce_from_model_parallel_region(output_parallel)
        
        return output
```

### 통신 연산

```python
def all_gather(tensor, dim=0):
    """모든 GPU에서 수집"""
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=dim)

def all_reduce(tensor):
    """모든 GPU의 결과 합산"""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def scatter(tensor, dim=0):
    """각 GPU에 일부 분배"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    chunk_size = tensor.size(dim) // world_size
    return tensor.narrow(dim, rank * chunk_size, chunk_size)
```

---

## 실전 구현

### 1. Megatron-LM 스타일

```python
import torch
import torch.distributed as dist

class TransformerLayer(nn.Module):
    def __init__(self, config, world_size):
        super().__init__()
        self.world_size = world_size
        
        # Attention
        self.attention = ParallelAttention(
            config.hidden_size,
            config.num_attention_heads,
            world_size
        )
        
        # MLP
        self.mlp = ParallelMLP(
            config.hidden_size,
            config.intermediate_size,
            world_size
        )
        
        # LayerNorm (replicated)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        # Attention (with residual)
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = x + residual
        
        # MLP (with residual)
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


# 초기화
def initialize_model_parallel(world_size):
    """Model parallel group 생성"""
    rank = dist.get_rank()
    
    # Model parallel group
    model_parallel_group = dist.new_group(
        ranks=list(range(world_size))
    )
    
    return model_parallel_group


# 사용
if __name__ == "__main__":
    # 분산 초기화
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 모델 생성
    model = TransformerLayer(config, world_size).cuda()
    
    # 추론
    x = torch.randn(32, 512, 768).cuda()  # Replicated input
    y = model(x)
    
    print(f"GPU {rank}: Output shape {y.shape}")
```

### 2. HuggingFace Accelerate

```python
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 1. 빈 모델 생성 (메모리 안 씀)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

# 2. Tensor parallel 설정
device_map = {
    "transformer.h.0": 0,
    "transformer.h.1": 1,
    "transformer.h.2": 2,
    # ...
}

# 3. 체크포인트 로드 & 분산
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="gpt2-xl",
    device_map="auto",  # 자동 분산
    offload_folder="offload"
)

# 추론
outputs = model.generate(inputs, max_new_tokens=100)
```

### 3. DeepSpeed Inference

```python
import deepspeed

# 모델 생성
model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b")

# DeepSpeed로 wrapping
ds_engine = deepspeed.init_inference(
    model,
    mp_size=4,  # Tensor parallel size
    dtype=torch.float16,
    replace_with_kernel_inject=True,  # 최적화된 커널
    replace_method="auto"
)

# 추론
outputs = ds_engine.generate(inputs, max_new_tokens=100)
```

---

## 통신 최적화

### 1. Overlapping Computation & Communication

```python
class OverlappedLinear(nn.Module):
    """통신과 계산을 동시에"""
    def forward(self, x):
        # 1. 계산 시작
        local_output = F.linear(x, self.weight)
        
        # 2. 비동기 All-reduce (백그라운드)
        handle = dist.all_reduce(local_output, async_op=True)
        
        # 3. 다른 작업 (예: bias 추가)
        if self.bias is not None:
            local_output = local_output + self.bias
        
        # 4. 통신 완료 대기
        handle.wait()
        
        return local_output
```

### 2. Gradient Accumulation

```python
# 작은 배치를 여러 번 (통신 줄이기)
for micro_batch in split_batch(batch, num_micro_batches):
    # Forward
    loss = model(micro_batch)
    
    # Backward (gradient accumulate)
    loss.backward()
    
    # No optimizer step yet!

# 모든 micro-batch 끝난 후 한 번에
optimizer.step()
optimizer.zero_grad()
```

### 3. Sequence Parallelism

긴 시퀀스도 분산:

```python
# Standard: 모든 GPU가 전체 시퀀스 처리
x: [batch, seq_len, hidden]

# Sequence Parallel: 시퀀스를 나눔
x_local: [batch, seq_len/world_size, hidden]

# LayerNorm, Dropout도 split
ln_output = layer_norm(x_local)  # Local
dropout_output = dropout(ln_output)  # Local

# Attention은 all-gather 필요
x_full = all_gather(x_local, dim=1)
attn_output = attention(x_full)
attn_output_local = scatter(attn_output, dim=1)
```

---

## 메모리 & 통신 분석

### 메모리 절감

**175B 모델, 4-way TP:**
```
Standard (1 GPU):
  Parameters: 350 GB
  Activations: 50 GB
  Total: 400 GB (불가능!)

Tensor Parallel (4 GPUs):
  Parameters per GPU: 350/4 = 87.5 GB
  Activations per GPU: 50/4 = 12.5 GB
  Total per GPU: 100 GB (A100 가능!)
```

### 통신 비용

**Per layer:**
```
MLP:
  Column parallel: No communication (forward)
  Row parallel: All-reduce (4d² elements)

Attention:
  Column parallel (QKV): No communication
  Row parallel (Output): All-reduce (4d² elements)

Total per layer: 2 × All-reduce (8d²)
```

**GPT-3 175B, 96 layers:**
```
d = 12,288
Communication per layer: 8 × 12,288² = 1.2 GB
Total: 96 × 1.2 GB = 115 GB per forward pass

With A100 (600 GB/s NVLink):
  Communication time: 115 / 600 = 0.19s
  Computation time: ~2s
  Overhead: 9.5% (acceptable!)
```

---

## Tensor Parallelism vs 다른 방식

| 방식 | 메모리/GPU | 통신량 | GPU 활용률 | 구현 난이도 |
|------|-----------|--------|-----------|------------|
| Data Parallel | 100% | Gradient only | 100% | 쉬움 |
| Pipeline Parallel | 1/N | Activation | ~50% | 중간 |
| **Tensor Parallel** | 1/N | Per layer | 90%+ | 어려움 |
| Hybrid (TP+PP) | 1/(N×M) | 둘 다 | 80%+ | 매우 어려움 |

---

## 실전 벤치마크

### GPT-3 175B, 8× A100 80GB

| 방식 | 처리량 (tokens/s) | 메모리/GPU |
|------|------------------|-----------|
| Impossible (1 GPU) | N/A | 350 GB |
| TP=8 | 1,240 | 48 GB |
| TP=4 + PP=2 | 1,850 | 52 GB |
| TP=8 + Zero | 2,100 | 45 GB |

### 통신 스케일링

**A100, NVLink:**

| TP Size | 이론 효율 | 실제 효율 | 통신 오버헤드 |
|---------|----------|----------|--------------|
| 2 | 100% | 95% | 5% |
| 4 | 100% | 90% | 10% |
| 8 | 100% | 85% | 15% |

**DGX A100 (NVSwitch):**
- TP=8까지 거의 linear scaling!

---

## Best Practices

### 1. TP Size 선택

```python
# Rule of thumb
if model_size < 20B:
    tp_size = 1  # 필요 없음
elif model_size < 70B:
    tp_size = 2  # 적당
elif model_size < 200B:
    tp_size = 4  # 권장
else:
    tp_size = 8  # 최대
```

### 2. 하이브리드 병렬화

```python
# 예: 175B 모델, 16 GPUs
config = {
    "tensor_parallel": 4,   # 레이어 내 분산
    "pipeline_parallel": 2,  # 레이어 간 분산
    "data_parallel": 2       # 배치 분산
}

# 총 GPU: 4 × 2 × 2 = 16
```

### 3. 네트워크 최적화

```python
# NVLink가 있으면
if has_nvlink():
    tensor_parallel_size = 8  # Aggressive
else:
    tensor_parallel_size = 2  # Conservative
```

---

## 코드 예제: 처음부터 구현

```python
import torch
import torch.distributed as dist
import torch.nn as nn

class TensorParallelLinear(nn.Module):
    """
    완전한 Tensor Parallel Linear 구현
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        gather_output=True,
        input_is_parallel=False
    ):
        super().__init__()
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Output을 world_size로 나눔
        self.output_size_per_partition = out_features // self.world_size
        
        # Weight (각 GPU는 일부만 소유)
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            in_features
        ))
        
        # Bias (옵션)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
        
        # 초기화
        self._initialize_weights()
        
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: [batch, seq, in_features] or [batch, seq, in_features/world_size]
        
        # Input이 이미 split되어 있으면 all-gather
        if self.input_is_parallel:
            # Row parallel: input은 split, output은 reduce
            input_parallel = x
        else:
            # Column parallel: input은 replicate, output은 split
            input_parallel = x
        
        # Local matmul
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        # Gather or reduce
        if self.gather_output:
            # All-gather across model parallel group
            output = self._gather(output_parallel)
        elif self.input_is_parallel:
            # All-reduce (row parallel)
            output = self._reduce(output_parallel)
        else:
            # Keep split (column parallel)
            output = output_parallel
        
        return output
    
    def _gather(self, tensor):
        """All-gather operation"""
        world_size = self.world_size
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=-1)
    
    def _reduce(self, tensor):
        """All-reduce operation"""
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


# 사용 예제
def example_usage():
    # 분산 초기화
    dist.init_process_group(backend='nccl')
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 모델 생성
    batch_size = 32
    seq_len = 512
    d_model = 1024
    
    # Input (replicated across all GPUs)
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    # Column parallel layer
    fc1 = TensorParallelLinear(
        d_model, 
        4 * d_model,
        gather_output=False  # Keep split for next layer
    ).cuda()
    
    # Row parallel layer
    fc2 = TensorParallelLinear(
        4 * d_model,
        d_model,
        input_is_parallel=True,  # Input is already split
        gather_output=True       # Final output needs to be replicated
    ).cuda()
    
    # Forward
    h = fc1(x)  # [batch, seq, 4*d_model/world_size]
    h = F.gelu(h)
    y = fc2(h)  # [batch, seq, d_model]
    
    print(f"GPU {rank}: x.shape={x.shape}, h.shape={h.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    example_usage()
```

---

## 요약

**Tensor Parallelism**은:

1. **하나의 레이어를 여러 GPU에 분산**
2. **Column/Row parallel** 전략
3. **메모리: 1/N 절감**
4. **통신: All-reduce (per layer)**
5. **GPU 활용률: 90%+**

**핵심 포인트:**
- Large model에 필수 (70B+)
- NVLink 필요 (통신 병목)
- Pipeline/Data Parallel과 조합
- Megatron-LM, DeepSpeed 사용 권장

**다음 단계:**
- Pipeline Parallelism 조합
- ZeRO optimizer
- 3D Parallelism

---

## 다음 글

**12편: Pipeline Parallelism**
- 레이어를 GPU에 분배
- Micro-batching
- Bubble 최소화

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
