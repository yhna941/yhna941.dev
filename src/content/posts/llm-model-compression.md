---
title: "LLM Inference 최적화 #10: Model Compression - 작지만 강력하게"
description: "Pruning, Distillation, Low-rank decomposition으로 모델 크기를 줄이면서 성능을 유지하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "compression", "pruning", "distillation"]
draft: false
---

# LLM Inference 최적화 #10: Model Compression

LLaMA-7B는 13GB입니다. 하지만 **실제로 필요한 파라미터는 얼마나 될까요?**

연구에 따르면:
- **20-30%는 pruning 가능** (정확도 거의 유지)
- **Student 모델 (1/4 크기)**이 teacher의 95% 성능
- **Low-rank로 50% 파라미터** 절감

**Model Compression**은 이를 실현합니다.

---

## 압축 방법 개요

| 방법 | 압축률 | 정확도 | 추론 속도 | 난이도 |
|------|--------|--------|----------|--------|
| Quantization | 75% | 98-99% | 2-3x | 쉬움 |
| **Pruning** | 50-70% | 95-98% | 1.5-2x | 중간 |
| **Distillation** | 75% | 90-95% | 4x | 어려움 |
| Low-rank | 40-60% | 98% | 1.2x | 중간 |

**이번 글:** Pruning + Distillation + Low-rank

---

## Pruning (가지치기)

### 핵심 아이디어

> **중요하지 않은 가중치를 0으로 만든다**

```python
# 예: 작은 가중치 제거
threshold = 0.01
mask = torch.abs(weight) > threshold
pruned_weight = weight * mask

# 50%가 0이 됨!
```

### Pruning 종류

**1. Unstructured Pruning**
```python
# 개별 가중치 단위
weight: [4096, 4096]

# 50% pruning
pruned: [4096, 4096] with 50% zeros

# 저장: Sparse matrix (COO format)
indices: [[row1, col1], [row2, col2], ...]
values: [val1, val2, ...]
```

**장점:** 높은 압축률  
**단점:** 하드웨어 가속 어려움

**2. Structured Pruning**
```python
# 행/열/채널 단위
weight: [4096, 4096]

# 채널 50% pruning
pruned: [4096, 2048]  # Dense!
```

**장점:** 하드웨어 친화적  
**단점:** 압축률 낮음

**3. N:M Sparsity**
```python
# N개 중 M개만 유지
# 2:4 = 4개 중 2개만 유지

weight = [0.1, 0.5, 0.2, 0.8]
# → [0, 0.5, 0, 0.8]  (Top-2 유지)

# NVIDIA A100 하드웨어 지원!
```

**장점:** 하드웨어 가속 + 압축  
**단점:** 제약적

---

## Magnitude-based Pruning

### 알고리즘

```python
def magnitude_pruning(model, sparsity=0.5):
    """
    작은 가중치부터 제거
    
    Args:
        sparsity: 제거할 비율 (0.5 = 50%)
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 가중치 크기 계산
            importance = torch.abs(param.data)
            
            # Threshold 계산 (하위 50%)
            threshold = torch.quantile(importance, sparsity)
            
            # Mask 생성
            mask = importance > threshold
            
            # Pruning
            param.data *= mask
            
            # Mask 저장 (gradient 계산 시 필요)
            param.register_buffer('mask', mask)
```

### Iterative Pruning

한 번에 크게 pruning하면 정확도 급락. 점진적으로:

```python
def iterative_pruning(model, dataset, target_sparsity=0.9, steps=10):
    """
    점진적 pruning + fine-tuning
    """
    current_sparsity = 0.0
    step_size = target_sparsity / steps
    
    for step in range(steps):
        # 1. Prune
        current_sparsity += step_size
        magnitude_pruning(model, sparsity=current_sparsity)
        
        # 2. Fine-tune
        fine_tune(model, dataset, epochs=1)
        
        # 3. Evaluate
        acc = evaluate(model, val_dataset)
        print(f"Step {step}: Sparsity={current_sparsity:.1%}, Acc={acc:.2%}")
    
    return model
```

---

## Structured Pruning

### Channel Pruning

채널 단위로 제거:

```python
def channel_pruning(layer, num_channels_to_prune):
    """
    CNN/Transformer에서 채널 pruning
    
    Args:
        layer: Conv or Linear layer
        num_channels_to_prune: 제거할 채널 수
    """
    # Channel importance 계산
    weight = layer.weight.data  # [out_channels, in_channels, ...]
    channel_norms = torch.norm(weight, dim=(1, 2, 3))  # L2 norm per channel
    
    # 중요도 낮은 채널 선택
    _, indices = torch.sort(channel_norms)
    prune_indices = indices[:num_channels_to_prune]
    
    # 새로운 가중치 생성 (채널 제거)
    keep_mask = torch.ones(weight.size(0), dtype=torch.bool)
    keep_mask[prune_indices] = False
    
    new_weight = weight[keep_mask]
    
    # 레이어 교체
    out_channels = weight.size(0) - num_channels_to_prune
    new_layer = nn.Conv2d(
        layer.in_channels,
        out_channels,
        layer.kernel_size,
        # ... other params
    )
    new_layer.weight.data = new_weight
    
    return new_layer
```

### Head Pruning (Attention)

Attention head 제거:

```python
def prune_attention_heads(attention_layer, num_heads_to_prune):
    """
    Multi-head attention에서 head pruning
    """
    num_heads = attention_layer.num_heads
    head_dim = attention_layer.head_dim
    
    # Head importance 계산 (Taylor approximation)
    head_importance = []
    for h in range(num_heads):
        # Head h의 gradient × activation
        grad = attention_layer.get_head_gradient(h)
        act = attention_layer.get_head_activation(h)
        importance = (grad * act).sum()
        head_importance.append(importance)
    
    # 낮은 importance head 제거
    head_importance = torch.tensor(head_importance)
    _, indices = torch.sort(head_importance)
    prune_indices = indices[:num_heads_to_prune]
    
    # QKV 가중치 재구성
    new_num_heads = num_heads - num_heads_to_prune
    new_qkv_weight = remove_heads_from_qkv(
        attention_layer.qkv.weight,
        prune_indices,
        num_heads,
        head_dim
    )
    
    # 새 레이어 생성
    new_attention = MultiHeadAttention(
        embed_dim=attention_layer.embed_dim,
        num_heads=new_num_heads
    )
    new_attention.qkv.weight.data = new_qkv_weight
    
    return new_attention
```

---

## Knowledge Distillation

### 핵심 아이디어

> **큰 모델(teacher)의 지식을 작은 모델(student)에게 전달**

```
Teacher (70B): "Cat"에 90% 확률
Student (7B): Teacher를 모방하도록 학습
```

### Distillation Loss

```python
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    Hinton's Distillation Loss
    
    Args:
        T: Temperature (높을수록 soft)
        alpha: Teacher loss 가중치
    """
    # 1. Hard loss (정답 레이블)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft loss (teacher 확률 분포)
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    
    soft_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (T ** 2)
    
    # 3. 결합
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss


# 학습
for batch in dataloader:
    inputs, labels = batch
    
    # Teacher prediction (frozen)
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)
    
    # Student prediction
    student_logits = student_model(inputs)
    
    # Loss
    loss = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        T=3.0,
        alpha=0.7
    )
    
    # Backprop
    loss.backward()
    optimizer.step()
```

### Feature Distillation

중간 레이어도 모방:

```python
class FeatureDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Logits
        self.beta = beta    # Features
    
    def forward(self, student_outputs, teacher_outputs, labels):
        # 1. Logits distillation
        logit_loss = distillation_loss(
            student_outputs['logits'],
            teacher_outputs['logits'],
            labels
        )
        
        # 2. Feature distillation (hidden states)
        feature_loss = 0
        for s_feat, t_feat in zip(
            student_outputs['hidden_states'],
            teacher_outputs['hidden_states']
        ):
            # MSE loss between features
            feature_loss += F.mse_loss(s_feat, t_feat)
        
        feature_loss /= len(student_outputs['hidden_states'])
        
        # 3. Total
        return self.alpha * logit_loss + self.beta * feature_loss
```

---

## Distillation 전략

### 1. Standard Distillation

```python
# Teacher: 70B
teacher = AutoModelForCausalLM.from_pretrained("llama-70b")
teacher.eval()

# Student: 7B
student = AutoModelForCausalLM.from_pretrained("llama-7b")

# Distill
for epoch in range(3):
    for batch in dataloader:
        with torch.no_grad():
            teacher_out = teacher(batch)
        
        student_out = student(batch)
        loss = distillation_loss(student_out, teacher_out, batch['labels'])
        
        loss.backward()
        optimizer.step()
```

### 2. On-the-fly Distillation

실시간 생성 데이터로:

```python
def on_the_fly_distillation(teacher, student, prompts):
    """Teacher가 생성한 데이터로 학습"""
    for prompt in prompts:
        # Teacher 생성
        with torch.no_grad():
            teacher_outputs = teacher.generate(
                prompt,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generated_text = teacher_outputs.sequences
        teacher_logits = teacher_outputs.scores
        
        # Student 학습
        student_logits = student(generated_text)
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
        
        loss.backward()
        optimizer.step()
```

### 3. Task-specific Distillation

특정 태스크에 집중:

```python
# 예: Summarization
def distill_for_summarization(teacher, student, dataset):
    for article, summary in dataset:
        # Teacher: 요약 생성
        with torch.no_grad():
            teacher_summary = teacher.generate(article)
            teacher_logits = teacher(article).logits
        
        # Student: Teacher 모방
        student_logits = student(article).logits
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            labels=None  # No ground truth needed!
        )
        
        loss.backward()
        optimizer.step()
```

---

## Low-Rank Decomposition

### Matrix Factorization

큰 행렬을 작은 행렬 곱으로:

```python
# 원본
W: [4096, 4096]  # 16M parameters

# Low-rank decomposition
W ≈ U @ V
U: [4096, 256]   # 1M
V: [256, 4096]   # 1M
# Total: 2M (8배 절감!)
```

### SVD-based Decomposition

```python
def low_rank_decompose(weight, rank):
    """
    SVD로 low-rank 분해
    
    Args:
        weight: [out_features, in_features]
        rank: Target rank
    """
    # SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    
    # Top-r 유지
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    
    # 재구성
    U_scaled = U_r * torch.sqrt(S_r)
    V_scaled = torch.sqrt(S_r).unsqueeze(1) * V_r
    
    return U_scaled, V_scaled


# 모델에 적용
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=True)
    
    def forward(self, x):
        return self.V(self.U(x))


def convert_to_low_rank(model, rank=256):
    """모델의 Linear를 Low-rank로 교체"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_feat = module.in_features
            out_feat = module.out_features
            
            # Decompose
            U, V = low_rank_decompose(module.weight.data, rank)
            
            # 새 레이어
            new_module = LowRankLinear(in_feat, out_feat, rank)
            new_module.U.weight.data = U.T
            new_module.V.weight.data = V
            if module.bias is not None:
                new_module.V.bias.data = module.bias.data
            
            # 교체
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], new_module)
    
    return model
```

---

## 실전 예제

### 1. LLM Pruning (Wanda)

```python
# Wanda: Weight + Activation pruning
def wanda_pruning(model, calibration_data, sparsity=0.5):
    """
    Activation-aware pruning
    """
    # 1. Activation 수집
    activations = collect_activations(model, calibration_data)
    
    # 2. Layer-wise pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            act = activations[name]
            
            # Importance = |weight| × activation magnitude
            importance = torch.abs(weight) * act.mean(dim=0)
            
            # Threshold
            threshold = torch.quantile(importance, sparsity)
            mask = importance > threshold
            
            # Apply mask
            module.weight.data *= mask
    
    return model


# 사용
model = AutoModelForCausalLM.from_pretrained("llama-7b")
calibration_data = load_dataset("c4", split="train[:1000]")

pruned_model = wanda_pruning(model, calibration_data, sparsity=0.5)

# 50% sparse, 정확도 거의 유지!
```

### 2. DistilBERT (실제 사례)

```python
from transformers import DistilBertModel, BertModel

# Teacher: BERT-base (110M)
teacher = BertModel.from_pretrained("bert-base-uncased")

# Student: DistilBERT (66M, 40% 작음)
student = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Distillation
class DistilBERTTrainer:
    def __init__(self, teacher, student):
        self.teacher = teacher.eval()
        self.student = student
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def train_step(self, batch):
        inputs = batch['input_ids']
        labels = batch['labels']
        
        # Teacher (frozen)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs, output_hidden_states=True)
        
        # Student
        student_outputs = self.student(inputs, output_hidden_states=True)
        
        # 1. Hard loss (task-specific)
        hard_loss = self.ce_loss(student_outputs.logits, labels)
        
        # 2. Soft loss (distillation)
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs.logits / 2.0, dim=-1),
            F.softmax(teacher_outputs.logits / 2.0, dim=-1),
            reduction='batchmean'
        ) * 4.0
        
        # 3. Hidden state loss
        hidden_loss = 0
        for s_hidden, t_hidden in zip(
            student_outputs.hidden_states,
            teacher_outputs.hidden_states
        ):
            hidden_loss += self.mse_loss(s_hidden, t_hidden)
        
        # Total
        loss = 0.5 * hard_loss + 0.5 * soft_loss + 0.1 * hidden_loss
        
        return loss

# 결과: 97% BERT 성능, 40% 작음, 60% 빠름!
```

---

## 벤치마크

### Pruning

**LLaMA-7B, WikiText-2:**

| 방법 | Sparsity | Perplexity | ∆ |
|------|----------|-----------|---|
| Dense | 0% | 5.68 | - |
| Magnitude | 50% | 5.95 | +4.8% |
| Wanda | 50% | 5.74 | +1.1% ✅ |
| Wanda | 70% | 6.12 | +7.7% |

### Distillation

**GPT-2 → DistilGPT-2:**

| 모델 | Size | Speed | Accuracy |
|------|------|-------|----------|
| GPT-2 | 117M | 1x | 100% |
| DistilGPT-2 | 82M (70%) | 1.5x | 97% |

### Low-Rank

**LLaMA-7B, rank=256:**

| Layer | Original | Low-rank | Compression |
|-------|----------|----------|-------------|
| Q/K/V | 16M | 3M | 5.3x |
| MLP | 64M | 16M | 4x |
| **Total** | 7B | **4.2B** | **1.7x** |

---

## 조합: Pruning + Quantization + Distillation

최고 압축:

```python
# 1. Distillation (70B → 7B)
student = distill(teacher_70b, target_size=7B)
# 7B, 95% accuracy

# 2. Pruning (7B → 3.5B)
student = prune(student, sparsity=0.5)
# 3.5B, 93% accuracy

# 3. Quantization (3.5B INT8)
student = quantize(student, bits=8)
# 1.75 GB, 93% accuracy

# 원본 70B (140GB) → 1.75GB (80배 압축!)
```

---

## Best Practices

### 1. Pruning 순서

```python
# 1. Global magnitude pruning (quick win)
# 2. Fine-tune
# 3. Layer-wise structured pruning
# 4. Fine-tune again
```

### 2. Distillation Tips

```python
# Temperature 선택
T = 3.0  # 일반적
T = 5.0  # 큰 모델 차이
T = 2.0  # 작은 모델 차이

# Alpha 선택
alpha = 0.7  # Teacher에 집중
alpha = 0.5  # 밸런스
alpha = 0.3  # Hard labels 중시
```

### 3. 검증

```python
# 항상 여러 메트릭 확인
metrics = {
    'perplexity': evaluate_perplexity(model),
    'accuracy': evaluate_accuracy(model, tasks),
    'speed': measure_throughput(model),
    'memory': measure_memory(model)
}
```

---

## 요약

**Model Compression**은:

1. **Pruning**: 가중치 제거 (50-70% 압축)
2. **Distillation**: 작은 모델에 지식 전달 (75% 압축)
3. **Low-rank**: 행렬 분해 (40-60% 압축)

**조합 효과:**
- Pruning + Quantization: **8배 압축**
- Distillation + Quantization: **16배 압축**
- All: **80배 이상** 가능!

**사용처:**
- Edge devices (모바일, IoT)
- 비용 절감 (API 호스팅)
- 레이턴시 중요한 서비스

---

## 다음 글

**14편: CUDA Kernel 최적화**
- Custom CUDA kernel 작성
- Memory coalescing
- Warp-level primitives
- 직접 짜는 고성능 연산

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
