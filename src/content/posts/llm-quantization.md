---
title: "LLM Inference 최적화 #7: Model Quantization - 메모리를 4배 줄이고 속도는 2배 올리기"
description: "INT8/INT4 양자화로 대규모 언어 모델을 효율적으로 실행하는 GPTQ, AWQ, QLoRA의 원리와 실전을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "quantization", "gptq", "awq", "qlora"]
draft: false
---

# LLM Inference 최적화 #7: Model Quantization

LLaMA-70B는 **140GB**의 메모리가 필요합니다 (FP16). A100 80GB로도 모자랍니다.

**Quantization**은 이를 해결합니다:
- **INT8**: 70GB (2배 절감)
- **INT4**: 35GB (4배 절감)
- **속도**: 2-3배 향상
- **정확도**: 1-2% 손실

RTX 4090 24GB로도 70B 모델을 돌릴 수 있습니다!

---

## 양자화란?

### 기본 개념

**Float16 (16-bit):**
```
범위: -65,504 ~ 65,504
정밀도: ~3 decimal places
메모리: 2 bytes
```

**INT8 (8-bit):**
```
범위: -128 ~ 127
정밀도: integer only
메모리: 1 byte (50% 절감)
```

**INT4 (4-bit):**
```
범위: -8 ~ 7
정밀도: integer only
메모리: 0.5 byte (75% 절감)
```

### 변환 과정

**Quantization:**
```python
# FP16 → INT8
def quantize(weight_fp16, scale):
    return round(weight_fp16 / scale).clamp(-128, 127)

# Example
weight = 0.523  # FP16
scale = 0.01
quantized = round(0.523 / 0.01) = 52  # INT8
```

**Dequantization (복원):**
```python
def dequantize(weight_int8, scale):
    return weight_int8 * scale

# Example
dequantized = 52 * 0.01 = 0.52  # ~0.523 (약간 손실)
```

---

## Symmetric vs Asymmetric

### Symmetric Quantization

**범위가 대칭:**
```
FP16: [-1.0, 1.0]
INT8: [-128, 127]

scale = max(abs(W)) / 127
Q(w) = round(w / scale)
```

**장점:** 간단, 빠름  
**단점:** 범위 낭비 (음수/양수 불균형 시)

### Asymmetric Quantization

**Zero-point 추가:**
```
FP16: [0.2, 1.8]  ← 음수 없음
INT8: [-128, 127]

scale = (max - min) / 255
zero_point = -round(min / scale)

Q(w) = round(w / scale) + zero_point
```

**장점:** 범위 최대 활용  
**단점:** 계산 복잡

---

## Post-Training Quantization (PTQ)

학습 없이 양자화합니다.

### 1. Naive Quantization

가장 간단한 방법:

```python
import torch

def naive_quantize(model):
    """모든 가중치를 INT8로"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Scale 계산
            scale = param.abs().max() / 127
            
            # 양자화
            quantized = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
            
            # 저장
            setattr(model, f'{name}_scale', scale)
            setattr(model, f'{name}_quantized', quantized)
    
    return model

# 추론 시
def forward_quantized(x, weight_quantized, scale):
    # Dequantize
    weight = weight_quantized.float() * scale
    
    # 계산
    return x @ weight.T
```

**문제:** 정확도 크게 하락 (5-10%)

### 2. Calibration-based

대표 데이터로 통계 수집:

```python
def calibrate_quantization(model, calibration_data):
    """데이터로 최적 scale 찾기"""
    activations = {}
    
    # Forward pass로 activation 수집
    def hook(module, input, output):
        activations[module] = output.detach()
    
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook))
    
    # Calibration
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Scale 계산 (percentile 사용)
    scales = {}
    for module, act in activations.items():
        # 99.9 percentile로 outlier 제거
        scale = torch.quantile(act.abs(), 0.999) / 127
        scales[module] = scale
    
    return scales
```

---

## Advanced Quantization Methods

### 1. GPTQ (GPT-Quantization)

**핵심 아이디어:** Layer-wise quantization with Hessian

```python
# GPTQ 알고리즘 (simplified)
def gptq_quantize(model, calibration_data):
    for layer in model.layers:
        # 1. Hessian 계산 (2nd order)
        H = compute_hessian(layer, calibration_data)
        
        # 2. Quantization error 최소화
        for i in range(layer.weight.shape[0]):
            # Optimal rounding
            w = layer.weight[i]
            q = round(w / scale)
            error = w - q * scale
            
            # Error를 다른 weight에 분산
            layer.weight[i+1:] -= H_inv @ error
```

**특징:**
- **정확도:** 매우 높음 (< 1% 손실)
- **속도:** 느림 (Hessian 계산)
- **메모리:** INT4까지 가능

**사용:**
```python
from transformers import AutoModelForCausalLM, GPTQConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=GPTQConfig(bits=4, dataset="c4", group_size=128)
)
```

### 2. AWQ (Activation-aware Weight Quantization)

**핵심 아이디어:** 중요한 가중치는 정밀하게

```python
def awq_quantize(weights, activations):
    # 1. Importance 계산
    importance = compute_importance(weights, activations)
    
    # 2. Top-k는 높은 정밀도
    top_k_idx = importance.topk(k=int(0.01 * len(importance))).indices
    
    # 3. 나머지만 양자화
    quantized = weights.clone()
    for i in range(len(weights)):
        if i not in top_k_idx:
            quantized[i] = quantize(weights[i], scale)
    
    return quantized
```

**Scaling factor:**
```python
# Channel-wise scaling
s = (weights.abs().max(dim=0) / activations.abs().max(dim=0)) ** 0.5

scaled_weights = weights / s
scaled_activations = activations * s

# 이제 양자화
quantized_weights = quantize(scaled_weights)
```

**특징:**
- **정확도:** GPTQ와 비슷
- **속도:** 매우 빠름 (no Hessian)
- **추론:** 빠름 (scale만)

**사용:**
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)
model.quantize(tokenizer, quant_config={"bits": 4, "group_size": 128})
model.save_quantized("llama2-7b-awq")
```

### 3. QLoRA (Quantized LoRA)

**핵심 아이디어:** 4-bit base + FP16 LoRA adapters

```python
# Base model: 4-bit
base_model = load_in_4bit(model_path)

# LoRA: FP16
lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float16))
lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float16))

# Forward
def forward(x):
    # Base (4-bit, frozen)
    base_out = base_model(x)  # Dequantize on-the-fly
    
    # LoRA (FP16, trainable)
    lora_out = (x @ lora_A.T) @ lora_B.T
    
    return base_out + lora_out
```

**특징:**
- **메모리:** 극도로 적음 (70B도 24GB에서 학습!)
- **정확도:** Full fine-tuning과 동일
- **속도:** LoRA 덕분에 빠름

**사용:**
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config
)

# LoRA 적용
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 학습
trainer.train()
```

---

## Quantization Granularity

### Per-tensor

전체 텐서에 하나의 scale:

```python
scale = weight.abs().max() / 127
quantized = round(weight / scale)
```

**장점:** 간단  
**단점:** Outlier에 민감

### Per-channel

채널(행)마다 다른 scale:

```python
# weight: [out_channels, in_channels]
scales = weight.abs().max(dim=1, keepdim=True) / 127
quantized = torch.round(weight / scales).clamp(-128, 127)
```

**장점:** 정확도 ↑  
**단점:** Scale 저장 공간 ↑

### Group-wise

채널을 그룹으로:

```python
group_size = 128
num_groups = in_channels // group_size

for g in range(num_groups):
    start = g * group_size
    end = start + group_size
    group = weight[:, start:end]
    
    scale = group.abs().max() / 127
    quantized[:, start:end] = round(group / scale)
```

**장점:** 정확도 + 효율 밸런스  
**단점:** 복잡

---

## INT8 vs INT4

### INT8

**정확도:**
- Perplexity: < 1% 증가
- 거의 무손실

**속도:**
- 1.5-2x 빠름

**메모리:**
- 2배 절감

**지원:**
- 거의 모든 하드웨어

### INT4

**정확도:**
- Perplexity: 1-2% 증가
- GPTQ/AWQ 사용 시 < 1%

**속도:**
- 2-3x 빠름

**메모리:**
- 4배 절감

**지원:**
- 최신 GPU (Ampere+)

---

## 실전 구현

### 1. bitsandbytes (QLoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4"  # Normal Float 4
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# 추론
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
```

### 2. AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False  # Activation quantization
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config
)

# Calibration
model.quantize(calibration_dataset)

# 저장
model.save_quantized("llama2-7b-gptq-4bit")

# 로드
model = AutoGPTQForCausalLM.from_quantized(
    "llama2-7b-gptq-4bit",
    device="cuda:0",
    use_safetensors=True
)
```

### 3. AutoAWQ

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Quantize
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
)

# 저장
model.save_quantized("llama2-7b-awq")
model.push_to_hub("your-name/llama2-7b-awq")

# 로드 & 추론
model = AutoAWQForCausalLM.from_quantized("llama2-7b-awq", fuse_layers=True)
```

---

## Mixed Precision

전략적으로 정밀도 조합:

```python
# Sensitive layers: FP16
# Other layers: INT4

quantization_config = {
    "layers.0-10": "fp16",      # 초반 레이어
    "layers.11-20": "int8",     # 중간
    "layers.21-31": "int4",     # 후반
    "lm_head": "fp16"           # 출력 레이어
}
```

**예시: LLaMA-70B**
```python
# Embedding: FP16 (중요)
# Attention: INT4 (대부분)
# MLP: INT4
# Layer Norm: FP16 (작음)
# Output: FP16 (중요)

total_memory = (
    0.5 GB  # Embeddings (FP16)
    + 34 GB  # Attention+MLP (INT4)
    + 0.1 GB  # LayerNorm (FP16)
    + 0.3 GB  # Output (FP16)
    = 35 GB  # RTX 4090으로 가능!
)
```

---

## 벤치마크

### 메모리 사용량

**LLaMA-70B:**

| 정밀도 | 메모리 (GB) | GPU |
|--------|------------|-----|
| FP32 | 280 | N/A |
| FP16 | 140 | 2x A100 80GB |
| INT8 | 70 | A100 80GB |
| INT4 (GPTQ) | 35 | A100 40GB, RTX 4090 |
| INT4 (QLoRA) | 35 | RTX 4090 |

### 속도

**LLaMA-7B, A100:**

| 정밀도 | Tokens/sec | Speedup |
|--------|-----------|---------|
| FP16 | 42 | 1x |
| INT8 | 68 | 1.6x |
| INT4 (GPTQ) | 94 | 2.2x |
| INT4 (AWQ) | 103 | 2.5x |

### 정확도

**LLaMA-7B, WikiText-2 Perplexity:**

| 정밀도 | Perplexity | Delta |
|--------|-----------|-------|
| FP16 | 5.68 | 0% |
| INT8 (Calibration) | 5.71 | +0.5% |
| INT4 (Naive) | 7.82 | +37.7% ❌ |
| INT4 (GPTQ) | 5.74 | +1.1% ✅ |
| INT4 (AWQ) | 5.72 | +0.7% ✅ |

**GPTQ/AWQ는 정확도 유지!**

---

## 고급 기법

### 1. SmoothQuant

Activation + Weight 동시 양자화:

```python
# Activation outlier 문제
activations = [0.1, 0.2, 0.15, 12.5]  # Outlier!

# Scale 조정
s = sqrt(max(abs(W)) / max(abs(X)))
W' = W / s
X' = X * s

# 이제 둘 다 양자화 가능
W_quant = quantize(W')
X_quant = quantize(X')
```

### 2. LLM.int8()

**Mixed INT8/FP16:**

```python
# Outlier feature (< 0.1%)는 FP16 유지
def llm_int8_forward(x, weight):
    # Outlier 감지
    outlier_idx = (x.abs() > threshold).any(dim=0)
    
    # 분리
    x_outlier = x[:, outlier_idx]
    x_normal = x[:, ~outlier_idx]
    
    w_outlier = weight[:, outlier_idx]
    w_normal = weight[:, ~outlier_idx]
    
    # 계산
    out_outlier = x_outlier @ w_outlier.T  # FP16
    out_normal = quantized_matmul(x_normal, quantize(w_normal))  # INT8
    
    return out_outlier + out_normal
```

### 3. GGUF/GGML (llama.cpp)

**CPU에 최적화:**

```bash
# 다양한 quantization 지원
Q4_0: 4-bit, fastest
Q4_K_M: 4-bit, mixed
Q5_K_M: 5-bit, balanced
Q8_0: 8-bit, best quality

# 사용
./llama.cpp/main \
  -m llama-2-7b-Q4_K_M.gguf \
  -p "Once upon a time" \
  -n 100
```

---

## 실전 예제: QLoRA Fine-tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 4-bit로 모델 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# 2. 모델 준비
model = prepare_model_for_kbit_training(model)

# 3. LoRA 설정
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4. 학습
training_args = TrainingArguments(
    output_dir="./llama2-70b-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_32bit",  # QLoRA optimizer
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512
)

trainer.train()

# 5. 저장 (LoRA adapter만, 몇 MB)
model.save_pretrained("./llama2-70b-qlora-adapter")
```

**메모리:** 24GB (RTX 4090으로 70B 학습!)

---

## Best Practices

### 1. 어떤 방법 선택?

**추론만:**
- **빠른 양자화**: AWQ
- **최고 정확도**: GPTQ
- **CPU 추론**: GGUF

**Fine-tuning:**
- **대규모 모델**: QLoRA
- **정확도 중요**: LoRA (no quantization)

### 2. Calibration Dataset

```python
# Good: Domain-specific
calibration_data = load_dataset("your_domain")

# Better: Diverse
calibration_data = load_dataset("c4", split="train[:1000]")

# Best: Task-relevant
calibration_data = your_training_data[:1000]
```

### 3. Validation

```python
# 항상 검증!
def validate_quantized_model(original, quantized, test_data):
    orig_ppl = compute_perplexity(original, test_data)
    quant_ppl = compute_perplexity(quantized, test_data)
    
    degradation = (quant_ppl - orig_ppl) / orig_ppl * 100
    print(f"Perplexity degradation: {degradation:.2f}%")
    
    if degradation > 5:
        print("⚠️ Too much quality loss!")
    else:
        print("✅ Acceptable quality")
```

---

## 요약

**Quantization**은:

1. **메모리**: 2-4배 절감 (INT8/INT4)
2. **속도**: 1.5-3배 향상
3. **정확도**: 1-2% 손실 (GPTQ/AWQ)
4. **접근성**: 소형 GPU로 대형 모델 실행

**방법 비교:**

| 방법 | 정확도 | 속도 | 메모리 | 사용처 |
|------|--------|------|--------|--------|
| GPTQ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 추론 |
| AWQ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 추론 (추천!) |
| QLoRA | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fine-tuning |
| GGUF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | CPU 추론 |

**핵심:** 모든 프로덕션 LLM에 필수!

---

## 시리즈 완료! 🎉

**LLM Inference 최적화 완전 정복:**

1. **Paged Attention**: 메모리 10배 절약
2. **KV Caching**: 속도 50-100배
3. **LoRA**: Fine-tuning 10배 효율
4. **Flash Attention**: 메모리 + 속도 모두
5. **Speculative Decoding**: 2-3배 가속
6. **Continuous Batching**: 처리량 극대화
7. **Quantization**: 메모리 4배, 속도 2배

**조합하면?**
- vLLM (Paged + Continuous + Flash): **10-20배 처리량**
- QLoRA (Quantization + LoRA): **24GB로 70B 학습**
- AWQ + Flash + Speculative: **50-100배 빠른 추론**

이제 여러분도 효율적인 LLM을 만들 수 있습니다! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*

*전체 시리즈가 도움이 되셨다면 ⭐ Star 부탁드립니다!*
