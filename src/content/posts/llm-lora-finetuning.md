---
title: "LLM Inference 최적화 #3: LoRA Fine-tuning 완전 정복"
description: "적은 파라미터로 거대 언어 모델을 효율적으로 학습하는 LoRA의 원리와 실전 구현을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "fine-tuning", "lora", "peft", "qlora"]
draft: false
---

# LLM Inference 최적화 #3: LoRA Fine-tuning

7B 파라미터 모델을 fine-tuning하려면 보통 **14GB 이상**의 GPU 메모리가 필요합니다. Adam optimizer까지 쓰면 **80GB**도 모자랍니다.

**LoRA**는 이 문제를 해결합니다. **0.1%의 파라미터**만 학습해도 full fine-tuning과 비슷한 성능을 냅니다.

어떻게 가능할까요?

---

## 문제: Fine-tuning은 비싸다

### Full Fine-tuning

모든 파라미터를 업데이트합니다.

```
LLaMA-7B: 7,000,000,000 parameters
```

**메모리 요구사항:**
- 모델: 14 GB (fp16)
- Gradients: 14 GB
- Optimizer states (Adam): 56 GB (4x)
- Activations: 10+ GB

**총합: 94 GB+**

A100 80GB로도 부족합니다!

### 기존 해결책들

**1. Adapter Layers**
- 작은 레이어 추가
- 약간의 성능 손실
- 여전히 많은 메모리 필요

**2. Prompt Tuning**
- Soft prompt만 학습
- 성능이 많이 떨어짐
- Task-specific

---

## 해결책: LoRA (Low-Rank Adaptation)

### 핵심 아이디어

> **대부분의 변화는 낮은 rank에서 일어난다**

신경망의 가중치 변화 ΔW는 **low-rank matrix**로 근사할 수 있습니다.

```
W' = W + ΔW

ΔW ≈ BA

where:
  B: [d, r]  (r << d)
  A: [r, k]  (r << k)
  rank(ΔW) = r  (매우 작음)
```

### 수식

**기존 선형 레이어:**
```
h = Wx
```

**LoRA:**
```
h = Wx + BAx
  = Wx + (BA)x
```

**여기서:**
- W: 원본 가중치 (frozen, 학습 안 함)
- B, A: LoRA 가중치 (학습함)
- r: rank (보통 8, 16, 32)

### 파라미터 계산

**원본 레이어:**
```
W: [4096, 4096]
parameters = 4096 × 4096 = 16,777,216
```

**LoRA (r=16):**
```
B: [4096, 16]
A: [16, 4096]
parameters = 4096×16 + 16×4096 = 131,072
```

**비율: 0.78%** 🎉

---

## LoRA 구현

### 1. 기본 LoRA Layer

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        """
        Args:
            x: [batch, ..., in_features]
        Returns:
            delta: [batch, ..., out_features]
        """
        # x @ A^T @ B^T
        result = x @ self.lora_A.T  # [batch, ..., rank]
        result = result @ self.lora_B.T  # [batch, ..., out_features]
        result = result * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=16, alpha=16):
        super().__init__()
        
        # 원본 레이어 (frozen)
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA 레이어
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        # h = Wx + BAx
        return self.linear(x) + self.lora(x)
```

### 2. 모델에 LoRA 적용

```python
def apply_lora_to_model(model, rank=16, alpha=16, target_modules=None):
    """
    모델의 특정 레이어를 LoRA로 교체
    
    Args:
        model: Transformer 모델
        rank: LoRA rank
        alpha: LoRA alpha (scaling)
        target_modules: LoRA를 적용할 모듈 이름들
                       (예: ['q_proj', 'v_proj'])
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for name, module in model.named_modules():
        # Attention의 linear layer들만 교체
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 부모 모듈 찾기
                parent_name = '.'.join(name.split('.')[:-1])
                parent = model.get_submodule(parent_name)
                
                # LoRA로 교체
                lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
                setattr(parent, name.split('.')[-1], lora_layer)
    
    return model
```

### 3. 학습 예제

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA 적용
model = apply_lora_to_model(model, rank=16, alpha=16)

# 학습 가능한 파라미터만 출력
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
# Trainable: 4,194,304 (0.06%)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

# 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# LoRA 가중치만 저장
torch.save({
    'lora_A': model.lora_A.state_dict(),
    'lora_B': model.lora_B.state_dict(),
}, 'lora_weights.pt')
```

---

## PEFT 라이브러리 사용

HuggingFace의 PEFT 라이브러리를 쓰면 더 쉽습니다.

### 설치

```bash
pip install peft
```

### 기본 사용

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                        # rank
    lora_alpha=16,               # alpha scaling
    target_modules=[             # 어떤 레이어에 적용할지
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,           # dropout
    bias="none",                 # bias 학습 안 함
    task_type="CAUSAL_LM"        # task 종류
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 파라미터 확인
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### 학습

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_llama2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 저장 & 로드

```python
# LoRA 어댑터만 저장 (몇 MB)
model.save_pretrained("./lora_adapter")

# 로드
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 추론
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
```

---

## 고급 기법

### 1. QLoRA (Quantized LoRA)

**아이디어:** 베이스 모델을 4-bit로 양자화해서 메모리 더 절약

```python
from transformers import BitsAndBytesConfig

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드 (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
```

**메모리:**
- Full fine-tuning: 94 GB
- LoRA (fp16): 18 GB
- QLoRA (4-bit): **9 GB** 🎉

**RTX 4090 24GB로도 학습 가능!**

### 2. LoRA+

**문제:** A와 B의 학습률이 같으면 비효율적

**해결:** B는 더 빠르게, A는 천천히

```python
# LoRA+ optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if 'lora_B' in n],
        'lr': 3e-4 * 16,  # B는 16배 빠르게
    },
    {
        'params': [p for n, p in model.named_parameters() if 'lora_A' in n],
        'lr': 3e-4,       # A는 기본 속도
    }
]

optimizer = torch.optim.AdamW(param_groups)
```

### 3. DoRA (Weight-Decomposed LoRA)

**아이디어:** 가중치를 크기(magnitude)와 방향(direction)으로 분해

```python
W_new = m * (W + BA) / ||W + BA||

where:
  m: learnable magnitude
  W + BA: direction
```

**구현:**
```python
class DoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
    def forward(self, W, x):
        # Direction
        direction = W + self.lora_B @ self.lora_A
        direction = direction / direction.norm(dim=1, keepdim=True)
        
        # Magnitude
        W_new = self.magnitude.unsqueeze(1) * direction
        
        return x @ W_new.T
```

### 4. AdaLoRA (Adaptive LoRA)

**아이디어:** 각 레이어마다 다른 rank 사용

```python
# 중요한 레이어는 높은 rank
lora_config = LoraConfig(
    r=16,  # 기본
    init_r=32,  # 초기 rank (pruning 됨)
    target_r=8,  # 목표 평균 rank
    # ...
)
```

---

## 실전 예제: 챗봇 Fine-tuning

### 데이터셋 준비

```python
from datasets import load_dataset

# 대화 데이터셋
dataset = load_dataset("databricks/databricks-dolly-15k")

def format_instruction(sample):
    """대화 형식으로 변환"""
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}"""

def tokenize(sample):
    text = format_instruction(sample)
    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 토크나이징
train_dataset = dataset['train'].map(tokenize, remove_columns=dataset['train'].column_names)
```

### LoRA 학습

```python
# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 모델 준비
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
model = get_peft_model(model, lora_config)

# 학습
training_args = TrainingArguments(
    output_dir="./llama2_dolly_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=3,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# 저장
model.save_pretrained("./llama2_dolly_lora")
```

### 추론

```python
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./llama2_dolly_lora")

# 추론 모드
model.eval()

# 테스트
prompt = """### Instruction:
Explain what is machine learning.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 성능 비교

### 메모리 사용량

**LLaMA-7B Fine-tuning:**

| 방식 | 메모리 (GB) | 학습 가능 |
|------|------------|----------|
| Full fine-tuning | 94 | A100 80GB 부족 |
| LoRA (fp16) | 18 | A100 40GB |
| QLoRA (4-bit) | 9 | RTX 4090 24GB |
| QLoRA (3-bit) | 6 | RTX 3090 24GB |

### 정확도

**GLUE Benchmark (RoBERTa-base):**

| 방식 | 평균 점수 |
|------|----------|
| Full fine-tuning | 87.6 |
| LoRA (r=8) | 87.2 (-0.4) |
| LoRA (r=16) | 87.5 (-0.1) |
| LoRA (r=32) | 87.6 (동일) |

**결론:** rank 16-32면 full fine-tuning과 거의 동일!

### 학습 속도

| 방식 | 시간 (epoch당) |
|------|---------------|
| Full fine-tuning | 45분 |
| LoRA | 38분 (15% 빠름) |
| QLoRA | 52분 (15% 느림) |

---

## Best Practices

### 1. Rank 선택

**일반 가이드:**
- **작은 모델** (< 1B): r=4-8
- **중간 모델** (1-10B): r=8-16
- **큰 모델** (> 10B): r=16-64

**실험 추천:**
```python
ranks = [4, 8, 16, 32]
for r in ranks:
    # 작은 데이터로 테스트
    test_lora(r, num_samples=1000)
```

### 2. Target Modules

**추천 우선순위:**
1. **Q, V**: 필수 (가장 효과적)
2. **K, O**: 추가 성능
3. **MLP layers**: 더 많은 용량 필요할 때

```python
# Minimal (빠름, 적은 메모리)
target_modules = ["q_proj", "v_proj"]

# Recommended (밸런스)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full (최고 성능, 느림)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### 3. Learning Rate

**LoRA는 더 높은 learning rate 필요:**
- Full fine-tuning: 1e-5 ~ 5e-5
- LoRA: **3e-4 ~ 1e-3**

### 4. Alpha 설정

**Alpha = 2 × rank** (일반적)
- rank=8 → alpha=16
- rank=16 → alpha=32

---

## 여러 LoRA 어댑터 관리

### 1. 여러 태스크

```python
# 영어 → 한국어
lora_en_ko = PeftModel.from_pretrained(base_model, "./lora_en_ko")

# 영어 → 일본어
lora_en_ja = PeftModel.from_pretrained(base_model, "./lora_en_ja")

# 코드 생성
lora_code = PeftModel.from_pretrained(base_model, "./lora_code")
```

### 2. 동적 전환

```python
from peft import PeftModel

class MultiLoRAModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.adapters = {}
        self.current_adapter = None
    
    def load_adapter(self, name, path):
        """어댑터 로드"""
        self.adapters[name] = PeftModel.from_pretrained(
            self.base_model, path
        )
    
    def switch_adapter(self, name):
        """어댑터 전환"""
        self.current_adapter = self.adapters[name]
    
    def generate(self, prompt):
        return self.current_adapter.generate(prompt)

# 사용
model = MultiLoRAModel(base_model)
model.load_adapter("translate", "./lora_translate")
model.load_adapter("code", "./lora_code")

# 번역
model.switch_adapter("translate")
output = model.generate("Hello world")

# 코드 생성
model.switch_adapter("code")
output = model.generate("Write a function to sort a list")
```

### 3. 어댑터 병합

여러 LoRA를 하나로:

```python
from peft import PeftModel

# LoRA 1
model1 = PeftModel.from_pretrained(base_model, "./lora1")

# LoRA 2 추가
model1.load_adapter("./lora2", adapter_name="lora2")

# 가중치 평균
model1.set_adapter(["default", "lora2"])  # 둘 다 활성화
```

---

## 요약

**LoRA**는:

1. **0.1-1%의 파라미터**만 학습
2. **저랭크 분해**: ΔW ≈ BA
3. **메모리**: 10배 절약
4. **성능**: Full fine-tuning과 동일
5. **속도**: 비슷하거나 더 빠름

**QLoRA**:
- 4-bit 양자화 + LoRA
- 24GB GPU로 70B 모델 학습!

**사용처**:
- Instruction tuning
- Domain adaptation
- Task-specific fine-tuning
- Multi-task learning

**핵심**: 적은 비용으로 강력한 모델 커스터마이징!

---

## 시리즈 완료! 🎉

LLM 최적화 시리즈:
1. **Paged Attention**: 메모리 효율 10배
2. **KV Caching**: 속도 50-100배
3. **LoRA**: 학습 비용 10배 절감

이제 여러분도 효율적인 LLM inference & fine-tuning을 할 수 있습니다!

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
