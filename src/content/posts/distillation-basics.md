---
title: "Knowledge Distillation #1: 기초 - Teacher-Student 학습"
description: "큰 모델의 지식을 작은 모델로 전이하는 Knowledge Distillation의 원리와 구현을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["distillation", "model-compression", "deep-learning", "pytorch"]
draft: false
---

# Knowledge Distillation #1: 기초

**"큰 모델의 지식을 작은 모델로"**

문제:
```
GPT-3: 175B parameters
→ 너무 느림, 비용 큼

해결:
Teacher (GPT-3) → Student (작은 모델)
→ 10배 빠르고, 90%+ 성능 유지!
```

---

## Knowledge Distillation이란?

### 정의

> **Teacher 모델의 지식을 Student 모델로 전이**

```
┌──────────────┐
│   Teacher    │ (Large, Accurate)
│  175B params │
└──────┬───────┘
       │ Knowledge Transfer
       ↓
┌──────────────┐
│   Student    │ (Small, Fast)
│   1B params  │
└──────────────┘
```

### 왜 필요?

**실제 배포 시:**

```
큰 모델:
- Latency: 2초
- Cost: $0.01/request
- Memory: 80GB

작은 모델:
- Latency: 200ms (10배 빠름)
- Cost: $0.001/request (10배 저렴)
- Memory: 8GB (10배 적음)
```

**Mobile/Edge:**
```
스마트폰에 175B 모델? 불가능!
→ Distillation으로 1B 모델 생성
→ 온디바이스 AI 가능!
```

---

## 기본 원리 (Hinton et al., 2015)

### Hard Labels vs Soft Labels

**일반 학습 (Hard Labels):**

```python
# Classification: "고양이" 이미지
Hard Label = [0, 0, 1, 0, 0]
             [개, 새, 고양이, 말, 소]

문제: 정보 손실!
- "고양이"인 것만 알 수 있음
- 개와의 유사도? 알 수 없음
```

**Distillation (Soft Labels):**

```python
# Teacher의 출력 (확률 분포)
Soft Label = [0.05, 0.02, 0.85, 0.03, 0.05]
             [개,   새,   고양이, 말,   소]

정보:
- 고양이다 (85%)
- 개와 약간 유사 (5%)
- 소와도 약간 유사 (5%)
- 새와는 거의 다름 (2%)

→ 더 풍부한 지식!
```

### Temperature Scaling

**Softmax:**

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

**Temperature Softmax:**

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- $T = 1$: 일반 softmax
- $T > 1$: 확률 분포 부드럽게 (softer)
- $T < 1$: 확률 분포 날카롭게 (sharper)

**예시:**

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])

# T=1 (일반)
p1 = F.softmax(logits, dim=0)
print(p1)  # [0.659, 0.242, 0.099]

# T=5 (부드럽게)
p5 = F.softmax(logits / 5.0, dim=0)
print(p5)  # [0.391, 0.331, 0.278] → 더 균등

# T=0.5 (날카롭게)
p05 = F.softmax(logits / 0.5, dim=0)
print(p05)  # [0.952, 0.047, 0.001] → 더 집중
```

**왜 Temperature?**

```
높은 T:
- "Dark knowledge" 드러남
- 클래스 간 관계 정보
- Student가 배우기 쉬움

예:
logits = [10, 5, 1]  (고양이, 개, 소)

T=1:  [0.9998, 0.0002, 0.0000]  → 정보 부족
T=10: [0.52,   0.36,   0.12]    → 관계 보임!
```

### Distillation Loss

**결합 손실:**

$$
\mathcal{L} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \mathcal{L}_{KL}(\sigma(z_t/T), \sigma(z_s/T))
$$

- $\mathcal{L}_{CE}$: Hard label loss (정확도)
- $\mathcal{L}_{KL}$: Soft label loss (Teacher 모방)
- $\alpha$: 균형 파라미터 (보통 0.1-0.3)
- $T$: Temperature

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.7):
    """
    student_logits: (batch, num_classes)
    teacher_logits: (batch, num_classes)
    labels: (batch,) - hard labels
    """
    # 1. Hard label loss (student predictions)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft label loss (KL divergence with teacher)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    # 3. Combine
    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return loss
```

**왜 $T^2$?**

```
KL divergence의 gradient는 T에 비례
→ T가 크면 gradient 작아짐
→ T^2 곱해서 보정
```

---

## 기본 구현

### Teacher 모델 학습

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Teacher: Large model
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Train teacher
teacher = TeacherModel()
optimizer = optim.Adam(teacher.parameters(), lr=0.001)

for epoch in range(20):
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        logits = teacher(images)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()

# Teacher accuracy: 95%
```

### Student 모델 정의

```python
# Student: Small model (10x smaller)
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Parameters
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(f"Teacher: {teacher_params:,} params")  # 10M
print(f"Student: {student_params:,} params")  # 1M (10x smaller!)
```

### Distillation 학습

```python
def train_distillation(teacher, student, train_loader, epochs=30, T=3.0, alpha=0.7):
    teacher.eval()  # Teacher는 freeze
    student.train()
    
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            # Student predictions
            student_logits = student(images)
            
            # Distillation loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                T=T,
                alpha=alpha
            )
            
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%")

# Train
student = StudentModel()
train_distillation(teacher, student, train_loader)

# Result
# Baseline (without distillation): 88%
# With distillation: 93%
# Teacher: 95%
```

---

## 고급 기법

### 1. Feature-based Distillation

**Idea:** 중간 feature도 전이

```python
class FeatureDistillation(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        
        # Feature adapters (dimension matching)
        self.adapters = nn.ModuleList([
            nn.Conv2d(student_dim, teacher_dim, 1)
            for student_dim, teacher_dim in zip(
                student_feature_dims,
                teacher_feature_dims
            )
        ])
    
    def forward(self, x, labels):
        # Teacher features (no grad)
        with torch.no_grad():
            teacher_features = self.teacher.extract_features(x)
            teacher_logits = self.teacher.classifier(teacher_features[-1])
        
        # Student features
        student_features = self.student.extract_features(x)
        student_logits = self.student.classifier(student_features[-1])
        
        # Logit distillation
        logit_loss = distillation_loss(student_logits, teacher_logits, labels)
        
        # Feature distillation
        feature_loss = 0
        for s_feat, t_feat, adapter in zip(
            student_features,
            teacher_features,
            self.adapters
        ):
            # Match dimensions
            s_feat_adapted = adapter(s_feat)
            # MSE loss
            feature_loss += F.mse_loss(s_feat_adapted, t_feat)
        
        # Total loss
        loss = logit_loss + 0.5 * feature_loss
        
        return loss
```

### 2. Attention Distillation

**Idea:** Attention map 전이

```python
def attention_distillation_loss(student_attn, teacher_attn):
    """
    student_attn: (batch, n_heads, seq_len, seq_len)
    teacher_attn: (batch, n_heads, seq_len, seq_len)
    """
    # Normalize
    student_attn = F.softmax(student_attn, dim=-1)
    teacher_attn = F.softmax(teacher_attn, dim=-1)
    
    # MSE on attention weights
    loss = F.mse_loss(student_attn, teacher_attn)
    
    return loss

# Transformer distillation
class TransformerDistillation(nn.Module):
    def forward(self, x, labels):
        # Teacher
        with torch.no_grad():
            teacher_logits, teacher_attns = self.teacher(x, return_attention=True)
        
        # Student
        student_logits, student_attns = self.student(x, return_attention=True)
        
        # Logit loss
        logit_loss = distillation_loss(student_logits, teacher_logits, labels)
        
        # Attention loss
        attn_loss = sum(
            attention_distillation_loss(s_attn, t_attn)
            for s_attn, t_attn in zip(student_attns, teacher_attns)
        )
        
        loss = logit_loss + 0.1 * attn_loss
        return loss
```

### 3. Self-Distillation

**Idea:** 자기 자신을 teacher로!

```python
class SelfDistillation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Multiple prediction heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(3)  # 3 auxiliary heads
        ])
        self.main_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, labels):
        # Features
        features = self.model.extract_features(x)
        
        # Main prediction
        main_logits = self.main_head(features)
        main_loss = F.cross_entropy(main_logits, labels)
        
        # Auxiliary predictions
        aux_losses = 0
        for head in self.heads:
            aux_logits = head(features)
            
            # Self-distillation: aux → main
            aux_loss = distillation_loss(
                aux_logits,
                main_logits.detach(),  # Detach main (it's the "teacher")
                labels
            )
            aux_losses += aux_loss
        
        loss = main_loss + 0.3 * aux_losses
        return loss
```

---

## 실전 예제: DistilBERT

**BERT → DistilBERT:**

```
BERT-base: 110M params
DistilBERT: 66M params (40% smaller)

Result:
- 60% faster
- 97% performance retained
- 40% smaller
```

**구현:**

```python
from transformers import BertModel, DistilBertModel, BertTokenizer
import torch.nn.functional as F

class BERTDistillation:
    def __init__(self):
        # Teacher: BERT
        self.teacher = BertModel.from_pretrained('bert-base-uncased')
        self.teacher.eval()
        
        # Student: DistilBERT (initialized randomly)
        self.student = DistilBertModel(config=distilbert_config)
        self.student.train()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def distill_step(self, texts, temperature=2.0):
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_hidden = teacher_outputs.last_hidden_state
        
        # Student forward
        student_outputs = self.student(**inputs)
        student_hidden = student_outputs.last_hidden_state
        
        # Hidden state distillation (MSE)
        hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
        
        # Attention distillation (optional)
        if hasattr(teacher_outputs, 'attentions'):
            attn_loss = sum(
                F.mse_loss(s_attn, t_attn)
                for s_attn, t_attn in zip(
                    student_outputs.attentions,
                    teacher_outputs.attentions
                )
            )
            loss = hidden_loss + 0.1 * attn_loss
        else:
            loss = hidden_loss
        
        return loss

# Train
distiller = BERTDistillation()
optimizer = torch.optim.Adam(distiller.student.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in dataloader:
        texts = batch['text']
        
        loss = distiller.distill_step(texts)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Distillation 전략

### 1. 온라인 vs 오프라인

**오프라인 (일반):**

```python
# 1. Teacher 먼저 학습
train_teacher(teacher, data)

# 2. Student 학습 (teacher frozen)
teacher.eval()
train_student(student, teacher, data)
```

**온라인 (동시 학습):**

```python
# Teacher와 Student 동시 학습
def train_online(teacher, student, data):
    teacher.train()
    student.train()
    
    for batch in data:
        # Teacher 학습
        teacher_loss = train_step(teacher, batch)
        
        # Student 학습 (teacher의 현재 지식 사용)
        student_loss = distill_step(student, teacher, batch)
        
        # 둘 다 업데이트
        update(teacher, teacher_loss)
        update(student, student_loss)
```

### 2. Data-free Distillation

**문제:** 원본 데이터 접근 불가능

**해결:** 합성 데이터 생성

```python
class DataFreeDistillation:
    def __init__(self, teacher, student, generator):
        self.teacher = teacher
        self.student = student
        self.generator = generator  # GAN-like
    
    def train_step(self):
        # 1. Generate synthetic data
        synthetic_data = self.generator.generate()
        
        # 2. Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(synthetic_data)
        
        # 3. Student learns from synthetic data
        student_logits = self.student(synthetic_data)
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1)
        )
        
        return loss
```

---

## 성능 벤치마크

### CIFAR-10

```
Teacher (ResNet-110):
- Params: 1.7M
- Accuracy: 95.0%

Student (ResNet-20):
- Params: 0.27M (6x smaller)
- Without distillation: 91.5%
- With distillation: 93.8%

Improvement: +2.3%!
```

### ImageNet

```
Teacher (ResNet-152):
- Top-1: 78.3%
- Top-5: 94.1%

Student (ResNet-18):
- Without distillation:
  * Top-1: 69.8%
  * Top-5: 89.1%
  
- With distillation:
  * Top-1: 71.4% (+1.6%)
  * Top-5: 90.3% (+1.2%)
```

---

## 요약

**Knowledge Distillation:**

1. **Soft Labels**: Teacher의 확률 분포 사용
2. **Temperature**: 부드러운 분포로 변환
3. **Combined Loss**: Hard + Soft
4. **압축**: 10배 작은 모델로 90%+ 성능

**핵심 수식:**

$$
\mathcal{L} = \alpha \mathcal{L}_{hard} + (1-\alpha) \mathcal{L}_{soft}
$$

**장점:**
- 작고 빠른 모델
- 비용 절감
- 배포 용이

**다음 글:**
- **Advanced Distillation**: CRD, DKD, ReviewKD
- **Multi-Teacher**: 여러 teacher 결합
- **Cross-Modal**: Vision → Language

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
