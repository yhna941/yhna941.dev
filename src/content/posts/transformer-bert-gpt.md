---
title: "Transformer #3: BERT & GPT - Pre-training의 혁명"
description: "Encoder-only BERT와 Decoder-only GPT의 구조, Pre-training 전략, 그리고 Fine-tuning을 완전히 이해합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["transformer", "bert", "gpt", "pre-training", "nlp", "pytorch"]
draft: false
---

# Transformer #3: BERT & GPT

**"Pre-training + Fine-tuning = New Paradigm"**

2018년, NLP의 패러다임이 완전히 바뀌었습니다.

이번 글:
- BERT (Encoder-only)
- GPT (Decoder-only)
- Pre-training 전략
- Fine-tuning

---

## Pre-training의 등장

### 이전: Task-specific 학습

```
Machine Translation: 번역 데이터로 학습
Sentiment Analysis: 감정 데이터로 학습
QA: QA 데이터로 학습

문제:
- 각 태스크마다 데이터 필요
- 처음부터 학습 (느림)
- 작은 데이터셋 (overfitting)
```

### 새로운 패러다임: Transfer Learning

```
1. Pre-training (대량의 unlabeled data)
   → 언어의 일반적 지식 학습

2. Fine-tuning (소량의 labeled data)
   → 특정 태스크에 적응

결과:
- 적은 데이터로 높은 성능
- 빠른 학습
- 범용 표현 학습
```

**비유: 대학 교육**

```
Pre-training = 일반 교육 (수학, 과학, 언어...)
Fine-tuning = 전공 교육 (의학, 공학, 법학...)

일반 교육 없이 전공만? 비효율!
```

---

## BERT (2018)

**Bidirectional Encoder Representations from Transformers**

### 아키텍처

**Encoder-only Transformer:**

```
Input
  ↓
Embedding + Positional Encoding
  ↓
Encoder × 12 (Base) or 24 (Large)
  ↓
Output (Contextualized Embeddings)
```

**크기:**

```
BERT-Base:
- Layers: 12
- Hidden: 768
- Heads: 12
- Parameters: 110M

BERT-Large:
- Layers: 24
- Hidden: 1024
- Heads: 16
- Parameters: 340M
```

### Pre-training Tasks

#### 1. Masked Language Model (MLM)

**아이디어:**

> "문장의 일부를 가리고 맞추기"

```
Input: "I love [MASK] learning"
Label: "deep"

Input: "The [MASK] is [MASK] the street"
Label: "animal", "crossing"
```

**방법:**

1. 15% 토큰 선택
   - 80%: [MASK]로 대체
   - 10%: 랜덤 토큰으로 대체
   - 10%: 그대로 (unchanged)

2. 모델이 원래 토큰 예측

**왜 랜덤/unchanged?**

```
Fine-tuning 시에는 [MASK]가 없음!
→ [MASK]에만 의존하지 않도록
```

**구현:**

```python
import torch
import torch.nn as nn
import random

class MLMDataset:
    def __init__(self, texts, tokenizer, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Create labels (copy of tokens)
        labels = tokens.copy()
        
        # Mask tokens
        for i in range(len(tokens)):
            if random.random() < self.mask_prob:
                rand = random.random()
                
                if rand < 0.8:
                    # 80%: Replace with [MASK]
                    tokens[i] = self.mask_token_id
                elif rand < 0.9:
                    # 10%: Replace with random token
                    tokens[i] = random.randint(0, self.vocab_size - 1)
                # 10%: Keep original (do nothing)
            else:
                # Not masked: ignore in loss
                labels[i] = -100  # PyTorch ignore_index
        
        return torch.tensor(tokens), torch.tensor(labels)
```

#### 2. Next Sentence Prediction (NSP)

**아이디어:**

> "두 문장이 이어지는가?"

```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]

Example 1 (IsNext):
A: "I love deep learning."
B: "It's very interesting."
Label: 1

Example 2 (NotNext):
A: "I love deep learning."
B: "The sky is blue."
Label: 0
```

**데이터 생성:**

```python
def create_nsp_data(documents):
    examples = []
    
    for doc in documents:
        sentences = doc.split('.')
        
        for i in range(len(sentences) - 1):
            # 50%: IsNext
            if random.random() < 0.5:
                sent_a = sentences[i]
                sent_b = sentences[i + 1]
                label = 1
            # 50%: NotNext
            else:
                sent_a = sentences[i]
                sent_b = random.choice(sentences)
                label = 0
            
            examples.append((sent_a, sent_b, label))
    
    return examples
```

### BERT 모델

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_len=512):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # For NSP
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers
        )
        
        # MLM head
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # NSP head
        self.nsp_head = nn.Linear(d_model, 2)
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        segment_ids: (batch, seq_len) - 0 for sent A, 1 for sent B
        attention_mask: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.size()
        
        # Position IDs
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        segment_emb = self.segment_embedding(segment_ids)
        
        embeddings = token_emb + position_emb + segment_emb
        
        # Encode
        encoder_output = self.encoder(embeddings, attention_mask)
        
        # MLM predictions (all tokens)
        mlm_logits = self.mlm_head(encoder_output)
        
        # NSP prediction ([CLS] token)
        cls_output = encoder_output[:, 0]  # First token
        nsp_logits = self.nsp_head(cls_output)
        
        return mlm_logits, nsp_logits

# 사용
model = BERT(vocab_size=30000)

input_ids = torch.randint(0, 30000, (32, 128))
segment_ids = torch.cat([
    torch.zeros(32, 64),
    torch.ones(32, 64)
], dim=1).long()

mlm_logits, nsp_logits = model(input_ids, segment_ids)
print(mlm_logits.shape)  # (32, 128, 30000)
print(nsp_logits.shape)  # (32, 2)
```

### Fine-tuning

**다양한 태스크:**

```python
class BERTForClassification(nn.Module):
    """Sentiment analysis, topic classification, etc."""
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, segment_ids, attention_mask):
        # Get [CLS] representation
        encoder_output = self.bert.encoder(input_ids, attention_mask)
        cls_output = encoder_output[:, 0]
        
        # Classify
        logits = self.classifier(cls_output)
        return logits

class BERTForQA(nn.Module):
    """Question Answering (SQuAD)"""
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(768, 2)  # Start & End
    
    def forward(self, input_ids, segment_ids, attention_mask):
        # Encode
        encoder_output = self.bert.encoder(input_ids, attention_mask)
        
        # Predict start & end positions
        logits = self.qa_outputs(encoder_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

class BERTForNER(nn.Module):
    """Named Entity Recognition"""
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, segment_ids, attention_mask):
        # Encode (get all token representations)
        encoder_output = self.bert.encoder(input_ids, attention_mask)
        
        # Classify each token
        logits = self.classifier(encoder_output)
        return logits
```

---

## GPT (2018)

**Generative Pre-trained Transformer**

### 아키텍처

**Decoder-only Transformer:**

```
Input
  ↓
Embedding + Positional Encoding
  ↓
Masked Decoder × 12
  ↓
Language Model Head
  ↓
Next Token Prediction
```

**왜 Decoder-only?**

```
언어 모델 = Auto-regressive 생성
→ Masked Self-Attention 필요
→ Decoder가 적합!
```

### Pre-training: Language Modeling

**아이디어:**

> "다음 단어 예측"

```
Input:  "I love deep"
Output: "learning"

Input:  "The quick brown"
Output: "fox"
```

**목적 함수:**

$$
\mathcal{L} = -\sum_{i=1}^n \log P(w_i | w_1, \ldots, w_{i-1})
$$

**구현:**

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_len=1024):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Decoder layers (masked self-attention)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (input embedding = output projection)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        
        x = token_emb + position_emb
        
        # Causal mask (can't see future)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # Language model prediction
        logits = self.lm_head(x)
        
        return logits

# Training
model = GPT(vocab_size=50000)
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

for batch in dataloader:
    input_ids = batch['input_ids']  # (batch, seq_len)
    
    # Forward
    logits = model(input_ids[:, :-1])  # All but last token
    
    # Loss (predict next token)
    targets = input_ids[:, 1:]  # All but first token
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Fine-tuning

**GPT의 특별한 점:**

> "Task-specific head 없이도 동작!"

```python
# Classification
def classify(model, text):
    prompt = f"{text} [SEP] Label:"
    tokens = tokenizer.encode(prompt)
    
    # Generate
    output = model.generate(tokens, max_len=5)
    
    # Extract label
    label = tokenizer.decode(output[-1])
    return label

# QA
def qa(model, question, context):
    prompt = f"Context: {context} [SEP] Question: {question} [SEP] Answer:"
    tokens = tokenizer.encode(prompt)
    
    # Generate
    answer = model.generate(tokens, max_len=50)
    return tokenizer.decode(answer)
```

---

## BERT vs GPT

### 구조 비교

```
BERT:
- Encoder-only
- Bidirectional
- [MASK] 기반 학습

GPT:
- Decoder-only
- Unidirectional (left-to-right)
- Next token 예측
```

### 장단점

**BERT:**

```
장점:
✅ Bidirectional → 더 풍부한 표현
✅ Classification 태스크 강함
✅ 문장 간 관계 이해

단점:
❌ 생성 불가능
❌ [MASK] token이 fine-tuning 시 없음
```

**GPT:**

```
장점:
✅ 텍스트 생성 가능
✅ Zero-shot learning
✅ Pre-training과 fine-tuning 일관성

단점:
❌ Unidirectional → 제한적 표현
❌ Classification 약함
```

### 언제 사용?

```python
if task == "classification":
    return "BERT"
elif task == "generation":
    return "GPT"
elif task == "qa":
    return "BERT"  # 문맥 이해 중요
elif task == "summarization":
    return "GPT"  # 생성 필요
```

---

## GPT-2, GPT-3의 진화

### GPT-2 (2019)

**크기 증가:**

```
GPT-2:
- Parameters: 1.5B
- Data: 40GB (WebText)

결과:
- Zero-shot 성능 향상
- Few-shot learning 가능
```

**Byte Pair Encoding (BPE):**

```python
# Subword tokenization
"playing" → ["play", "ing"]
"unbelievable" → ["un", "believ", "able"]

장점:
- OOV (Out-of-Vocabulary) 해결
- 더 작은 vocabulary
```

### GPT-3 (2020)

**거대화:**

```
GPT-3:
- Parameters: 175B
- Data: 300B tokens
- Context: 2048 tokens → 4096

결과:
- Few-shot learning 강력
- Prompt engineering
```

**In-context Learning:**

```
# Few-shot example
Input:
Q: What is the capital of France?
A: Paris

Q: What is the capital of Germany?
A: Berlin

Q: What is the capital of Italy?
A:

Output: Rome
```

---

## 실전 사용 (Hugging Face)

### BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Fine-tuning
text = "I love this movie!"
inputs = tokenizer(text, return_tensors='pt')
labels = torch.tensor([1])  # Positive

outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

### GPT-2

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

---

## 요약

**BERT:**
- Encoder-only
- Masked Language Model + NSP
- Classification 강함
- Bidirectional 표현

**GPT:**
- Decoder-only
- Language Modeling
- 생성 가능
- Unidirectional

**핵심 인사이트:**

> "Pre-training으로 일반적 언어 지식을 학습하고, Fine-tuning으로 특정 태스크에 적응"

**다음 글:**
- **Vision Transformer**: 이미지에 Transformer
- **Multimodal Models**: CLIP, Flamingo
- **Efficient Transformers**: Linformer, Performer

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
