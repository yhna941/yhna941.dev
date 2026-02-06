---
title: "MLOps #2: 실험 관리 - MLflow와 Weights & Biases"
description: "수백 번의 실험을 추적하고 비교하며 재현 가능한 ML 연구 환경을 구축합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["mlops", "mlflow", "wandb", "experiment-tracking", "reproducibility"]
draft: false
---

# MLOps #2: 실험 관리

**"어제 학습한 모델이 어디 갔지?"**

실험 관리가 없을 때:
```
model_v1.pt
model_v2_final.pt
model_v2_final_REAL.pt
model_v3_this_time_for_sure.pt
model_best_95accuracy.pt ← 이게 뭐였더라?
```

실험 관리가 있을 때:
```
실험 #142:
- 하이퍼파라미터: lr=0.001, batch=32
- 메트릭: acc=95.2%, loss=0.12
- 코드 버전: commit abc123
- 데이터: train_v5.parquet
- 재현 가능! ✅
```

---

## 실험 추적의 필요성

### 문제

**연구 중:**
```python
# 실험 1
model = Model(hidden_dim=128)
train(model, lr=0.001)
# accuracy: 92%

# 실험 2 (다음 날)
model = Model(hidden_dim=256)  # 뭐가 달라졌지?
train(model, lr=0.001)
# accuracy: 94%

# 실험 3 (일주일 후)
# ... 실험 1이 뭐였더라? 🤔
```

**추적해야 할 것:**
1. **하이퍼파라미터**: lr, batch_size, hidden_dim, ...
2. **메트릭**: accuracy, loss, F1, ...
3. **아티팩트**: 모델 파일, 차트, 로그
4. **환경**: Python 버전, 라이브러리, 시드
5. **코드**: Git commit, 변경사항
6. **데이터**: 어떤 데이터셋 사용?

---

## MLflow

### 기본 사용

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

# 1. 실험 설정
mlflow.set_experiment("image_classification")

# 2. 실험 시작
with mlflow.start_run(run_name="resnet50_baseline"):
    
    # 3. 하이퍼파라미터 로깅
    params = {
        'model': 'resnet50',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'optimizer': 'Adam'
    }
    mlflow.log_params(params)
    
    # 4. 모델 학습
    model = ResNet50()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    for epoch in range(params['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)
        
        # 5. 메트릭 로깅 (각 epoch마다)
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, step=epoch)
        
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}")
    
    # 6. 최종 메트릭
    test_acc = evaluate(model, test_loader)
    mlflow.log_metric('test_acc', test_acc)
    
    # 7. 모델 저장
    mlflow.pytorch.log_model(model, "model")
    
    # 8. 아티팩트 (차트, 설정 파일 등)
    plot_confusion_matrix(model, test_loader, save_path='confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # 9. 태그
    mlflow.set_tags({
        'team': 'research',
        'project': 'image-classification',
        'model_architecture': 'resnet50'
    })
```

### MLflow UI

```bash
# MLflow 서버 실행
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 브라우저에서: http://localhost:5000
```

**UI 기능:**
- 모든 실험 비교
- 메트릭 차트 시각화
- 하이퍼파라미터 필터링
- 모델 다운로드

---

## MLflow 고급 기능

### 1. Nested Runs (계층적 실험)

```python
# 부모 실험: 하이퍼파라미터 탐색
with mlflow.start_run(run_name="hyperparameter_search") as parent_run:
    
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [16, 32, 64]:
            # 자식 실험
            with mlflow.start_run(
                run_name=f"lr_{lr}_batch_{batch_size}",
                nested=True
            ):
                mlflow.log_params({
                    'learning_rate': lr,
                    'batch_size': batch_size
                })
                
                model = train_model(lr, batch_size)
                acc = evaluate(model)
                
                mlflow.log_metric('accuracy', acc)
    
    # 부모 실험에 최고 성능 기록
    best_acc = max(all_accuracies)
    mlflow.log_metric('best_accuracy', best_acc)
```

### 2. Autologging (자동 로깅)

```python
import mlflow.pytorch

# PyTorch autologging 활성화
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False
)

# 이제 자동으로 로깅됨!
with mlflow.start_run():
    model = YourModel()
    train(model)  # 메트릭, 파라미터, 모델 자동 저장
```

**지원 프레임워크:**
- PyTorch, TensorFlow, Keras
- XGBoost, LightGBM, Scikit-learn
- Spark MLlib, Fastai

### 3. Model Registry (프로덕션 관리)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. 모델 등록
run_id = "abc123"
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(
    model_uri,
    "ImageClassifier"
)

# 2. 모델 버전 정보
print(f"Name: {model_details.name}")
print(f"Version: {model_details.version}")

# 3. Stage 관리
client.transition_model_version_stage(
    name="ImageClassifier",
    version=5,
    stage="Staging"  # None, Staging, Production, Archived
)

# 4. 모델 설명 추가
client.update_model_version(
    name="ImageClassifier",
    version=5,
    description="ResNet50 trained on ImageNet, 95% accuracy"
)

# 5. Production 모델 로드
model = mlflow.pyfunc.load_model("models:/ImageClassifier/Production")
predictions = model.predict(test_data)

# 6. 모델 비교
def compare_model_versions(model_name, versions):
    comparison = []
    
    for version in versions:
        mv = client.get_model_version(model_name, version)
        run = client.get_run(mv.run_id)
        
        comparison.append({
            'version': version,
            'stage': mv.current_stage,
            'accuracy': run.data.metrics.get('test_acc'),
            'created': mv.creation_timestamp
        })
    
    return pd.DataFrame(comparison)

df = compare_model_versions("ImageClassifier", [3, 4, 5])
print(df)
```

---

## Weights & Biases (W&B)

### 기본 사용

```python
import wandb

# 1. 초기화
wandb.init(
    project="image-classification",
    name="resnet50-experiment",
    config={
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'model': 'resnet50'
    }
)

# 2. 학습 루프
for epoch in range(wandb.config.epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    # 3. 메트릭 로깅
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

# 4. 완료
wandb.finish()
```

### W&B 고급 기능

#### 1. 이미지/비디오 로깅

```python
import wandb
import matplotlib.pyplot as plt

# 이미지 로깅
images = []
for img, pred, label in zip(sample_images, predictions, labels):
    images.append(wandb.Image(
        img,
        caption=f"Pred: {pred}, Label: {label}"
    ))

wandb.log({"predictions": images})

# Matplotlib 차트
fig, ax = plt.subplots()
ax.plot(train_losses)
ax.set_title("Training Loss")

wandb.log({"loss_curve": wandb.Image(fig)})
plt.close(fig)

# Confusion Matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names
    )
})
```

#### 2. 하이퍼파라미터 Sweep (자동 튜닝)

```python
# sweep 설정
sweep_config = {
    'method': 'bayes',  # grid, random, bayes
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        }
    }
}

# Sweep 초기화
sweep_id = wandb.sweep(sweep_config, project="image-classification")

# 학습 함수
def train():
    wandb.init()
    
    # wandb.config에서 하이퍼파라미터 가져오기
    config = wandb.config
    
    model = create_model(
        dropout=config.dropout,
        optimizer=config.optimizer
    )
    
    for epoch in range(50):
        train_loss = train_epoch(
            model,
            train_loader,
            lr=config.learning_rate,
            batch_size=config.batch_size
        )
        val_acc = validate(model, val_loader)
        
        wandb.log({
            'train_loss': train_loss,
            'val_accuracy': val_acc
        })

# Sweep 실행 (10회 시도)
wandb.agent(sweep_id, train, count=10)
```

#### 3. Artifacts (데이터/모델 버저닝)

```python
import wandb

run = wandb.init(project="my-project")

# 데이터셋 저장
artifact = wandb.Artifact('training-data', type='dataset')
artifact.add_file('train.csv')
artifact.add_file('val.csv')
run.log_artifact(artifact)

# 모델 저장
model_artifact = wandb.Artifact('resnet50', type='model')
model_artifact.add_file('model.pt')
run.log_artifact(model_artifact)

# 사용 (다른 실험에서)
run = wandb.init(project="my-project")

# 데이터셋 다운로드
artifact = run.use_artifact('training-data:latest')
artifact_dir = artifact.download()

# 모델 다운로드
model_artifact = run.use_artifact('resnet50:v3')
model_path = model_artifact.download()
model = torch.load(f"{model_path}/model.pt")
```

#### 4. Reports (실험 공유)

```python
# W&B UI에서 리포트 생성 후...

# API로 리포트 생성
import wandb

api = wandb.Api()

# 실험 가져오기
runs = api.runs("my-project")

# 비교 테이블
comparison = []
for run in runs:
    comparison.append({
        'name': run.name,
        'accuracy': run.summary.get('val_accuracy'),
        'loss': run.summary.get('val_loss'),
        'config': run.config
    })

df = pd.DataFrame(comparison)
print(df.sort_values('accuracy', ascending=False))
```

---

## MLflow vs W&B

### 비교

| Feature | MLflow | W&B |
|---------|--------|-----|
| **비용** | 무료 (self-hosted) | Free tier + 유료 |
| **UI** | 기본적 | 강력, 인터랙티브 |
| **설치** | 쉬움 | 더 쉬움 |
| **모델 레지스트리** | ✅ 강력 | ✅ Artifacts |
| **하이퍼파라미터 튜닝** | ❌ | ✅ Sweep |
| **실시간 시각화** | 제한적 | ✅ 강력 |
| **협업** | 제한적 | ✅ 강력 |
| **자체 호스팅** | ✅ | ✅ (Enterprise) |

### 함께 사용하기

```python
import mlflow
import wandb

# 둘 다 사용!
wandb.init(project="my-project", sync_tensorboard=True)
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    for epoch in range(50):
        loss = train_epoch()
        acc = validate()
        
        # MLflow
        mlflow.log_metrics({
            'loss': loss,
            'accuracy': acc
        }, step=epoch)
        
        # W&B
        wandb.log({
            'loss': loss,
            'accuracy': acc
        })
    
    # MLflow에 모델 저장
    mlflow.pytorch.log_model(model, "model")
    
    # W&B에도 저장
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.pt')
    wandb.log_artifact(artifact)

wandb.finish()
```

---

## 재현 가능성 (Reproducibility)

### 1. 환경 추적

```python
import mlflow
import torch
import random
import numpy as np

def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with mlflow.start_run():
    seed = 42
    set_seed(seed)
    
    # 환경 정보 로깅
    mlflow.log_params({
        'seed': seed,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0)
    })
    
    # Git 정보
    import subprocess
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    mlflow.log_param('git_commit', commit)
    
    # 코드 저장
    mlflow.log_artifact('train.py')
    mlflow.log_artifact('model.py')
```

### 2. Docker로 환경 캡슐화

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 코드 복사
COPY . .

# MLflow 환경변수
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# 학습 실행
CMD ["python", "train.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    image: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db
  
  training:
    build: .
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

---

## 실전 워크플로우

### 연구 팀 워크플로우

```python
class ExperimentManager:
    """실험 관리 헬퍼"""
    
    def __init__(self, project_name, experiment_name):
        self.project = project_name
        self.experiment = experiment_name
        
        # W&B 초기화
        wandb.init(project=project_name)
        
        # MLflow 초기화
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name, config):
        """실험 시작"""
        # W&B run
        wandb.run.name = run_name
        wandb.config.update(config)
        
        # MLflow run
        self.mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)
        
        return self
    
    def log_metrics(self, metrics, step=None):
        """메트릭 로깅"""
        wandb.log(metrics, step=step)
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, name):
        """모델 저장"""
        # W&B
        artifact = wandb.Artifact(name, type='model')
        torch.save(model.state_dict(), f'{name}.pt')
        artifact.add_file(f'{name}.pt')
        wandb.log_artifact(artifact)
        
        # MLflow
        mlflow.pytorch.log_model(model, name)
    
    def finish(self):
        """실험 종료"""
        wandb.finish()
        mlflow.end_run()

# 사용
manager = ExperimentManager("image-classification", "resnet-experiments")

config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

manager.start_run("baseline", config)

for epoch in range(config['epochs']):
    loss = train_epoch()
    acc = validate()
    
    manager.log_metrics({
        'loss': loss,
        'accuracy': acc
    }, step=epoch)

manager.log_model(model, "resnet50")
manager.finish()
```

---

## 요약

**실험 관리 핵심:**

1. **추적**: 하이퍼파라미터, 메트릭, 코드, 환경
2. **비교**: 실험 간 성능 비교
3. **재현**: 같은 결과 다시 만들기
4. **공유**: 팀과 결과 공유

**MLflow:**
- Self-hosted
- 모델 레지스트리 강력
- Production 관리

**W&B:**
- SaaS (쉬운 시작)
- 하이퍼파라미터 튜닝
- 협업 기능 강력

**Best Practices:**
- 모든 실험 추적
- 시드 고정
- Git commit 기록
- 환경 정보 저장

**다음 글:**
- **MLOps #3**: 모니터링 & A/B 테스팅
- **Serving #1**: 추론 서버 최적화
- **Serving #2**: 배치 vs 스트리밍

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
