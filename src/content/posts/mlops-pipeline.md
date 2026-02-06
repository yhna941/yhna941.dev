---
title: "MLOps #1: ML 파이프라인 - Training부터 Production까지"
description: "실전 ML 시스템의 전체 파이프라인 구축과 자동화 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["mlops", "pipeline", "production-ml", "automation", "ml-system"]
draft: false
---

# MLOps #1: ML 파이프라인

**"모델 학습이 끝이 아니다"**

연구와 프로덕션의 차이:
```
Research:
Jupyter Notebook → 모델 학습 → acc 95% → 논문 제출 ✅

Production:
데이터 수집 → 전처리 → 학습 → 평가 → 배포 → 모니터링 
→ 재학습 → 다시 배포 → 모니터링 → ... (무한 반복)
```

---

## ML 시스템의 현실

### 코드의 5%만 ML

**실제 ML 시스템:**

```
┌────────────────────────────────────┐
│     Configuration (설정 관리)      │
├────────────────────────────────────┤
│  Data Collection (데이터 수집)     │
├────────────────────────────────────┤
│  Data Verification (검증)          │
├────────────────────────────────────┤
│  Feature Engineering (특성 추출)   │
├────────────────────────────────────┤
│  ╔══════════════╗                  │
│  ║   ML Code    ║  ← 5%만!         │
│  ╚══════════════╝                  │
├────────────────────────────────────┤
│  Model Analysis (분석)             │
├────────────────────────────────────┤
│  Serving Infrastructure (배포)     │
├────────────────────────────────────┤
│  Monitoring (모니터링)             │
├────────────────────────────────────┤
│  Resource Management (리소스)      │
└────────────────────────────────────┘
```

**Google의 통계 (Hidden Technical Debt in ML Systems):**
- ML 코드: 5%
- 인프라/파이프라인: 95%

---

## ML 파이프라인 개요

### 전체 흐름

```
┌──────────────┐
│ Data Source  │ (S3, DB, API)
└──────┬───────┘
       │
┌──────▼───────┐
│ Data Pipeline│ (수집, 정제)
└──────┬───────┘
       │
┌──────▼───────┐
│   Training   │ (모델 학습)
└──────┬───────┘
       │
┌──────▼───────┐
│  Validation  │ (평가, 검증)
└──────┬───────┘
       │
┌──────▼───────┐
│  Registry    │ (모델 저장)
└──────┬───────┘
       │
┌──────▼───────┐
│  Deployment  │ (배포)
└──────┬───────┘
       │
┌──────▼───────┐
│  Monitoring  │ (성능 추적)
└──────┬───────┘
       │
       └──────────┐ (재학습 트리거)
                  │
            ┌─────▼──────┐
            │ Retrain?   │
            └────────────┘
```

---

## 1. Data Pipeline

### 데이터 수집

```python
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import boto3

class DataCollector:
    def __init__(self, db_url, s3_bucket):
        self.engine = create_engine(db_url)
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
    
    def collect_daily_data(self, date):
        """일일 데이터 수집"""
        # 1. DB에서 데이터 가져오기
        query = f"""
        SELECT user_id, action, timestamp, features
        FROM user_events
        WHERE date = '{date}'
        """
        
        df = pd.read_sql(query, self.engine)
        
        # 2. S3에 저장 (Parquet 포맷)
        s3_key = f"raw_data/{date}/events.parquet"
        
        df.to_parquet(
            f"s3://{self.bucket}/{s3_key}",
            compression='snappy',
            index=False
        )
        
        print(f"Collected {len(df)} records for {date}")
        return s3_key
    
    def collect_streaming_data(self, kafka_topic):
        """실시간 데이터 수집 (Kafka)"""
        from kafka import KafkaConsumer
        import json
        
        consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        batch = []
        batch_size = 1000
        
        for message in consumer:
            batch.append(message.value)
            
            if len(batch) >= batch_size:
                # 배치 처리
                df = pd.DataFrame(batch)
                self.save_batch(df)
                batch = []
```

### 데이터 검증

```python
import great_expectations as ge
from typing import Dict, List

class DataValidator:
    def __init__(self, expectations_suite):
        self.suite = expectations_suite
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """데이터 품질 검증"""
        # Great Expectations로 검증
        gdf = ge.from_pandas(df)
        
        results = gdf.validate(
            expectation_suite=self.suite,
            only_return_failures=False
        )
        
        return {
            'success': results['success'],
            'failed_expectations': [
                exp for exp in results['results']
                if not exp['success']
            ],
            'statistics': results['statistics']
        }
    
    def create_expectations(self):
        """데이터 검증 룰 정의"""
        expectations = [
            # 1. 컬럼 존재
            {
                'expectation_type': 'expect_table_columns_to_match_ordered_list',
                'kwargs': {
                    'column_list': ['user_id', 'action', 'timestamp', 'features']
                }
            },
            # 2. Null 체크
            {
                'expectation_type': 'expect_column_values_to_not_be_null',
                'kwargs': {'column': 'user_id'}
            },
            # 3. 값 범위
            {
                'expectation_type': 'expect_column_values_to_be_between',
                'kwargs': {
                    'column': 'age',
                    'min_value': 0,
                    'max_value': 150
                }
            },
            # 4. 카테고리
            {
                'expectation_type': 'expect_column_values_to_be_in_set',
                'kwargs': {
                    'column': 'action',
                    'value_set': ['click', 'view', 'purchase']
                }
            }
        ]
        
        return expectations

# 사용
validator = DataValidator(expectations_suite)
df = pd.read_parquet('s3://bucket/raw_data/2024-01-01/events.parquet')

validation_result = validator.validate(df)

if not validation_result['success']:
    print("Validation failed!")
    for failure in validation_result['failed_expectations']:
        print(f"- {failure['expectation_config']['expectation_type']}")
    # 알람 발송
    send_alert("Data validation failed", validation_result)
```

### Feature Engineering Pipeline

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.pipeline = self.build_pipeline()
    
    def build_pipeline(self) -> Pipeline:
        """Feature 변환 파이프라인"""
        return Pipeline([
            ('imputer', ImputeMissing()),
            ('encoder', EncodeCategories()),
            ('scaler', ScaleFeatures()),
            ('engineer', CreateFeatures())
        ])
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """학습 + 변환"""
        return self.pipeline.fit_transform(df)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """변환만 (추론 시)"""
        return self.pipeline.transform(df)
    
    def save(self, path: str):
        """파이프라인 저장"""
        import joblib
        joblib.dump(self.pipeline, path)

class ImputeMissing(BaseEstimator, TransformerMixin):
    """결측치 처리"""
    def fit(self, X, y=None):
        self.fill_values_ = {}
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                self.fill_values_[col] = X[col].median()
            else:
                self.fill_values_[col] = X[col].mode()[0]
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, value in self.fill_values_.items():
            X[col].fillna(value, inplace=True)
        return X

class CreateFeatures(BaseEstimator, TransformerMixin):
    """파생 변수 생성"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 시간 기반 특성
        X['hour'] = pd.to_datetime(X['timestamp']).dt.hour
        X['day_of_week'] = pd.to_datetime(X['timestamp']).dt.dayofweek
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        
        # 사용자 행동 특성
        X['action_count'] = X.groupby('user_id')['action'].transform('count')
        X['avg_session_length'] = X.groupby('user_id')['session_length'].transform('mean')
        
        return X
```

---

## 2. Training Pipeline

### 학습 자동화

```python
from typing import Dict, Any
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.best_metric = float('inf')
    
    def run(self, train_data, val_data):
        """전체 학습 파이프라인 실행"""
        # 1. MLflow 실험 시작
        mlflow.set_experiment(self.config['experiment_name'])
        
        with mlflow.start_run():
            # 2. 하이퍼파라미터 로깅
            mlflow.log_params(self.config)
            
            # 3. 모델 초기화
            self.model = self.build_model()
            
            # 4. 학습
            history = self.train(train_data, val_data)
            
            # 5. 평가
            metrics = self.evaluate(val_data)
            mlflow.log_metrics(metrics)
            
            # 6. 모델 저장 (best만)
            if metrics['val_loss'] < self.best_metric:
                self.save_model(self.model)
                mlflow.pytorch.log_model(self.model, "model")
                self.best_metric = metrics['val_loss']
            
            # 7. 아티팩트 저장
            mlflow.log_artifact("training_history.json")
            mlflow.log_artifact("confusion_matrix.png")
            
            return metrics
    
    def build_model(self) -> nn.Module:
        """모델 생성"""
        model = YourModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        )
        return model
    
    def train(self, train_data, val_data) -> Dict:
        """학습 루프"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_data:
                optimizer.zero_grad()
                
                outputs = self.model(batch['features'])
                loss = criterion(outputs, batch['labels'])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self.validate(val_data, criterion)
            
            history['train_loss'].append(train_loss / len(train_data))
            history['val_loss'].append(val_loss)
            
            # MLflow 로깅
            mlflow.log_metrics({
                'train_loss': train_loss / len(train_data),
                'val_loss': val_loss
            }, step=epoch)
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        return history
    
    def validate(self, val_data, criterion) -> float:
        """검증"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_data:
                outputs = self.model(batch['features'])
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item()
        
        return val_loss / len(val_data)
    
    def evaluate(self, test_data) -> Dict:
        """최종 평가"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_data:
                outputs = self.model(batch['features'])
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        return metrics
    
    def save_model(self, model):
        """모델 저장"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
        }, 'model.pt')

# 사용
config = {
    'experiment_name': 'user_behavior_prediction',
    'input_dim': 128,
    'hidden_dim': 256,
    'output_dim': 3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32
}

pipeline = TrainingPipeline(config)
metrics = pipeline.run(train_loader, val_loader)
```

---

## 3. Model Registry

### 모델 버저닝

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        description: str = None
    ):
        """모델 등록"""
        # 모델 등록
        model_uri = f"runs:/{run_id}/model"
        
        mv = mlflow.register_model(
            model_uri,
            model_name
        )
        
        # 설명 추가
        if description:
            self.client.update_model_version(
                name=model_name,
                version=mv.version,
                description=description
            )
        
        print(f"Registered {model_name} version {mv.version}")
        return mv
    
    def promote_to_production(self, model_name: str, version: int):
        """프로덕션으로 승격"""
        # 기존 production 모델 → archived
        for mv in self.client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Production":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
        
        # 새 버전 → production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"Promoted {model_name} v{version} to Production")
    
    def get_production_model(self, model_name: str):
        """프로덕션 모델 가져오기"""
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        return model
    
    def compare_models(self, model_name: str, versions: List[int]):
        """모델 버전 비교"""
        results = []
        
        for version in versions:
            # 모델 메트릭 가져오기
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)
            
            results.append({
                'version': version,
                'stage': mv.current_stage,
                'metrics': run.data.metrics,
                'created': mv.creation_timestamp
            })
        
        return pd.DataFrame(results)

# 사용
registry = ModelRegistry(tracking_uri="http://localhost:5000")

# 모델 등록
mv = registry.register_model(
    model_name="user_behavior_model",
    run_id="abc123",
    description="Transformer-based model with 95% accuracy"
)

# 프로덕션 승격
registry.promote_to_production("user_behavior_model", version=5)

# 모델 비교
comparison = registry.compare_models("user_behavior_model", versions=[3, 4, 5])
print(comparison)
```

---

## 4. Deployment Pipeline

### CI/CD for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 매일 2am 재학습

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Validate Data
        run: |
          python scripts/validate_data.py
          
      - name: Upload validation report
        uses: actions/upload-artifact@v2
        with:
          name: validation-report
          path: validation_report.html

  training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Train Model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          python scripts/train.py --config configs/production.yaml
      
      - name: Evaluate Model
        run: |
          python scripts/evaluate.py
      
      - name: Check metrics threshold
        run: |
          python scripts/check_threshold.py --min-accuracy 0.90

  deployment:
    needs: training
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl apply -f k8s/staging/
      
      - name: Run integration tests
        run: |
          python scripts/integration_test.py --env staging
      
      - name: Deploy to production
        if: success()
        run: |
          kubectl apply -f k8s/production/
```

### Kubernetes 배포

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
        version: v5
    spec:
      containers:
      - name: model-server
        image: your-registry/ml-model:v5
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "user_behavior_model"
        - name: MODEL_VERSION
          value: "5"
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: tracking-uri
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 5. 전체 파이프라인 오케스트레이션

### Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Daily ML model training pipeline',
    schedule_interval='0 2 * * *',  # 매일 2am
    catchup=False
)

# Task 1: 데이터 수집
collect_data = PythonOperator(
    task_id='collect_data',
    python_callable=collect_daily_data,
    op_kwargs={'date': '{{ ds }}'},
    dag=dag
)

# Task 2: 데이터 검증
validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_quality,
    dag=dag
)

# Task 3: Feature Engineering
create_features = PythonOperator(
    task_id='create_features',
    python_callable=create_feature_pipeline,
    dag=dag
)

# Task 4: 모델 학습
train_model = BashOperator(
    task_id='train_model',
    bash_command='python scripts/train.py --date {{ ds }}',
    dag=dag
)

# Task 5: 모델 평가
evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_and_register,
    dag=dag
)

# Task 6: 배포 (조건부)
deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_if_better,
    dag=dag
)

# 의존성 설정
collect_data >> validate_data >> create_features >> train_model >> evaluate_model >> deploy_model
```

---

## 요약

**ML 파이프라인 핵심:**

1. **Data Pipeline**: 수집 → 검증 → 특성 추출
2. **Training Pipeline**: 학습 → 평가 → 로깅
3. **Model Registry**: 버저닝 → 비교 → 승격
4. **Deployment**: CI/CD → K8s → 모니터링
5. **Orchestration**: Airflow로 자동화

**핵심 도구:**
- MLflow: 실험 추적, 모델 레지스트리
- Great Expectations: 데이터 검증
- Airflow: 파이프라인 오케스트레이션
- Kubernetes: 배포 및 스케일링

**Best Practices:**
- 모든 것을 코드로 (Infrastructure as Code)
- 버전 관리 (데이터, 모델, 코드)
- 자동화 (CI/CD)
- 모니터링 필수

**다음 글:**
- **MLOps #2**: 실험 관리 (MLflow, W&B 심화)
- **MLOps #3**: 모니터링 & 알람
- **Serving #1**: 추론 최적화

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
