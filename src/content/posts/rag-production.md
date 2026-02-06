---
title: "RAG #2: Production RAG - 실전 최적화와 스케일링"
description: "프로덕션 환경에서 RAG 시스템을 안정적이고 빠르게 운영하는 모든 것"
pubDate: 2026-02-06
author: "Yh Na"
tags: ["rag", "production", "optimization", "caching", "monitoring"]
draft: false
---

# RAG #2: Production RAG

**"프로토타입에서 프로덕션으로"**

프로토타입:
```python
vectorstore.similarity_search(query, k=3)
# 응답 시간: 2초
# 비용: $0.05/query
```

프로덕션:
```python
# 캐싱 + 최적화 + 모니터링
# 응답 시간: 200ms (10배 빠름!)
# 비용: $0.005/query (10배 저렴!)
```

---

## Production RAG 요구사항

### 1. 성능

```
목표:
- Retrieval: < 100ms
- Generation: < 1s
- 전체 응답: < 2s
```

### 2. 비용

```
1M queries/month 기준:
- 임베딩: $0.0001/1K 토큰
- Vector DB: $50/month
- LLM: $0.001/1K 토큰
→ 총 비용 최적화 필요!
```

### 3. 정확도

```
- 검색 정확도 > 90%
- 답변 품질 > 95%
- 환각률 < 5%
```

### 4. 안정성

```
- Uptime: 99.9%
- 에러율: < 0.1%
- 모니터링 & 알람
```

---

## 1. Retrieval 최적화

### Hybrid Search (키워드 + 시맨틱)

**문제:** Vector search만으로는 부족

```
질문: "GPT-4 가격"
Vector 검색: "AI 모델의 비용..." ← 일반적
Keyword 검색: "GPT-4: $0.03/1K tokens" ← 정확!

→ 둘 다 필요!
```

**구현:**

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents, vectorstore, alpha=0.5):
        """
        alpha: 0.0 (keyword only) ~ 1.0 (vector only)
        """
        self.vectorstore = vectorstore
        self.alpha = alpha
        
        # BM25 (keyword) retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10
        
        # Vector retriever
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )
        
        # Ensemble
        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[1-alpha, alpha]
        )
    
    def retrieve(self, query: str, k: int = 3):
        """하이브리드 검색"""
        results = self.retriever.get_relevant_documents(query)
        return results[:k]

# 사용
retriever = HybridRetriever(chunks, vectorstore, alpha=0.7)
docs = retriever.retrieve("GPT-4 가격", k=3)
```

### Re-ranking

**2단계 검색:**
```
1. Fast retrieval (Top-100)
   - BM25 + Vector
   - 빠르지만 덜 정확

2. Re-ranking (Top-10 → Top-3)
   - Cross-encoder
   - 느리지만 매우 정확
```

**구현:**

```python
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """문서 re-ranking"""
        # 쿼리-문서 쌍 생성
        pairs = [[query, doc] for doc in documents]
        
        # Cross-encoder 스코어
        scores = self.model.predict(pairs)
        
        # 정렬
        ranked_indices = np.argsort(scores)[::-1]
        
        results = [
            (documents[i], scores[i])
            for i in ranked_indices[:top_k]
        ]
        
        return results

# 사용
reranker = Reranker()

# 1차: Hybrid search (Top-100)
candidates = hybrid_retriever.retrieve(query, k=100)
candidate_texts = [doc.page_content for doc in candidates]

# 2차: Re-rank (Top-3)
top_docs = reranker.rerank(query, candidate_texts, top_k=3)

for doc, score in top_docs:
    print(f"Score: {score:.4f}")
    print(f"Doc: {doc[:100]}...")
```

### Query Expansion

**문제:** 짧은 질문은 검색 어려움

```
질문: "가격"
→ 너무 모호!

Query Expansion:
"가격" → ["가격", "비용", "요금", "pricing", "cost"]
→ 검색 개선!
```

**구현:**

```python
from typing import List

class QueryExpander:
    def __init__(self, llm):
        self.llm = llm
    
    def expand(self, query: str, n: int = 3) -> List[str]:
        """질문 확장"""
        prompt = f"""Generate {n} alternative phrasings of this query:

Original: {query}

Alternatives:
1."""
        
        response = self.llm.generate(prompt)
        
        # 파싱
        alternatives = [line.strip() for line in response.split('\n') if line.strip()]
        
        return [query] + alternatives[:n]
    
    def multi_query_retrieve(
        self,
        query: str,
        retriever,
        k: int = 3
    ) -> List[Document]:
        """여러 쿼리로 검색 후 합침"""
        # 쿼리 확장
        queries = self.expand(query, n=3)
        
        # 각 쿼리로 검색
        all_docs = []
        for q in queries:
            docs = retriever.retrieve(q, k=k)
            all_docs.extend(docs)
        
        # 중복 제거 (content 기반)
        unique_docs = []
        seen = set()
        
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
        
        return unique_docs[:k]

# 사용
expander = QueryExpander(llm)
docs = expander.multi_query_retrieve("가격", retriever, k=3)
```

---

## 2. Caching

### Query 캐싱

```python
import hashlib
from functools import lru_cache
import redis

class RAGCache:
    def __init__(self, redis_client=None):
        self.redis = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        self.ttl = 3600  # 1시간
    
    def get_cache_key(self, query: str) -> str:
        """쿼리 해시"""
        return f"rag:query:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, query: str):
        """캐시 조회"""
        key = self.get_cache_key(query)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query: str, result: dict):
        """캐시 저장"""
        key = self.get_cache_key(query)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(result, ensure_ascii=False)
        )
    
    def query_with_cache(self, query: str, rag_chain) -> dict:
        """캐시된 답변 또는 새로 생성"""
        # 1. 캐시 확인
        cached = self.get(query)
        if cached:
            print("✅ Cache hit!")
            return cached
        
        # 2. RAG 실행
        print("🔄 Cache miss - generating...")
        result = rag_chain({"query": query})
        
        # 3. 캐시 저장
        self.set(query, result)
        
        return result

# 사용
cache = RAGCache()

result = cache.query_with_cache("휴가는 며칠?", qa_chain)
# 첫 실행: 2초
result = cache.query_with_cache("휴가는 며칠?", qa_chain)
# 두 번째: 10ms! (200배 빠름)
```

### Semantic 캐싱

```python
class SemanticCache:
    """의미적으로 유사한 질문도 캐싱"""
    
    def __init__(self, vectorstore, threshold=0.95):
        self.vectorstore = vectorstore
        self.threshold = threshold
        self.cache = {}  # {query_id: result}
    
    def find_similar_query(self, query: str):
        """유사한 캐시된 질문 찾기"""
        # 벡터 검색
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=1
        )
        
        if not results:
            return None
        
        doc, score = results[0]
        
        # 임계값 이상이면 캐시 히트
        if score >= self.threshold:
            query_id = doc.metadata['query_id']
            return self.cache.get(query_id)
        
        return None
    
    def add_to_cache(self, query: str, result: dict):
        """캐시에 추가"""
        query_id = str(uuid.uuid4())
        
        # 결과 저장
        self.cache[query_id] = result
        
        # 쿼리를 vectorstore에 저장
        doc = Document(
            page_content=query,
            metadata={'query_id': query_id}
        )
        self.vectorstore.add_documents([doc])
    
    def query(self, query: str, rag_chain):
        """시맨틱 캐싱"""
        # 유사 질문 확인
        cached = self.find_similar_query(query)
        if cached:
            print(f"✅ Similar query found!")
            return cached
        
        # RAG 실행
        result = rag_chain({"query": query})
        
        # 캐시에 추가
        self.add_to_cache(query, result)
        
        return result

# 사용
semantic_cache = SemanticCache(cache_vectorstore)

result = semantic_cache.query("휴가는 며칠?", qa_chain)
# 두 번째 (약간 다른 질문)
result = semantic_cache.query("연차는 몇 일이야?", qa_chain)
# ✅ 캐시 히트! (의미가 같음)
```

---

## 3. Batch Processing

### Batch Embedding

```python
class BatchEmbedder:
    def __init__(self, embeddings, batch_size=100):
        self.embeddings = embeddings
        self.batch_size = batch_size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (효율적)"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # 배치 임베딩
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            print(f"Processed {i + len(batch)}/{len(texts)}")
        
        return all_embeddings

# 사용
embedder = BatchEmbedder(embeddings, batch_size=100)

# 10,000개 문서 임베딩
texts = [chunk.page_content for chunk in chunks]
vectors = embedder.embed_documents(texts)

# 단일 처리: 10분
# 배치 처리: 2분 (5배 빠름!)
```

### Batch Inference

```python
import asyncio
from typing import List

class BatchRAG:
    def __init__(self, rag_chain, batch_size=10):
        self.rag_chain = rag_chain
        self.batch_size = batch_size
    
    async def process_batch(self, queries: List[str]) -> List[dict]:
        """배치 처리"""
        tasks = [
            asyncio.to_thread(self.rag_chain, {"query": q})
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_all(self, queries: List[str]) -> List[dict]:
        """전체 쿼리 배치 처리"""
        all_results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            
            print(f"Processing batch {i//self.batch_size + 1}...")
            results = await self.process_batch(batch)
            all_results.extend(results)
        
        return all_results

# 사용
batch_rag = BatchRAG(qa_chain, batch_size=10)

queries = [
    "휴가는 며칠?",
    "재택근무 정책은?",
    # ... 1000개
]

results = asyncio.run(batch_rag.process_all(queries))
```

---

## 4. Monitoring

### 메트릭 추적

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class RAGMetrics:
    def __init__(self):
        # 카운터
        self.query_count = Counter(
            'rag_queries_total',
            'Total RAG queries'
        )
        
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Cache hits'
        )
        
        # 히스토그램
        self.retrieval_latency = Histogram(
            'rag_retrieval_latency_seconds',
            'Retrieval latency'
        )
        
        self.generation_latency = Histogram(
            'rag_generation_latency_seconds',
            'Generation latency'
        )
        
        # Gauge
        self.active_queries = Gauge(
            'rag_active_queries',
            'Active queries'
        )
    
    def track_query(self, func):
        """쿼리 추적 데코레이터"""
        def wrapper(*args, **kwargs):
            self.query_count.inc()
            self.active_queries.inc()
            
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                self.generation_latency.observe(duration)
                self.active_queries.dec()
        
        return wrapper

# 사용
metrics = RAGMetrics()

@metrics.track_query
def query_rag(query: str):
    return qa_chain({"query": query})

# Prometheus endpoint
from prometheus_client import start_http_server

start_http_server(8000)  # http://localhost:8000/metrics
```

### 로깅

```python
import logging
import json
from datetime import datetime

class RAGLogger:
    def __init__(self, log_file='rag.log'):
        self.logger = logging.getLogger('rag')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.logger.addHandler(handler)
    
    def log_query(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str,
        latency: float,
        cached: bool = False
    ):
        """쿼리 로그"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'retrieved_docs': [
                {
                    'content': doc.page_content[:200],
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ],
            'answer': answer,
            'latency_ms': latency * 1000,
            'cached': cached
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_error(self, query: str, error: Exception):
        """에러 로그"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'error': str(error),
            'type': type(error).__name__
        }
        
        self.logger.error(json.dumps(log_entry, ensure_ascii=False))

# 사용
logger = RAGLogger()

try:
    start = time.time()
    result = qa_chain({"query": query})
    latency = time.time() - start
    
    logger.log_query(
        query=query,
        retrieved_docs=result['source_documents'],
        answer=result['result'],
        latency=latency
    )
except Exception as e:
    logger.log_error(query, e)
```

---

## 5. Error Handling

### Retry 전략

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RobustRAG:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    def query_with_retry(self, query: str) -> dict:
        """재시도 로직"""
        try:
            return self.qa_chain({"query": query})
        except Exception as e:
            print(f"Error: {e}, retrying...")
            raise
    
    def query_with_fallback(self, query: str) -> dict:
        """Fallback 전략"""
        try:
            # 1차: 정상 RAG
            return self.query_with_retry(query)
        
        except Exception as e:
            print(f"RAG failed: {e}")
            
            # 2차: 캐시된 유사 답변
            cached = self.get_similar_cached_answer(query)
            if cached:
                return cached
            
            # 3차: 기본 LLM (RAG 없이)
            return self.fallback_to_llm(query)
    
    def fallback_to_llm(self, query: str) -> dict:
        """RAG 없이 LLM만"""
        answer = self.llm.generate(query)
        return {
            'result': answer,
            'source_documents': [],
            'fallback': True
        }

# 사용
robust_rag = RobustRAG(qa_chain)
result = robust_rag.query_with_fallback("휴가는 며칠?")
```

---

## 6. Cost Optimization

### Token 절약

```python
class CostOptimizedRAG:
    def __init__(self, qa_chain, max_context_tokens=2000):
        self.qa_chain = qa_chain
        self.max_context_tokens = max_context_tokens
    
    def truncate_context(
        self,
        docs: List[Document]
    ) -> List[Document]:
        """문맥을 토큰 제한에 맞춤"""
        import tiktoken
        
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        truncated = []
        total_tokens = 0
        
        for doc in docs:
            tokens = enc.encode(doc.page_content)
            
            if total_tokens + len(tokens) > self.max_context_tokens:
                # 남은 토큰만큼만 추가
                remaining = self.max_context_tokens - total_tokens
                truncated_content = enc.decode(tokens[:remaining])
                
                truncated.append(Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                ))
                break
            
            truncated.append(doc)
            total_tokens += len(tokens)
        
        return truncated
    
    def query(self, query: str) -> dict:
        """비용 최적화 쿼리"""
        # 문서 검색
        docs = self.retriever.get_relevant_documents(query)
        
        # 토큰 제한
        docs = self.truncate_context(docs)
        
        # 생성
        return self.qa_chain({
            "query": query,
            "context": "\n\n".join([d.page_content for d in docs])
        })

# 절약 효과
# Before: 4000 tokens → $0.004
# After:  2000 tokens → $0.002 (50% 절감!)
```

### Streaming

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class StreamingRAG:
    def __init__(self, llm):
        self.llm = llm
    
    async def stream_answer(self, query: str, context: str):
        """스트리밍 답변"""
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        
        # 스트리밍
        async for chunk in self.llm.astream(prompt):
            yield chunk
            
# 사용
async def main():
    streaming_rag = StreamingRAG(llm)
    
    async for chunk in streaming_rag.stream_answer(query, context):
        print(chunk, end='', flush=True)

# 사용자 경험 개선:
# - 전체 대기 시간 동일
# - 첫 토큰까지 시간 단축 (perceived latency ↓)
```

---

## 7. A/B 테스팅

```python
import random

class ABTestRAG:
    def __init__(self, rag_v1, rag_v2, split_ratio=0.5):
        self.rag_v1 = rag_v1
        self.rag_v2 = rag_v2
        self.split_ratio = split_ratio
        
        self.metrics = {
            'v1': {'count': 0, 'latency': [], 'feedback': []},
            'v2': {'count': 0, 'latency': [], 'feedback': []}
        }
    
    def query(self, query: str, user_id: str) -> dict:
        """A/B 테스트"""
        # 사용자별 일관된 버전 선택
        version = 'v1' if hash(user_id) % 100 < self.split_ratio * 100 else 'v2'
        
        start = time.time()
        
        if version == 'v1':
            result = self.rag_v1({"query": query})
        else:
            result = self.rag_v2({"query": query})
        
        latency = time.time() - start
        
        # 메트릭 기록
        self.metrics[version]['count'] += 1
        self.metrics[version]['latency'].append(latency)
        
        result['version'] = version
        return result
    
    def log_feedback(self, version: str, feedback: int):
        """피드백 기록 (1-5점)"""
        self.metrics[version]['feedback'].append(feedback)
    
    def get_stats(self):
        """통계"""
        stats = {}
        
        for version in ['v1', 'v2']:
            m = self.metrics[version]
            
            stats[version] = {
                'count': m['count'],
                'avg_latency': np.mean(m['latency']) if m['latency'] else 0,
                'avg_feedback': np.mean(m['feedback']) if m['feedback'] else 0
            }
        
        return stats

# 사용
ab_test = ABTestRAG(rag_v1, rag_v2, split_ratio=0.5)

result = ab_test.query("휴가는 며칠?", user_id="user123")
print(f"Version: {result['version']}")

# 피드백
ab_test.log_feedback(result['version'], feedback=5)

# 통계
print(ab_test.get_stats())
# v1: avg_latency=1.2s, avg_feedback=4.2
# v2: avg_latency=0.8s, avg_feedback=4.5
# → v2 승!
```

---

## 요약

**Production RAG 핵심:**

1. **성능 최적화**
   - Hybrid search (BM25 + Vector)
   - Re-ranking
   - Query expansion

2. **Caching**
   - Query 캐싱 (200배 빠름)
   - Semantic 캐싱

3. **Batch Processing**
   - Batch embedding
   - Batch inference

4. **Monitoring**
   - Prometheus 메트릭
   - 상세 로깅

5. **Error Handling**
   - Retry 전략
   - Fallback

6. **Cost Optimization**
   - Token 절약
   - Streaming

7. **A/B Testing**
   - 버전 비교
   - 메트릭 추적

**결과:**
- 응답 시간: 2s → 200ms (10배)
- 비용: $0.05 → $0.005 (10배)
- 정확도: 85% → 95% (개선)

**다음 글:**
- **RAG #3**: Advanced RAG (HyDE, Self-RAG, CRAG)

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
