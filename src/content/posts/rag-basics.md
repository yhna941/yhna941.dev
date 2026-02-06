---
title: "RAG #1: 기초 - Retrieval-Augmented Generation 완전 정복"
description: "LLM의 한계를 극복하는 RAG의 원리와 구현을 처음부터 끝까지 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["rag", "retrieval", "llm", "vector-search", "embeddings"]
draft: false
---

# RAG #1: 기초

**"LLM에게 지식을 주입하라"**

문제:
```
User: "2024년 3분기 매출이 얼마야?"
LLM: "죄송합니다. 2024년 데이터는 학습되지 않았습니다."

해결 (RAG):
User: "2024년 3분기 매출이 얼마야?"
→ [검색] → 관련 문서 찾기
→ [생성] → "2024년 3분기 매출은 120억원입니다."
```

---

## RAG란?

### 정의

> **Retrieval-Augmented Generation**  
> = 검색(Retrieval) + 생성(Generation)

**기본 아이디어:**
```
1. 질문 받음
2. 관련 문서 검색 (Retrieval)
3. 문서 + 질문을 LLM에 입력
4. 답변 생성 (Generation)
```

### 왜 필요?

**LLM의 한계:**

1. **학습 시점 이후 데이터 모름**
   ```
   GPT-4 (2023년 4월 학습)
   → 2024년 뉴스? 모름
   → 오늘 날씨? 모름
   ```

2. **회사 내부 문서 모름**
   ```
   "우리 회사 휴가 정책은?"
   → LLM: 모름 (학습 안 됨)
   ```

3. **Hallucination (환각)**
   ```
   "2024년 대통령은?"
   → LLM: "김철수입니다" (지어냄!)
   ```

**RAG로 해결:**
- 최신 데이터 접근
- 회사 문서 활용
- 근거 기반 답변 (환각 감소)

---

## RAG 아키텍처

### 전체 흐름

```
┌─────────────────────────────────────┐
│          Indexing (오프라인)         │
├─────────────────────────────────────┤
│                                     │
│  Documents → Chunking → Embedding  │
│      ↓                      ↓       │
│  PDF, TXT, ...         Vector DB    │
│                       (Pinecone,    │
│                        Weaviate)    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         Retrieval (온라인)           │
├─────────────────────────────────────┤
│                                     │
│  User Query                         │
│      ↓                              │
│  Query Embedding                    │
│      ↓                              │
│  Vector Search (Top-K)              │
│      ↓                              │
│  Retrieved Documents                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         Generation (온라인)          │
├─────────────────────────────────────┤
│                                     │
│  Query + Retrieved Docs             │
│      ↓                              │
│  Prompt Engineering                 │
│      ↓                              │
│  LLM (GPT-4, Claude)                │
│      ↓                              │
│  Final Answer                       │
└─────────────────────────────────────┘
```

---

## 1. Document Indexing

### 문서 로딩

```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader
)

# PDF
loader = PyPDFLoader("company_policy.pdf")
documents = loader.load()

# 웹페이지
loader = WebBaseLoader("https://example.com/docs")
documents = loader.load()

# 여러 파일
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./docs",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
```

### 문서 분할 (Chunking)

**왜 분할?**
```
전체 문서 (10,000 토큰)
→ LLM context 제한 (4,096 토큰)
→ 작은 청크로 나눔 (512 토큰씩)
```

**전략:**

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

# 1. 문자 기반 (가장 일반적)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 청크 크기
    chunk_overlap=200,      # 오버랩 (문맥 유지)
    separators=["\n\n", "\n", " ", ""]  # 분할 우선순위
)

chunks = text_splitter.split_documents(documents)

# 2. 토큰 기반 (정확한 토큰 제어)
token_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

chunks = token_splitter.split_documents(documents)

# 결과
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}:")
    print(f"  Length: {len(chunk.page_content)}")
    print(f"  Content: {chunk.page_content[:100]}...")
    print(f"  Metadata: {chunk.metadata}")
```

**청크 오버랩 중요!**

```
청크 1: "...기업 휴가 정책은 연차 15일입니다."
              ↑ 오버랩 영역 ↓
청크 2: "연차 15일입니다. 병가는 별도로..."

→ 문맥이 끊기지 않음!
```

### 임베딩 (Embedding)

**개념:**
```
텍스트 → 벡터 (숫자 배열)

"강아지가 귀엽다" → [0.2, 0.8, -0.3, ..., 0.5]  (1536차원)
"개가 사랑스럽다" → [0.3, 0.7, -0.2, ..., 0.6]  (유사!)

"자동차가 빠르다" → [-0.5, 0.1, 0.9, ..., -0.2] (다름)
```

**구현:**

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# 1. OpenAI Embeddings (text-embedding-3-small)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="sk-..."
)

# 2. 오픈소스 (무료!)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 임베딩 생성
text = "강아지가 귀엽다"
vector = embeddings.embed_query(text)

print(f"Dimension: {len(vector)}")  # 384 or 1536
print(f"Vector: {vector[:5]}...")   # [0.123, -0.456, ...]
```

**임베딩 모델 비교:**

| 모델 | 차원 | 속도 | 품질 | 비용 |
|------|------|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 빠름 | 우수 | 유료 |
| OpenAI text-embedding-3-large | 3072 | 느림 | 최고 | 비싸 |
| all-MiniLM-L6-v2 | 384 | 매우 빠름 | 괜찮음 | 무료 |
| bge-large-en | 1024 | 보통 | 우수 | 무료 |

### Vector Database 저장

```python
from langchain.vectorstores import FAISS, Chroma, Pinecone

# 1. FAISS (로컬, 빠름)
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 저장
vectorstore.save_local("faiss_index")

# 로드
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 2. Chroma (로컬, 간편)
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. Pinecone (클라우드, 프로덕션)
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

index_name = "company-docs"

vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)
```

---

## 2. Retrieval

### 유사도 검색

```python
# 질문
query = "휴가는 며칠이야?"

# 검색 (Top-3)
results = vectorstore.similarity_search(
    query,
    k=3  # 상위 3개
)

for i, doc in enumerate(results):
    print(f"\n=== Result {i+1} ===")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### 스코어 포함 검색

```python
# 유사도 스코어 포함
results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
```

### MMR (Maximum Marginal Relevance)

**문제:** 유사도 검색은 중복된 결과 반환

```
질문: "휴가 정책"
결과:
1. "휴가는 15일입니다."
2. "휴가는 연차 15일입니다." ← 중복!
3. "휴가는 15일 제공됩니다." ← 또 중복!
```

**MMR:** 유사하면서도 다양한 결과

```python
# MMR 검색
results = vectorstore.max_marginal_relevance_search(
    query,
    k=3,
    fetch_k=10,      # 먼저 10개 후보 가져오기
    lambda_mult=0.5  # 0=다양성, 1=유사성
)

# 결과가 더 다양해짐!
```

---

## 3. Generation

### Prompt 구성

```python
from langchain.prompts import PromptTemplate

template = """당신은 회사 정책을 안내하는 AI 어시스턴트입니다.

아래 문서를 참고하여 질문에 답변하세요.
문서에 없는 내용은 "정보가 없습니다"라고 답하세요.

문서:
{context}

질문: {question}

답변:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

### 전체 RAG 체인

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# LLM
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0  # 일관된 답변
)

# RAG 체인
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 모든 문서를 하나로 합침
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True  # 출처 반환
)

# 질문
query = "연차는 며칠이야?"
result = qa_chain({"query": query})

print(f"답변: {result['result']}")
print(f"\n출처:")
for doc in result['source_documents']:
    print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
```

### Chain Types

**1. Stuff (기본)**

```python
# 모든 문서를 하나로 합쳐서 전달
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
prompt = f"Context: {context}\n\nQuestion: {query}"
```

- 장점: 간단, 빠름
- 단점: 문서 많으면 context 초과

**2. Map-Reduce**

```python
# 각 문서마다 답변 → 결합
for doc in retrieved_docs:
    partial_answer = llm(f"Context: {doc}\nQuestion: {query}")

final_answer = llm(f"Combine these: {partial_answers}")
```

- 장점: 많은 문서 처리 가능
- 단점: LLM 호출 많음 (비용↑)

**3. Refine**

```python
# 순차적으로 답변 개선
answer = llm(f"Context: {doc1}\nQuestion: {query}")
answer = llm(f"Previous: {answer}\nNew context: {doc2}\nRefine:")
...
```

- 장점: 점진적 개선
- 단점: 순차 처리 (느림)

---

## 4. 완전한 RAG 구현

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGSystem:
    def __init__(self, docs_path, persist_dir="./faiss_index"):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.qa_chain = None
    
    def index_documents(self):
        """문서 인덱싱"""
        print("📚 Loading documents...")
        
        # 1. 문서 로드
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.md",
            show_progress=True
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        # 2. 청크 분할
        print("✂️ Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # 3. 임베딩 & 저장
        print("🔢 Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        
        self.vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )
        
        # 4. 디스크에 저장
        self.vectorstore.save_local(self.persist_dir)
        print(f"✅ Saved to {self.persist_dir}")
    
    def load_index(self):
        """저장된 인덱스 로드"""
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(
            self.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded index from {self.persist_dir}")
    
    def setup_qa_chain(self):
        """QA 체인 설정"""
        llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",  # MMR 사용
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        print("✅ QA chain ready")
    
    def query(self, question: str):
        """질문 답변"""
        if not self.qa_chain:
            self.setup_qa_chain()
        
        print(f"\n❓ Question: {question}")
        
        result = self.qa_chain({"query": question})
        
        print(f"\n💡 Answer: {result['result']}")
        print(f"\n📄 Sources:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"   {doc.page_content[:200]}...")
        
        return result

# 사용
rag = RAGSystem("./company_docs")

# 첫 실행: 인덱싱
# rag.index_documents()

# 이후: 로드만
rag.load_index()

# 질문
rag.query("휴가는 며칠이야?")
rag.query("재택근무 정책은?")
```

---

## 5. 고급 기법

### 하이브리드 검색 (Keyword + Semantic)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword 검색 (BM25)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Semantic 검색 (Vector)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 결합 (Ensemble)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 동등한 가중치
)

# 검색
results = ensemble_retriever.get_relevant_documents(query)
```

### Re-ranking

```python
from sentence_transformers import CrossEncoder

# 1차 검색 (Top-10)
candidates = vectorstore.similarity_search(query, k=10)

# 2차 re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

pairs = [[query, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)

# 스코어 순으로 정렬
ranked_docs = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

# Top-3
top_docs = [doc for doc, score in ranked_docs[:3]]
```

---

## 평가 (Evaluation)

### Retrieval 평가

```python
from ragas.metrics import context_precision, context_recall

# Ground truth
ground_truth = [
    {
        'question': '휴가는 며칠?',
        'ground_truth': '연차 15일',
        'retrieved_contexts': [doc.page_content for doc in retrieved_docs]
    }
]

# 평가
results = evaluate(
    ground_truth,
    metrics=[context_precision, context_recall]
)

print(f"Precision: {results['context_precision']}")
print(f"Recall: {results['context_recall']}")
```

### Generation 평가

```python
# Faithfulness (충실성)
# 답변이 문서에 근거했는가?

# Answer Relevance
# 답변이 질문에 적절한가?

from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(
    test_dataset,
    metrics=[faithfulness, answer_relevancy]
)
```

---

## 요약

**RAG 파이프라인:**

1. **Indexing**: 문서 → 청크 → 임베딩 → Vector DB
2. **Retrieval**: 질문 → 유사도 검색 → Top-K 문서
3. **Generation**: 문서 + 질문 → LLM → 답변

**핵심 컴포넌트:**
- Document Loader
- Text Splitter
- Embedding Model
- Vector Database
- LLM

**장점:**
- 최신 정보 활용
- 환각 감소
- 출처 제공

**다음 글:**
- **RAG #2**: Production RAG (성능 최적화, 캐싱, 모니터링)
- **RAG #3**: Advanced RAG (Query Rewriting, HyDE, Self-RAG)

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
