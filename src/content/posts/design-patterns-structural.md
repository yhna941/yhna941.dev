---
title: "Design Patterns #2: 구조 패턴 - Adapter, Decorator, Proxy"
description: "클래스와 객체를 조합하여 더 큰 구조를 만드는 구조 디자인 패턴을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["design-patterns", "oop", "software-engineering", "python", "structural-patterns"]
draft: false
---

# Design Patterns #2: 구조 패턴

**"기존 코드를 어떻게 재사용할 것인가?"**

구조 패턴 (Structural Patterns)은:
- 기존 클래스 수정 없이
- 새로운 기능 추가
- 인터페이스 호환
- 유연한 구조

---

## Adapter Pattern

### 문제

**호환되지 않는 인터페이스:**

```python
# 기존 코드
class OldPaymentSystem:
    def make_payment(self, amount):
        print(f"Old system: ${amount}")

# 새 라이브러리
class NewPaymentGateway:
    def process_transaction(self, money):
        print(f"New gateway: ${money}")

# 클라이언트는 Old 인터페이스 기대
def checkout(payment_system, amount):
    payment_system.make_payment(amount)  # Error with NewPaymentGateway!
```

### 해결

**Adapter로 변환:**

```python
class PaymentAdapter:
    """Old → New 변환"""
    def __init__(self, new_gateway):
        self.gateway = new_gateway
    
    def make_payment(self, amount):
        # Old interface → New interface
        self.gateway.process_transaction(amount)

# 사용
old_system = OldPaymentSystem()
checkout(old_system, 100)  # OK

new_gateway = NewPaymentGateway()
adapted = PaymentAdapter(new_gateway)
checkout(adapted, 100)  # OK!
```

### Class Adapter (상속)

```python
class PaymentClassAdapter(NewPaymentGateway):
    """다중 상속 사용"""
    def make_payment(self, amount):
        self.process_transaction(amount)

# 사용
adapter = PaymentClassAdapter()
checkout(adapter, 100)
```

### 실전 예제: Database Driver

```python
from abc import ABC, abstractmethod

# Target interface
class DatabaseInterface(ABC):
    @abstractmethod
    def connect(self, host, port):
        pass
    
    @abstractmethod
    def query(self, sql):
        pass

# Adaptee (기존 MongoDB driver)
class MongoDBDriver:
    def __init__(self):
        self.client = None
    
    def establish_connection(self, uri):
        print(f"MongoDB connected: {uri}")
        self.client = "mongo_client"
    
    def find(self, collection, filter):
        print(f"MongoDB find: {collection}, {filter}")
        return []

# Adapter
class MongoDBAdapter(DatabaseInterface):
    def __init__(self):
        self.driver = MongoDBDriver()
    
    def connect(self, host, port):
        uri = f"mongodb://{host}:{port}"
        self.driver.establish_connection(uri)
    
    def query(self, sql):
        # SQL → MongoDB query 변환 (간단 예제)
        if sql.startswith("SELECT"):
            collection = sql.split("FROM")[1].strip()
            return self.driver.find(collection, {})
        raise NotImplementedError("Complex SQL not supported")

# 클라이언트는 SQL 인터페이스만 알면 됨
def run_query(db: DatabaseInterface, sql):
    db.query(sql)

# PostgreSQL
postgres = PostgreSQLDriver()
postgres.connect("localhost", 5432)
run_query(postgres, "SELECT * FROM users")

# MongoDB (adapted)
mongo = MongoDBAdapter()
mongo.connect("localhost", 27017)
run_query(mongo, "SELECT * FROM users")
```

---

## Decorator Pattern

### 문제

**기능 동적 추가:**

```python
# Bad: 상속으로 모든 조합
class Coffee: pass
class CoffeeWithMilk(Coffee): pass
class CoffeeWithSugar(Coffee): pass
class CoffeeWithMilkAndSugar(Coffee): pass  # 조합 폭발!
```

### 해결

**Decorator로 감싸기:**

```python
from abc import ABC, abstractmethod

# Component
class Coffee(ABC):
    @abstractmethod
    def cost(self):
        pass
    
    @abstractmethod
    def description(self):
        pass

# Concrete Component
class SimpleCoffee(Coffee):
    def cost(self):
        return 5
    
    def description(self):
        return "Simple coffee"

# Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()

# Concrete Decorators
class MilkDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 2
    
    def description(self):
        return self._coffee.description() + ", milk"

class SugarDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + ", sugar"

class WhipDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 3
    
    def description(self):
        return self._coffee.description() + ", whip"

# 사용 (동적 조합!)
coffee = SimpleCoffee()
print(f"{coffee.description()}: ${coffee.cost()}")
# Simple coffee: $5

coffee = MilkDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# Simple coffee, milk: $7

coffee = SugarDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# Simple coffee, milk, sugar: $8

coffee = WhipDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# Simple coffee, milk, sugar, whip: $11
```

### Python의 함수 Decorator

```python
import time
import functools

def timer(func):
    """실행 시간 측정"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

def cache(func):
    """결과 캐싱"""
    cached_results = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args in cached_results:
            print(f"Cache hit: {args}")
            return cached_results[args]
        
        result = func(*args)
        cached_results[args] = result
        return result
    return wrapper

def log(func):
    """로깅"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# 사용 (decorator stacking)
@timer
@cache
@log
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 첫 호출
result = fibonacci(10)
# 두번째 호출 (캐시됨)
result = fibonacci(10)
```

### 실전 예제: API Middleware

```python
class APIEndpoint:
    def handle(self, request):
        return {"data": "response"}

class APIMiddleware:
    """Base decorator"""
    def __init__(self, endpoint):
        self._endpoint = endpoint
    
    def handle(self, request):
        return self._endpoint.handle(request)

class AuthenticationMiddleware(APIMiddleware):
    def handle(self, request):
        # Before
        token = request.get("token")
        if not token or not self._verify_token(token):
            return {"error": "Unauthorized"}
        
        # Proceed
        response = self._endpoint.handle(request)
        
        # After
        return response
    
    def _verify_token(self, token):
        return token == "valid_token"

class LoggingMiddleware(APIMiddleware):
    def handle(self, request):
        print(f"Request: {request}")
        response = self._endpoint.handle(request)
        print(f"Response: {response}")
        return response

class RateLimitMiddleware(APIMiddleware):
    def __init__(self, endpoint, limit=10):
        super().__init__(endpoint)
        self.limit = limit
        self.requests = {}
    
    def handle(self, request):
        user = request.get("user_id")
        count = self.requests.get(user, 0)
        
        if count >= self.limit:
            return {"error": "Rate limit exceeded"}
        
        self.requests[user] = count + 1
        return self._endpoint.handle(request)

class CompressionMiddleware(APIMiddleware):
    def handle(self, request):
        response = self._endpoint.handle(request)
        # Compress response
        response["compressed"] = True
        return response

# 사용 (middleware stack)
endpoint = APIEndpoint()
endpoint = AuthenticationMiddleware(endpoint)
endpoint = LoggingMiddleware(endpoint)
endpoint = RateLimitMiddleware(endpoint, limit=100)
endpoint = CompressionMiddleware(endpoint)

# Request
response = endpoint.handle({
    "token": "valid_token",
    "user_id": 123,
    "data": "request"
})
```

---

## Proxy Pattern

### 문제

**객체 접근 제어:**

```python
# 직접 접근
expensive_object = ExpensiveObject()  # 즉시 초기화 (비용 큼)
result = expensive_object.operation()
```

### 해결

**Proxy로 중개:**

```python
from abc import ABC, abstractmethod

# Subject
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

# Real Subject
class RealImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        print(f"Loading image: {self.filename}")
        # Expensive operation
    
    def display(self):
        print(f"Displaying: {self.filename}")

# Proxy
class ImageProxy(Image):
    def __init__(self, filename):
        self.filename = filename
        self._real_image = None  # Lazy loading
    
    def display(self):
        if self._real_image is None:
            self._real_image = RealImage(self.filename)
        self._real_image.display()

# 사용
image = ImageProxy("large_photo.jpg")  # 빠름 (아직 로드 안 함)
# ...
image.display()  # 이때 로드
```

### Virtual Proxy (Lazy Loading)

```python
class DatabaseProxy:
    """DB connection을 필요할 때만"""
    def __init__(self, config):
        self.config = config
        self._connection = None
    
    def _get_connection(self):
        if self._connection is None:
            print("Establishing DB connection...")
            self._connection = create_real_connection(self.config)
        return self._connection
    
    def query(self, sql):
        conn = self._get_connection()
        return conn.execute(sql)
    
    def close(self):
        if self._connection:
            self._connection.close()
```

### Protection Proxy (Access Control)

```python
class UserService:
    def get_user(self, user_id):
        return {"id": user_id, "name": "John"}
    
    def delete_user(self, user_id):
        print(f"Deleted user {user_id}")

class ProtectedUserService:
    def __init__(self, service, current_user):
        self._service = service
        self._current_user = current_user
    
    def get_user(self, user_id):
        # Anyone can read
        return self._service.get_user(user_id)
    
    def delete_user(self, user_id):
        # Only admin can delete
        if self._current_user.role != "admin":
            raise PermissionError("Admin only")
        return self._service.delete_user(user_id)

# 사용
service = UserService()
proxy = ProtectedUserService(service, current_user)

proxy.get_user(123)  # OK
proxy.delete_user(123)  # PermissionError if not admin
```

### Remote Proxy (분산 시스템)

```python
import requests

class RemoteServiceProxy:
    """원격 서비스를 로컬처럼 사용"""
    def __init__(self, base_url):
        self.base_url = base_url
    
    def get_user(self, user_id):
        response = requests.get(f"{self.base_url}/users/{user_id}")
        return response.json()
    
    def create_user(self, data):
        response = requests.post(f"{self.base_url}/users", json=data)
        return response.json()

# 사용 (원격 API를 로컬 객체처럼)
service = RemoteServiceProxy("https://api.example.com")
user = service.get_user(123)
```

### Caching Proxy

```python
import time

class CachingProxy:
    def __init__(self, real_service):
        self._service = real_service
        self._cache = {}
        self._cache_ttl = 300  # 5분
    
    def get_data(self, key):
        # Check cache
        if key in self._cache:
            cached_data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                print(f"Cache hit: {key}")
                return cached_data
        
        # Cache miss
        print(f"Cache miss: {key}")
        data = self._service.get_data(key)
        self._cache[key] = (data, time.time())
        return data
    
    def invalidate(self, key):
        if key in self._cache:
            del self._cache[key]
```

---

## Facade Pattern

### 문제

**복잡한 하위 시스템:**

```python
# Client가 모든 것을 알아야 함
encoder = VideoEncoder()
codec = CodecFactory.get_codec("mp4")
buffer = BitrateReader()
audio = AudioMixer()

# 복잡한 워크플로우
encoder.set_codec(codec)
buffer.configure(...)
audio.process(...)
# ...
```

### 해결

**Facade로 단순화:**

```python
class VideoConverter:
    """간단한 인터페이스 제공"""
    def __init__(self):
        self._encoder = VideoEncoder()
        self._codec_factory = CodecFactory()
        self._buffer = BitrateReader()
        self._audio = AudioMixer()
    
    def convert(self, filename, format):
        """모든 복잡성 숨김"""
        print(f"Converting {filename} to {format}")
        
        # 1. 코덱 선택
        codec = self._codec_factory.get_codec(format)
        
        # 2. 파일 읽기
        file = VideoFile(filename)
        source_codec = self._codec_factory.get_codec(file.get_codec())
        
        # 3. 변환
        buffer = self._buffer.read(file, source_codec)
        result = self._encoder.encode(buffer, codec)
        
        # 4. 오디오 처리
        if file.has_audio():
            audio = self._audio.fix(result)
            result = audio
        
        # 5. 저장
        new_file = f"converted.{format}"
        result.save(new_file)
        
        return new_file

# 사용 (매우 간단!)
converter = VideoConverter()
converter.convert("video.avi", "mp4")
```

### 실전 예제: API Client

```python
class ComplexAPIClient:
    """여러 서비스를 한 번에"""
    def __init__(self, api_key):
        self.api_key = api_key
        self._auth_service = AuthService()
        self._user_service = UserService()
        self._payment_service = PaymentService()
        self._notification_service = NotificationService()
    
    def create_paid_user_account(self, email, password, card_info):
        """복잡한 워크플로우를 하나로"""
        # 1. 인증
        token = self._auth_service.authenticate(self.api_key)
        
        # 2. 사용자 생성
        user = self._user_service.create_user(email, password, token)
        
        # 3. 결제 설정
        payment = self._payment_service.add_payment_method(
            user.id,
            card_info,
            token
        )
        
        # 4. 환영 이메일
        self._notification_service.send_welcome_email(
            user.email,
            token
        )
        
        return {
            "user": user,
            "payment": payment
        }

# 사용
client = ComplexAPIClient(api_key="xxx")
result = client.create_paid_user_account(
    email="user@example.com",
    password="secret",
    card_info={...}
)
```

---

## Composite Pattern

### 문제

**트리 구조 처리:**

```python
# File system
# Folder
#   ├─ File1
#   ├─ File2
#   └─ Subfolder
#       ├─ File3
#       └─ File4
```

### 해결

**Composite로 통일된 인터페이스:**

```python
from abc import ABC, abstractmethod

# Component
class FileSystemItem(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_size(self):
        pass
    
    @abstractmethod
    def print(self, indent=0):
        pass

# Leaf
class File(FileSystemItem):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def get_size(self):
        return self.size
    
    def print(self, indent=0):
        print("  " * indent + f"📄 {self.name} ({self.size} KB)")

# Composite
class Folder(FileSystemItem):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, item):
        self.children.append(item)
    
    def remove(self, item):
        self.children.remove(item)
    
    def get_size(self):
        return sum(child.get_size() for child in self.children)
    
    def print(self, indent=0):
        print("  " * indent + f"📁 {self.name}")
        for child in self.children:
            child.print(indent + 1)

# 사용
root = Folder("root")
documents = Folder("documents")
pictures = Folder("pictures")

documents.add(File("resume.pdf", 100))
documents.add(File("letter.doc", 50))

pictures.add(File("photo1.jpg", 2000))
pictures.add(File("photo2.jpg", 1500))

root.add(documents)
root.add(pictures)
root.add(File("readme.txt", 10))

# 트리 출력
root.print()
# 📁 root
#   📁 documents
#     📄 resume.pdf (100 KB)
#     📄 letter.doc (50 KB)
#   📁 documents
#     📄 photo1.jpg (2000 KB)
#     📄 photo2.jpg (1500 KB)
#   📄 readme.txt (10 KB)

# 전체 크기
print(f"Total: {root.get_size()} KB")  # 3660 KB
```

---

## Bridge Pattern

### 문제

**다차원 확장:**

```python
# Bad: 조합 폭발
class RedCircle: pass
class BlueCircle: pass
class RedSquare: pass
class BlueSquare: pass
# Color × Shape = N × M 클래스!
```

### 해결

**Bridge로 분리:**

```python
# Implementor
class Color(ABC):
    @abstractmethod
    def fill(self):
        pass

class Red(Color):
    def fill(self):
        return "red"

class Blue(Color):
    def fill(self):
        return "blue"

# Abstraction
class Shape(ABC):
    def __init__(self, color):
        self.color = color
    
    @abstractmethod
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        return f"Circle filled with {self.color.fill()}"

class Square(Shape):
    def draw(self):
        return f"Square filled with {self.color.fill()}"

# 사용 (조합 자유!)
red = Red()
blue = Blue()

circle = Circle(red)
print(circle.draw())  # Circle filled with red

square = Square(blue)
print(square.draw())  # Square filled with blue
```

---

## 패턴 비교

### Adapter vs Decorator vs Proxy

```python
# Adapter: 인터페이스 변환
adapter = Adapter(incompatible_object)

# Decorator: 기능 추가
decorated = Decorator(original_object)

# Proxy: 접근 제어
proxy = Proxy(real_object)
```

### Facade vs Adapter

```
Adapter: 1:1 변환
Facade: N:1 단순화
```

---

## 요약

**구조 패턴:**

1. **Adapter**: 인터페이스 호환
2. **Decorator**: 동적 기능 추가
3. **Proxy**: 접근 제어
4. **Facade**: 복잡성 숨김
5. **Composite**: 트리 구조
6. **Bridge**: 독립적 확장

**언제 사용?**

```
호환 문제 → Adapter
기능 추가 → Decorator
접근 제어 → Proxy
단순화 → Facade
트리 구조 → Composite
다차원 확장 → Bridge
```

**다음 글:**
- **행위 패턴**: Observer, Strategy, Command

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
