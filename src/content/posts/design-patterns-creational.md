---
title: "Design Patterns #1: 생성 패턴 - Singleton, Factory, Builder"
description: "객체 생성을 유연하고 재사용 가능하게 만드는 생성 디자인 패턴을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["design-patterns", "oop", "software-engineering", "python", "creational-patterns"]
draft: false
---

# Design Patterns #1: 생성 패턴

**"어떻게 객체를 만들 것인가?"**

간단해 보이지만, 실제로는:
- Thread-safe하게
- 메모리 효율적으로
- 유연하게
- 테스트 가능하게

**생성 패턴 (Creational Patterns)**이 답입니다.

---

## Singleton Pattern

### 문제

**전역적으로 하나만 필요한 객체:**

```python
# Database connection
db1 = Database()
db2 = Database()  # 또 다른 connection?

# 문제: 리소스 낭비, 일관성 문제
```

### 해결

**하나의 인스턴스만:**

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 사용
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True (같은 객체!)
```

### Thread-Safe Version

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

### 실전 예제: Database Connection

```python
class Database:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._connection = None
        return cls._instance
    
    def connect(self, host, port):
        if self._connection is None:
            self._connection = create_connection(host, port)
            print(f"Connected to {host}:{port}")
        return self._connection
    
    def query(self, sql):
        if self._connection is None:
            raise Exception("Not connected")
        return self._connection.execute(sql)

# 어디서든 같은 connection
db1 = Database()
db1.connect("localhost", 5432)

db2 = Database()
db2.query("SELECT * FROM users")  # 같은 connection 사용
```

### 언제 사용?

```
✅ Database connection pool
✅ Configuration manager
✅ Logger
✅ Cache manager

❌ 일반 데이터 객체
❌ 상태가 자주 변하는 객체
```

### 주의점

```python
# Anti-pattern: Global state
class BadSingleton:
    _instance = None
    counter = 0  # Shared state!
    
    def increment(self):
        self.counter += 1  # 어디서든 변경 가능 (위험!)

# Better: Encapsulation
class GoodSingleton:
    _instance = None
    
    def __init__(self):
        self._counter = 0  # Private
    
    def increment(self):
        self._counter += 1
    
    def get_count(self):
        return self._counter
```

---

## Factory Pattern

### 문제

**객체 생성 로직이 복잡:**

```python
# Bad: Client가 구체적인 클래스 알아야 함
if notification_type == "email":
    notifier = EmailNotifier()
elif notification_type == "sms":
    notifier = SMSNotifier()
elif notification_type == "push":
    notifier = PushNotifier()
```

### 해결

**Factory가 생성 책임:**

```python
from abc import ABC, abstractmethod

# Abstract Product
class Notifier(ABC):
    @abstractmethod
    def send(self, message):
        pass

# Concrete Products
class EmailNotifier(Notifier):
    def send(self, message):
        print(f"Email: {message}")

class SMSNotifier(Notifier):
    def send(self, message):
        print(f"SMS: {message}")

class PushNotifier(Notifier):
    def send(self, message):
        print(f"Push: {message}")

# Factory
class NotifierFactory:
    @staticmethod
    def create_notifier(type):
        if type == "email":
            return EmailNotifier()
        elif type == "sms":
            return SMSNotifier()
        elif type == "push":
            return PushNotifier()
        else:
            raise ValueError(f"Unknown type: {type}")

# 사용
notifier = NotifierFactory.create_notifier("email")
notifier.send("Hello!")
```

### Abstract Factory

**관련 객체들을 한 번에:**

```python
# Abstract Products
class Button(ABC):
    @abstractmethod
    def render(self):
        pass

class Checkbox(ABC):
    @abstractmethod
    def render(self):
        pass

# Concrete Products - Windows
class WindowsButton(Button):
    def render(self):
        return "[Windows Button]"

class WindowsCheckbox(Checkbox):
    def render(self):
        return "[Windows Checkbox]"

# Concrete Products - Mac
class MacButton(Button):
    def render(self):
        return "[Mac Button]"

class MacCheckbox(Checkbox):
    def render(self):
        return "[Mac Checkbox]"

# Abstract Factory
class UIFactory(ABC):
    @abstractmethod
    def create_button(self):
        pass
    
    @abstractmethod
    def create_checkbox(self):
        pass

# Concrete Factories
class WindowsFactory(UIFactory):
    def create_button(self):
        return WindowsButton()
    
    def create_checkbox(self):
        return WindowsCheckbox()

class MacFactory(UIFactory):
    def create_button(self):
        return MacButton()
    
    def create_checkbox(self):
        return MacCheckbox()

# 사용
def render_ui(factory: UIFactory):
    button = factory.create_button()
    checkbox = factory.create_checkbox()
    print(button.render())
    print(checkbox.render())

# OS에 따라
if os_type == "windows":
    factory = WindowsFactory()
else:
    factory = MacFactory()

render_ui(factory)
```

### 실전 예제: Database Driver

```python
class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def query(self, sql):
        pass

class PostgreSQLConnection(DatabaseConnection):
    def connect(self):
        print("Connected to PostgreSQL")
    
    def query(self, sql):
        return f"PostgreSQL: {sql}"

class MySQLConnection(DatabaseConnection):
    def connect(self):
        print("Connected to MySQL")
    
    def query(self, sql):
        return f"MySQL: {sql}"

class MongoDBConnection(DatabaseConnection):
    def connect(self):
        print("Connected to MongoDB")
    
    def query(self, sql):
        return f"MongoDB: {sql}"

class DatabaseFactory:
    _drivers = {
        "postgresql": PostgreSQLConnection,
        "mysql": MySQLConnection,
        "mongodb": MongoDBConnection
    }
    
    @classmethod
    def register_driver(cls, name, driver_class):
        cls._drivers[name] = driver_class
    
    @classmethod
    def create_connection(cls, driver_name):
        driver_class = cls._drivers.get(driver_name)
        if not driver_class:
            raise ValueError(f"Unknown driver: {driver_name}")
        return driver_class()

# 사용 (config에서 읽음)
db_type = config.get("database", "postgresql")
db = DatabaseFactory.create_connection(db_type)
db.connect()
```

---

## Builder Pattern

### 문제

**생성자가 너무 복잡:**

```python
# Bad: 매개변수 많음
user = User(
    "John",
    "Doe",
    30,
    "john@example.com",
    "123-456-7890",
    "123 Main St",
    "New York",
    "NY",
    "10001"
)  # 순서 헷갈림, 선택적 매개변수 어려움
```

### 해결

**단계별로 구성:**

```python
class User:
    def __init__(self):
        self.first_name = None
        self.last_name = None
        self.age = None
        self.email = None
        self.phone = None
        self.address = None
        self.city = None
        self.state = None
        self.zip = None
    
    def __str__(self):
        return f"User({self.first_name} {self.last_name}, {self.email})"

class UserBuilder:
    def __init__(self):
        self.user = User()
    
    def set_name(self, first, last):
        self.user.first_name = first
        self.user.last_name = last
        return self  # Method chaining
    
    def set_age(self, age):
        self.user.age = age
        return self
    
    def set_email(self, email):
        self.user.email = email
        return self
    
    def set_phone(self, phone):
        self.user.phone = phone
        return self
    
    def set_address(self, address, city, state, zip):
        self.user.address = address
        self.user.city = city
        self.user.state = state
        self.user.zip = zip
        return self
    
    def build(self):
        # Validation
        if not self.user.email:
            raise ValueError("Email is required")
        return self.user

# 사용 (Fluent Interface)
user = (UserBuilder()
    .set_name("John", "Doe")
    .set_email("john@example.com")
    .set_age(30)
    .set_phone("123-456-7890")
    .build())

# 선택적 매개변수 쉽게
simple_user = (UserBuilder()
    .set_name("Jane", "Smith")
    .set_email("jane@example.com")
    .build())
```

### Director Pattern

**미리 정의된 구성:**

```python
class UserDirector:
    def __init__(self, builder):
        self.builder = builder
    
    def build_minimal_user(self, email):
        return (self.builder
            .set_name("User", "Unknown")
            .set_email(email)
            .build())
    
    def build_full_user(self, data):
        return (self.builder
            .set_name(data['first'], data['last'])
            .set_email(data['email'])
            .set_age(data['age'])
            .set_phone(data['phone'])
            .set_address(
                data['address'],
                data['city'],
                data['state'],
                data['zip']
            )
            .build())

# 사용
director = UserDirector(UserBuilder())
user = director.build_minimal_user("test@example.com")
```

### 실전 예제: SQL Query Builder

```python
class Query:
    def __init__(self):
        self.select_fields = []
        self.table = None
        self.where_conditions = []
        self.order_by = None
        self.limit_value = None
    
    def to_sql(self):
        sql = f"SELECT {', '.join(self.select_fields)}"
        sql += f" FROM {self.table}"
        
        if self.where_conditions:
            sql += " WHERE " + " AND ".join(self.where_conditions)
        
        if self.order_by:
            sql += f" ORDER BY {self.order_by}"
        
        if self.limit_value:
            sql += f" LIMIT {self.limit_value}"
        
        return sql

class QueryBuilder:
    def __init__(self):
        self.query = Query()
    
    def select(self, *fields):
        self.query.select_fields = list(fields)
        return self
    
    def from_table(self, table):
        self.query.table = table
        return self
    
    def where(self, condition):
        self.query.where_conditions.append(condition)
        return self
    
    def order_by(self, field):
        self.query.order_by = field
        return self
    
    def limit(self, n):
        self.query.limit_value = n
        return self
    
    def build(self):
        if not self.query.select_fields:
            self.query.select_fields = ['*']
        if not self.query.table:
            raise ValueError("Table is required")
        return self.query

# 사용
query = (QueryBuilder()
    .select("id", "name", "email")
    .from_table("users")
    .where("age > 18")
    .where("status = 'active'")
    .order_by("created_at DESC")
    .limit(10)
    .build())

print(query.to_sql())
# SELECT id, name, email FROM users WHERE age > 18 AND status = 'active' ORDER BY created_at DESC LIMIT 10
```

---

## Prototype Pattern

### 문제

**객체 복사가 복잡:**

```python
# Deep copy 필요
original = ComplexObject()
copy = ???  # 어떻게?
```

### 해결

**Clone 메서드:**

```python
import copy

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class Shape(Prototype):
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

class Circle(Shape):
    def __init__(self, x, y, color, radius):
        super().__init__(x, y, color)
        self.radius = radius
    
    def __str__(self):
        return f"Circle at ({self.x},{self.y}), r={self.radius}, color={self.color}"

# 사용
original = Circle(10, 20, "red", 5)
clone = original.clone()
clone.x = 100  # 독립적
print(original)  # Circle at (10,20)
print(clone)     # Circle at (100,20)
```

### Prototype Registry

```python
class ShapeRegistry:
    def __init__(self):
        self._prototypes = {}
    
    def register(self, name, prototype):
        self._prototypes[name] = prototype
    
    def create(self, name):
        prototype = self._prototypes.get(name)
        if not prototype:
            raise ValueError(f"Unknown prototype: {name}")
        return prototype.clone()

# 등록
registry = ShapeRegistry()
registry.register("red_circle", Circle(0, 0, "red", 10))
registry.register("blue_square", Square(0, 0, "blue", 20))

# 사용 (빠른 생성)
shape1 = registry.create("red_circle")
shape1.x = 50

shape2 = registry.create("red_circle")
shape2.x = 100
```

---

## 패턴 비교

### 언제 무엇을?

```python
# Singleton: 전역적으로 하나
logger = Logger()  # 어디서든 같은 인스턴스

# Factory: 타입에 따라 다른 객체
notifier = NotifierFactory.create(type)

# Builder: 복잡한 생성
user = UserBuilder().set_name(...).set_email(...).build()

# Prototype: 복사
new_shape = existing_shape.clone()
```

### 조합

```python
# Singleton + Factory
class DatabaseFactory(Singleton):
    def create_connection(self, driver):
        # Factory method
        pass

# Builder + Factory
class CarFactory:
    @staticmethod
    def create_sports_car():
        return (CarBuilder()
            .set_engine("V8")
            .set_color("red")
            .set_max_speed(300)
            .build())
```

---

## 실전 팁

### 1. Singleton 대안

```python
# Module-level singleton (Python)
# database.py
_connection = None

def get_connection():
    global _connection
    if _connection is None:
        _connection = create_connection()
    return _connection

# 어디서든
from database import get_connection
db = get_connection()
```

### 2. Factory 확장

```python
# Plugin system
class PluginFactory:
    _plugins = {}
    
    @classmethod
    def register(cls, name):
        def decorator(plugin_class):
            cls._plugins[name] = plugin_class
            return plugin_class
        return decorator
    
    @classmethod
    def create(cls, name):
        return cls._plugins[name]()

# 사용
@PluginFactory.register("json")
class JSONPlugin:
    pass

@PluginFactory.register("xml")
class XMLPlugin:
    pass

plugin = PluginFactory.create("json")
```

### 3. Builder 검증

```python
class ValidatedBuilder:
    def build(self):
        self._validate()
        return self.product
    
    def _validate(self):
        if not self.product.email:
            raise ValueError("Email required")
        if not self.product.email.contains("@"):
            raise ValueError("Invalid email")
```

---

## 요약

**생성 패턴:**

1. **Singleton**: 하나의 인스턴스
2. **Factory**: 객체 생성 위임
3. **Builder**: 단계별 구성
4. **Prototype**: 복사로 생성

**언제 사용?**

```
복잡한 생성 → Builder
타입별 생성 → Factory
전역 인스턴스 → Singleton
복사 필요 → Prototype
```

**다음 글:**
- **구조 패턴**: Adapter, Decorator, Proxy
- **행위 패턴**: Observer, Strategy, Command

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
