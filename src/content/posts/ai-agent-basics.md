---
title: "AI Agent #1: Agent의 기초 - ReAct와 Tool Use"
description: "LLM이 도구를 사용하고 추론하는 AI Agent의 기본 원리와 구현을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["ai-agent", "llm", "react", "tool-use", "reasoning"]
draft: false
---

# AI Agent #1: Agent의 기초

**"LLM + Tools = Agent"**

ChatGPT는 대화만 합니다. 하지만 Agent는:
- 웹 검색
- 코드 실행
- 파일 읽기/쓰기
- API 호출

**진짜 유용한 AI!**

---

## Agent란?

### 정의

> **환경을 인식하고, 목표 달성을 위해 행동하는 시스템**

```
User: "오늘 서울 날씨 알려줘"

일반 LLM:
"죄송합니다. 실시간 정보는 모릅니다."

Agent:
1. "날씨 API 사용해야겠다" (Reasoning)
2. weather_api.get("Seoul") (Action)
3. "서울은 현재 15도, 맑음입니다" (Response)
```

### 구성 요소

```
┌─────────────────────┐
│   Agent Loop        │
│                     │
│  1. Observe         │ ← Environment
│  2. Think           │
│  3. Act             │ → Environment
│  4. Repeat          │
└─────────────────────┘
```

---

## ReAct (Reasoning + Acting)

**논문: ReAct (2022)**

### 핵심 아이디어

**기존 방법들:**

```
Chain-of-Thought (CoT):
- 추론만 (Thought → Thought → Answer)
- 행동 없음

Act-only:
- 행동만 (Action → Action → Answer)
- 추론 없음
```

**ReAct:**

> "추론과 행동을 번갈아가며!"

```
Thought → Action → Observation →
Thought → Action → Observation →
...
→ Answer
```

### 예시

**질문:** "현재 미국 대통령의 나이는?"

```
Thought 1: 먼저 현재 미국 대통령이 누군지 알아야 한다.
Action 1: Search["current US president 2024"]
Observation 1: Joe Biden is the current president.

Thought 2: Joe Biden의 생년월일을 찾아야 한다.
Action 2: Search["Joe Biden birth date"]
Observation 2: Born November 20, 1942

Thought 3: 2024 - 1942 = 82세
Action 3: Calculate[2024 - 1942]
Observation 3: 82

Answer: 현재 미국 대통령 Joe Biden은 82세입니다.
```

### 구현

```python
from typing import List, Dict, Any
import openai

class ReActAgent:
    def __init__(self, llm, tools):
        """
        llm: Language model
        tools: Dict of available tools
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = 10
    
    def run(self, question: str) -> str:
        """Run ReAct loop"""
        context = f"Question: {question}\n\n"
        
        for step in range(self.max_steps):
            # 1. Think
            prompt = self._create_prompt(context)
            response = self.llm.generate(prompt)
            
            # Parse response
            thought, action, action_input = self._parse_response(response)
            
            context += f"Thought {step+1}: {thought}\n"
            context += f"Action {step+1}: {action}[{action_input}]\n"
            
            # 2. Act
            if action == "Finish":
                return action_input
            
            if action not in self.tools:
                context += f"Observation {step+1}: Error - Unknown action\n"
                continue
            
            # 3. Observe
            observation = self.tools[action](action_input)
            context += f"Observation {step+1}: {observation}\n\n"
        
        return "Max steps reached without answer"
    
    def _create_prompt(self, context: str) -> str:
        return f"""Answer the following question using available tools.

Available tools:
- Search[query]: Search the web
- Calculate[expression]: Evaluate math expression
- Finish[answer]: Return final answer

Format:
Thought: [your reasoning]
Action: [tool name][input]

{context}"""
    
    def _parse_response(self, response: str):
        """Parse LLM response into thought, action, input"""
        lines = response.strip().split('\n')
        
        thought = ""
        action = ""
        action_input = ""
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.split("Thought:", 1)[1].strip()
            elif line.startswith("Action:"):
                action_line = line.split("Action:", 1)[1].strip()
                # Parse "ActionName[input]"
                if '[' in action_line:
                    action = action_line.split('[')[0].strip()
                    action_input = action_line.split('[')[1].rstrip(']')
        
        return thought, action, action_input

# Tools
def search_tool(query: str) -> str:
    """Simulate web search"""
    # In practice: use Google API, Bing API, etc.
    results = {
        "current US president 2024": "Joe Biden is the president",
        "Joe Biden birth date": "November 20, 1942"
    }
    return results.get(query, "No results found")

def calculate_tool(expression: str) -> str:
    """Evaluate math expression"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error in calculation"

# Agent
agent = ReActAgent(
    llm=OpenAILLM(model="gpt-4"),
    tools={
        "Search": search_tool,
        "Calculate": calculate_tool
    }
)

# Run
answer = agent.run("현재 미국 대통령의 나이는?")
print(answer)
```

---

## Tool Use (Function Calling)

### OpenAI Function Calling

**LLM에게 도구 목록 제공:**

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Seoul"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "서울 날씨 알려줘"}
    ],
    tools=tools,
    tool_choice="auto"
)

# LLM이 도구 선택!
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        # Function: get_weather
        # Arguments: {'location': 'Seoul', 'unit': 'celsius'}
```

### Agent with Function Calling

```python
import json

class FunctionCallingAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.tools = self._define_tools()
        self.functions = self._define_functions()
    
    def _define_tools(self):
        """Tool definitions for OpenAI"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def _define_functions(self):
        """Actual function implementations"""
        return {
            "get_weather": self._get_weather,
            "calculate": self._calculate
        }
    
    def _get_weather(self, location: str) -> str:
        # Call weather API
        return f"{location}: 15°C, Sunny"
    
    def _calculate(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except:
            return "Error"
    
    def run(self, user_message: str) -> str:
        """Run agent loop"""
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            # LLM call
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # No tool call → final answer
            if not message.tool_calls:
                return message.content
            
            # Execute tools
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute
                result = self.functions[function_name](**arguments)
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })

# 사용
agent = FunctionCallingAgent()
answer = agent.run("서울 날씨가 어때? 화씨로 변환하면?")
print(answer)
# 1. get_weather("Seoul") → "15°C"
# 2. calculate("15 * 9/5 + 32") → "59°F"
# 3. "서울은 현재 59°F입니다"
```

---

## LangChain Agent

**LangChain으로 간단히:**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculate_tool,
        description="Evaluate math expressions"
    )
]

# Initialize agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run
result = agent.run("현재 미국 대통령의 나이는?")
print(result)
```

**Output:**

```
> Entering new AgentExecutor chain...
I need to find who the current US president is and their age.
Action: Search
Action Input: "current US president 2024"
Observation: Joe Biden is the current president
Thought: Now I need to find Joe Biden's age
Action: Search
Action Input: "Joe Biden age"
Observation: Born November 20, 1942
Thought: I need to calculate the age
Action: Calculator
Action Input: 2024 - 1942
Observation: 82
Thought: I now know the final answer
Final Answer: Joe Biden is 82 years old

> Finished chain.
```

---

## Multi-step Planning

### Task Decomposition

**복잡한 작업 분해:**

```
User: "Python으로 웹 크롤러 만들어줘"

Agent:
1. 요구사항 분석
   - 어떤 사이트?
   - 어떤 데이터?

2. 계획 수립
   - BeautifulSoup 사용
   - requests로 HTML 가져오기
   - 데이터 파싱
   - CSV 저장

3. 구현
   - 코드 작성
   - 테스트
   - 수정

4. 완료
   - 코드 제공
   - 사용법 설명
```

### Plan-and-Execute Agent

```python
class PlanAndExecuteAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def run(self, task: str) -> str:
        # 1. Plan
        plan = self._create_plan(task)
        print("Plan:")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
        
        # 2. Execute
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append(result)
            print(f"✓ {step}: {result}")
        
        # 3. Synthesize
        return self._synthesize(task, results)
    
    def _create_plan(self, task: str) -> List[str]:
        """Create step-by-step plan"""
        prompt = f"""Break down this task into steps:
Task: {task}

Steps:"""
        
        response = self.llm.generate(prompt)
        steps = [s.strip() for s in response.split('\n') if s.strip()]
        return steps
    
    def _execute_step(self, step: str, previous_results: List[str]) -> str:
        """Execute a single step"""
        context = "\n".join([f"Step {i+1}: {r}" for i, r in enumerate(previous_results)])
        
        prompt = f"""Execute this step:
{step}

Previous results:
{context}

Use available tools if needed.
"""
        # Similar to ReAct execution
        return self._react_loop(prompt)
    
    def _synthesize(self, task: str, results: List[str]) -> str:
        """Combine results into final answer"""
        prompt = f"""Task: {task}

Results:
{chr(10).join([f'{i+1}. {r}' for i, r in enumerate(results)])}

Final answer:"""
        
        return self.llm.generate(prompt)
```

---

## Memory

**Agent가 이전 대화 기억:**

```python
class AgentWithMemory:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []  # Conversation history
    
    def run(self, user_input: str) -> str:
        # Add to memory
        self.memory.append({"role": "user", "content": user_input})
        
        # Create prompt with memory
        prompt = self._create_prompt_with_memory()
        
        # ReAct loop
        response = self._react_loop(prompt)
        
        # Add to memory
        self.memory.append({"role": "assistant", "content": response})
        
        return response
    
    def _create_prompt_with_memory(self) -> str:
        context = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.memory
        ])
        
        return f"""Previous conversation:
{context}

Continue the conversation using available tools."""

# 대화
agent = AgentWithMemory(llm, tools)

agent.run("서울 날씨 알려줘")
# "서울은 15도, 맑음입니다"

agent.run("그럼 부산은?")
# Memory: "서울 날씨" → "그럼 부산은?" = 날씨 질문!
# "부산은 18도, 흐림입니다"
```

---

## 실전 예제: 코드 실행 Agent

```python
import subprocess

class CodeExecutionAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, task: str) -> str:
        # 1. Generate code
        code = self._generate_code(task)
        print("Generated code:")
        print(code)
        
        # 2. Execute
        result = self._execute_code(code)
        print(f"Result: {result}")
        
        # 3. Fix if error
        if result.startswith("Error"):
            code = self._fix_code(code, result)
            result = self._execute_code(code)
        
        return result
    
    def _generate_code(self, task: str) -> str:
        prompt = f"""Write Python code for this task:
{task}

Code:"""
        return self.llm.generate(prompt)
    
    def _execute_code(self, code: str) -> str:
        try:
            # Save to file
            with open("temp.py", "w") as f:
                f.write(code)
            
            # Execute
            result = subprocess.run(
                ["python", "temp.py"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Timeout"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _fix_code(self, code: str, error: str) -> str:
        prompt = f"""Fix this code:

Code:
{code}

Error:
{error}

Fixed code:"""
        return self.llm.generate(prompt)

# 사용
agent = CodeExecutionAgent(llm)
result = agent.run("피보나치 수열 10개 출력")
```

---

## 요약

**AI Agent 핵심:**

1. **ReAct**: Reasoning + Acting
2. **Tool Use**: LLM이 도구 사용
3. **Planning**: 복잡한 작업 분해
4. **Memory**: 대화 기억

**구조:**

```
Observe → Think → Act → Repeat
```

**실전:**
- OpenAI Function Calling
- LangChain Agents
- Custom Agent 구현

**다음 글:**
- **Advanced Agents**: AutoGPT, BabyAGI
- **Multi-Agent Systems**: 여러 Agent 협업
- **Agent Evaluation**: 성능 측정

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
