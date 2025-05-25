# Claude Agent SDK

Anthropic Python SDK를 기반으로 한 경량화된 에이전트 프레임워크입니다. OpenAI의 Agent SDK에서 영감을 받아 제작되었습니다.

## 주요 기능

- **Agent**: 지시사항, 도구, 핸드오프가 구성된 Claude 에이전트
- **Function Tools**: Python 함수를 에이전트 도구로 쉽게 변환
- **Handoffs**: 에이전트 간 제어권 전달을 위한 특수 도구 호출
- **Runner**: 동기/비동기 에이전트 실행을 위한 유틸리티

## 설치

```bash
# 의존성 설치
uv sync

# 환경 변수 설정
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 사용법

### 기본 에이전트

```python
from app.core.agent import Agent, Runner

agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

### 함수 도구 사용

```python
from app.core.agent import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Assistant",
    instructions="You are a helpful assistant.",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

### 멀티 에이전트 핸드오프

```python
from app.core.agent import Agent, Runner
import asyncio

korean_agent = Agent(
    name="Korean Agent",
    instructions="You only speak Korean."
)

english_agent = Agent(
    name="English Agent", 
    instructions="You only speak English"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Handoff to the appropriate agent based on the language.",
    handoffs=[korean_agent, english_agent]
)

async def main():
    result = await Runner.run(triage_agent, "안녕하세요!")
    print(result.final_output)

asyncio.run(main())
```

### 스트림 실행

실시간으로 응답을 받아보고 싶을 때 스트림 기능을 사용할 수 있습니다:

```python
from app.core.agent import Agent, Runner
import asyncio

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

# 비동기 스트림
async def stream_example():
    async for chunk in Runner.run_stream(agent, "Write a story about AI."):
        if not chunk.is_complete:
            print(chunk.content, end='', flush=True)
        else:
            print("\n[완료]")
            break

asyncio.run(stream_example())

# 동기 스트림
for chunk in Runner.run_stream_sync(agent, "Explain machine learning."):
    if not chunk.is_complete:
        print(chunk.content, end='', flush=True)
    else:
        print("\n[완료]")
        break
```

## 에이전트 루프

`Runner.run()`을 호출하면 최종 출력을 얻을 때까지 루프를 실행합니다:

1. 에이전트의 모델과 설정, 메시지 히스토리를 사용하여 Claude API 호출
2. Claude가 응답을 반환 (도구 호출 포함 가능)
3. 응답에 최종 출력이 있으면 반환하고 루프 종료
4. 응답에 핸드오프가 있으면 새 에이전트로 설정하고 1단계로 돌아감
5. 도구 호출이 있으면 처리하고 도구 응답 메시지를 추가한 후 1단계로 돌아감

## 예제 실행

```bash
cd /Users/yspark/workspace/toy/claude-agent
python app/main.py
```

## 프로젝트 구조

```
claud-agent/
├── app/
│   ├── core/
│   │   └── agent.py      # 핵심 Agent 클래스들
│   └── main.py           # 사용 예제
├── pyproject.toml        # 프로젝트 설정
└── README.md
```

## API 참조

### Agent 클래스

```python
Agent(
    name: str,                              # 에이전트 이름
    instructions: str,                      # 시스템 지시사항
    model: str = "claude-3-5-sonnet-20241022",  # 사용할 모델
    tools: List[FunctionTool] = None,       # 사용 가능한 도구들
    handoffs: List[Agent] = None,           # 핸드오프 가능한 에이전트들
    max_tokens: int = 4096,                 # 최대 토큰 수
    temperature: float = 0.0,               # 온도 설정
    client: Anthropic = None                # Anthropic 클라이언트
)
```

### function_tool 데코레이터

```python
@function_tool
def your_function(param: type) -> return_type:
    """함수 설명"""
    return result
```

### Runner 클래스

```python
# 비동기 실행
result = await Runner.run(agent, messages, max_turns=10)

# 동기 실행
result = Runner.run_sync(agent, messages, max_turns=10)

# 비동기 스트림 실행
async for chunk in Runner.run_stream(agent, messages, max_turns=10):
    if not chunk.is_complete:
        print(chunk.content, end='', flush=True)
    else:
        break

# 동기 스트림 실행
for chunk in Runner.run_stream_sync(agent, messages, max_turns=10):
    if not chunk.is_complete:
        print(chunk.content, end='', flush=True)
    else:
        break
```

### StreamChunk 클래스

```python
@dataclass
class StreamChunk:
    content: str                    # 스트림된 텍스트 조각
    is_complete: bool = False       # 스트림 완료 여부
    agent_name: str = ""           # 현재 에이전트 이름
    messages: List[MessageParam] = field(default_factory=list)  # 메시지 히스토리
```

## 라이선스

MIT License