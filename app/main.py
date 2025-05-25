import asyncio
import os
from typing import List
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam
from app.core.agent import Agent, Runner
from dotenv import load_dotenv
load_dotenv()

async def main():
    print("=== Talk to Anthorpic Agent ===")
    messages: List[MessageParam] = []
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    simple_agent = Agent(
        name="Assistant",
        model="claude-sonnet-4-20250514",
        instructions="You are a helpful assistant. Be concise and friendly.",
        client=client
    )
    
    while True:
        user_input = input("> ")
        messages.append({"role": "user", "content": user_input})

        if user_input.lower() == 'exit':
            break
        async for chunk in Runner.run_stream(simple_agent, messages):
            if not chunk.is_complete:
                print(chunk.content, end='', flush=True)
            else:
                print("\n")
                break

if __name__ == "__main__":
    asyncio.run(main())
