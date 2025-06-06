import asyncio
import os
import traceback
from typing import List
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam
from app.core.agent import Agent, Runner
from dotenv import load_dotenv

from app.tools.upbit import (
    get_current_ticker,
    get_candles_for_minutes,
    get_candles_for_daily,
    get_candles_for_weekly
)
from app.tools.analyze import analyze_btc_mareket

load_dotenv()

async def run():
    messages: List[MessageParam] = []
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tools = [
        get_current_ticker,
        get_candles_for_minutes,
        get_candles_for_daily,
        get_candles_for_weekly,
        analyze_btc_mareket
    ]
    simple_agent = Agent(
        name="Assistant",
        model="claude-sonnet-4-20250514",
        instructions="You are a helpful assistant. Be concise and friendly.",
        max_tokens=2000,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        tools=tools,
        client=client
    )

    try:
        while True:
            user_input = input("> ")
            messages.append({"role": "user", "content": user_input})
            if user_input.lower() == 'exit':
                break
            response = await Runner.run(simple_agent, messages)
            print(response.final_output)
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")

async def run_stream():
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

async def main():
    print("=== Talk to Anthorpic Agent ===")
    await run()

if __name__ == "__main__":
    asyncio.run(main())
