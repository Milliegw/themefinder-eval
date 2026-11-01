import asyncio
import json
import pandas as pd
from langchain_ollama import ChatOllama
from themefinder import OpenAILLM, find_themes

llm = OpenAILLM(
    model="llama3.1",
    base_url="http://localhost:11434/v1",
    api_key="ollama",                    # required by the client, ignored by Ollama
    request_kwargs={"temperature": 0},   # consistent outputs between runs
)

responses_df = pd.DataFrame({
    "response_id": ["1", "2", "3", "4", "5", "6"],
    "response": [
        "Bus fares are too expensive for families on low incomes.",
        "Services in my village were cut and now I can't get to work.",
        "Buses are often late, so I've started driving instead.",
        "Cheaper fares would help, but reliability matters more.",
        "There should be more evening services for shift workers.",
        "I can't use the bus because the stop isn't accessible for my wheelchair.",
    ],
})

question = "How could local bus services be improved?"
system_prompt = "You are analysing public consultation responses for a local transport authority."

async def main():
    result = await find_themes(responses_df, llm, question, system_prompt=system_prompt)
    print(result)
    with open("first_run.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(main())