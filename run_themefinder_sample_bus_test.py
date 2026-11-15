import asyncio
import json
import pandas as pd
from langchain_ollama import ChatOllama
from themefinder import OpenAILLM, find_themes
import os
import pandas as pd
from dotenv import load_dotenv
from themefinder import find_themes
from themefinder.llm import OpenAILLM
from datetime import datetime
from pathlib import Path

load_dotenv()

# llm = OpenAILLM(
#     model="llama3.1",
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",                    # required by the client, ignored by Ollama
#     request_kwargs={"temperature": 0},   # consistent outputs between runs
# )

llm = OpenAILLM(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    request_kwargs={"temperature": 0},
)

MODEL = "openai/gpt-4o-mini"   

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

# async def main():
#     result = await find_themes(responses_df, llm, question, system_prompt=system_prompt)
#     print(result)
#     with open("first_run.json", "w") as f:
#         json.dump(result, f, indent=2, default=str)

def to_jsonable(value):
    """Turn ThemeFinder's pandas tables into plain lists so they save cleanly."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value

async def main():
    result = await find_themes(responses_df, llm, question, system_prompt=system_prompt)

    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "temperature": 0,
        "themefinder_version": "0.8.2",
        "question": question,
        "result": {key: to_jsonable(value) for key, value in result.items()},
    }

    Path("runs").mkdir(exist_ok=True)
    filename = Path("runs") / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    filename.write_text(json.dumps(run, indent=2, default=str))
    print(f"Saved {filename}")

if __name__ == "__main__":
    asyncio.run(main())