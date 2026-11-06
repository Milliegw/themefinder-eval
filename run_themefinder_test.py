import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from themefinder import OpenAILLM, find_themes

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing")

llm = OpenAILLM(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    request_kwargs={"temperature": 0},
)

question = "What themes are present in this consultation response?"
system_prompt = "You are analysing public consultation responses for a council consultation."
responses_df = pd.read_csv("data/golden/themefinder_input.csv")

def to_jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return value
    return value

async def main():
    result = await find_themes(responses_df, llm, question, system_prompt=system_prompt)

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "result": {k: to_jsonable(v) for k, v in result.items()},
    }

    out_path = Path("runs") / f"themefinder_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved {out_path}")

asyncio.run(main())