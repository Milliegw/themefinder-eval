import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from .env

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

reply = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words"}],
)
print(reply.choices[0].message.content)