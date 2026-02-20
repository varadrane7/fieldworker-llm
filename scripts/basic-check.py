from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
endpoint = os.getenv("FW_API_URL")
deployment_name = "gpt-4.1"
api_key = os.getenv("FW_API_KEY")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Italy?",
        }
    ],
)

print(completion.choices[0].message)