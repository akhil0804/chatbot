from openai import OpenAI
import os
from dotenv import load_dotenv,find_dotenv
from .base import LLM 

load_dotenv(find_dotenv(usecwd=True), override=False)

class openrouter(LLM):
    def __init__(self):
        api_key = os.getenv("Openrouter_key")

                # Validate envs with clear errors
        missing = [k for k,v in {
            "Openrouter_key": api_key,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required Azure env(s): {', '.join(missing)}")
        

        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        )

        
    def complete(self, prompt: str, **kwargs) -> str:
        system_msg = kwargs.get("system", "You are a precise assistant.")
        temperature = kwargs.get("temperature", 0)
        resp = self.client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
             {
            "role": "system",
            "content": system_msg
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=temperature
        )
        return resp.choices[0].message.content.strip()
        # print(completion.choices[0].message.content)
