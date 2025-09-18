import os
from .base import LLM
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)
class AzureOpenAILLM(LLM):

    def __init__(self):
        # import here to surface version errors early
        try:
            from openai import AzureOpenAI
        except Exception as e:
            raise RuntimeError(
                "Failed to import AzureOpenAI. Upgrade the SDK: pip install -U openai"
            ) from e

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Validate envs with clear errors
        missing = [k for k,v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required Azure env(s): {', '.join(missing)}")

        # Construct client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment = deployment

    def complete(self, prompt: str, **kwargs) -> str:
        system_msg = kwargs.get("system", "You are a precise assistant.")
        temperature = kwargs.get("temperature", 0)
        resp = self.client.chat.completions.create(
            model=self.deployment,  # deployment name (not base model)
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()