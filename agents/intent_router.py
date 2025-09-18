from typing import Dict, Any
from .base import Agent
from llm.base import LLM

class IntentLLMAgent(Agent):
    def __init__(self, llm: LLM, classify_prompt_path="config/prompts/classify_intent.txt"):
        self.llm = llm
        with open(classify_prompt_path, "r") as f:
            self.template = f.read()

    def run(self, user_query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        prompt = self.template.replace("{{question}}", user_query)
        out = self.llm.complete(prompt, temperature=0).strip().upper()
        # Take only the first token (LLMs sometimes add extra text/newlines)
        label = out.split()[0] if out else ""
        if label not in {"DB_QUERY", "SMALL_TALK"}:
            label = ""  # let app.py handle fallback
        return {"type": "intent", "label": label}