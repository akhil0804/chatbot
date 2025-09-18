from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        ...