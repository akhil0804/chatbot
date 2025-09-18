from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    @abstractmethod
    def run(self, user_query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ...