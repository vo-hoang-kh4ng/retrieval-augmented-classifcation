from typing import Optional
from dataclasses import dataclass
import os

@dataclass
class LLMClients:
openai_api_key: Optional[str]
google_api_key: Optional[str]

@classmethod
def from_env(cls) -> "LLMClients":
return cls(
openai_api_key=os.getenv("OPENAI_API_KEY"),
google_api_key=os.getenv("GOOGLE_API_KEY"),
)
