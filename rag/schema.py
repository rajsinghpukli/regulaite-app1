from dataclasses import dataclass
from typing import Optional

@dataclass
class Route:
    intent: str
    scope: Optional[str] = None
