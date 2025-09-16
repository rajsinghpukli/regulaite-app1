from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

RegSource = Literal["IFRS", "AAOIFI", "CBB", "InternalPolicy", "General"]

class Quote(BaseModel):
    framework: RegSource
    snippet: str
    citation: Optional[str] = None  # e.g., "IFRS9_Handbook.pdf: p.143" or URL

class PerSourceAnswer(BaseModel):
    status: Literal["addressed", "not_found"] = "not_found"
    notes: Optional[str] = None
    quotes: List[Quote] = Field(default_factory=list)

class RegulAIteAnswer(BaseModel):
    # Canonical 9-section structure you asked to keep
    summary: str
    per_source: Dict[RegSource, PerSourceAnswer]
    comparative_analysis: str = ""
    recommendation: str = ""
    general_knowledge: str = ""
    gaps_or_next_steps: str = ""
    citations: List[str] = Field(default_factory=list)
    ai_opinion: str = ""
    follow_up_suggestions: List[str] = Field(default_factory=list)

    def as_markdown(self) -> str:
        out = []
        out += ["### Summary", self.summary.strip(), ""]
        for fw in ["IFRS", "AAOIFI", "CBB", "InternalPolicy"]:
            ps = self.per_source.get(fw) or PerSourceAnswer()
            out += [f"### {fw}", f"Status: **{ps.status}**"]
            if ps.notes:
                out += [ps.notes]
            if ps.quotes:
                out += ["**Evidence (2–5 quotes):**"]
                for q in ps.quotes:
                    tag = f" — _{q.citation}_" if q.citation else ""
                    out += [f"> {q.snippet}{tag}"]
            out += [""]
        if self.comparative_analysis:
            out += ["### Comparative analysis", self.comparative_analysis, ""]
        if self.recommendation:
            out += ["### Recommendation", self.recommendation, ""]
        if self.general_knowledge:
            out += ["### General knowledge", self.general_knowledge, ""]
        if self.gaps_or_next_steps:
            out += ["### Gaps / Next steps", self.gaps_or_next_steps, ""]
        if self.ai_opinion:
            out += ["### AI opinion", self.ai_opinion, ""]
        if self.citations:
            out += ["### Citations", *[f"- {c}" for c in self.citations], ""]
        return "\n".join(out)

# Default blank shell (useful if parsing fails)
DEFAULT_EMPTY = RegulAIteAnswer(
    summary="",
    per_source={fw: PerSourceAnswer() for fw in ["IFRS", "AAOIFI", "CBB", "InternalPolicy"]},
)
