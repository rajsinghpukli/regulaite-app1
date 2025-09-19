from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

RegSource = Literal["IFRS", "AAOIFI", "CBB", "InternalPolicy"]

class Quote(BaseModel):
    framework: RegSource
    snippet: str
    citation: Optional[str] = None

class PerSourceAnswer(BaseModel):
    # We no longer expose 'status' to the UI. If a framework has no content, it won't be present.
    notes: Optional[str] = None
    quotes: List[Quote] = Field(default_factory=list)

class RegulAIteAnswer(BaseModel):
    summary: str
    per_source: Dict[RegSource, PerSourceAnswer] = Field(default_factory=dict)
    comparative_analysis: str = ""
    recommendation: str = ""
    general_knowledge: str = ""
    gaps_or_next_steps: str = ""
    citations: List[str] = Field(default_factory=list)
    ai_opinion: str = ""
    follow_up_suggestions: List[str] = Field(default_factory=list)
    comparison_table_md: Optional[str] = None  # Optional Markdown table

    def as_markdown(self) -> str:
        parts: List[str] = []

        if self.summary:
            parts += ["### Summary", self.summary.strip(), ""]

        # Only render frameworks that actually exist in the object
        order = ["IFRS", "AAOIFI", "CBB", "InternalPolicy"]
        for fw in order:
            ps = self.per_source.get(fw)  # may be None
            if not ps:
                continue
            sec: List[str] = [f"### {fw}"]
            if ps.notes:
                sec.append(ps.notes.strip())

            if ps.quotes:
                sec.append("")
                sec.append("**Evidence (2–5 quotes):**")
                for q in ps.quotes:
                    cite = f" — {q.citation}" if q.citation else ""
                    sec.append(f"> {q.snippet}{cite}")
            parts += sec + [""]

        if self.comparison_table_md:
            parts += ["### Comparison", self.comparison_table_md.strip(), ""]

        if self.comparative_analysis:
            parts += ["### Comparative analysis", self.comparative_analysis.strip(), ""]
        if self.recommendation:
            parts += ["### Recommendation", self.recommendation.strip(), ""]
        if self.general_knowledge:
            parts += ["### General knowledge", self.general_knowledge.strip(), ""]
        if self.gaps_or_next_steps:
            parts += ["### Gaps / Next steps", self.gaps_or_next_steps.strip(), ""]
        if self.ai_opinion:
            parts += ["### AI opinion", self.ai_opinion.strip(), ""]
        if self.citations:
            parts += ["### Citations"]
            for c in self.citations:
                parts.append(f"- {c}")
            parts.append("")

        return "\n".join(parts).strip()

DEFAULT_EMPTY = RegulAIteAnswer(
    summary="No answer was produced.",
    per_source={},
)
