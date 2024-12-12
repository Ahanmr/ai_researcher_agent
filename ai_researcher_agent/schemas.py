from pydantic import BaseModel, Field
from typing import Optional, List

class ResearchTopic(BaseModel):
    topic: str = Field(..., description="The main topic to research")
    context: Optional[str] = Field(None, description="Additional context or specific aspects to focus on")
    depth: Optional[str] = Field("comprehensive", description="Desired depth of research: 'brief', 'moderate', or 'comprehensive'")
    research_objective: Optional[str] = Field(None, description="The main objective of the research")
    specific_focus: Optional[str] = Field(None, description="Any specific focus areas or constraints")

class ResearchOutput(BaseModel):
    keyword_analysis: dict = Field(..., description="Initial keyword analysis and search phrases")
    search_analysis: dict = Field(..., description="Evaluation of search results and refinements")
    final_recommendations: dict = Field(..., description="Synthesized findings and search strategy")
    
class InputSchema(BaseModel):
    research_topic: ResearchTopic
    max_sources: Optional[int] = Field(5, description="Maximum number of sources to consult")
    include_citations: Optional[bool] = Field(True, description="Whether to include citations in the research")
