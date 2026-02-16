from typing import TypedDict, List
from agent.structured_output.summary_output import SummaryOutput

class GraphState(TypedDict):
    directory: str
    files: List[str] # change to stack
    summary: SummaryOutput
    total_number_of_files: int
