from typing import TypedDict, List

class GraphState(TypedDict):
    directory: str
    files: List[str] # change to stack
    summary: SummaryOutput # need to creare object
    total_number_of_files: int
