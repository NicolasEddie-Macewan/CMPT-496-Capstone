from agent.states.state1 import GraphState
from agent.structured_output.summary_output import SummaryOutput

def summary_node(state: GraphState) -> SummaryOutput:
    return state['summary']