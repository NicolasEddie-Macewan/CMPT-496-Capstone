from agent.states.state1 import GraphState
from agent.structured_output.summary_output import SummaryOutput

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"))

def summary_node(state: GraphState) -> SummaryOutput:
    """
    """
    file = state['files'].pop()
    with open(file, 'r') as f:
        contents = f.read()

    structured_llm = llm.with_structured_output(SummaryOutput)
    messages = [
        ("system", "You are a helpful assistant that summarizes code."),
        ("user", f"Summarize the following code file: \n\n{contents}")
    ]

    response = structured_llm.invoke(messages)