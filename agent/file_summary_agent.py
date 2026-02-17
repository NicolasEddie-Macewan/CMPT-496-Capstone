"""
@file file_summary_agent.py
@brief Contains the FileSummaryAgent class, which is responsible for summarizing code files in a given directory
@details Wrapper class which contains the graph, its construction and the LLM, alongside the logic of all graph nodes
"""

from states.state1 import GraphState
from langgraph.graph import StateGraph, START, END
from structured_output.summary_output import SummaryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

class FileSummaryAgent:
    """
    @brief Agent responsible for generating function/class level summaries for all files in a given directory
    @param state The initial state of the agent, containing the directory to summarize
    """
    def __init__(self, state: GraphState):
        self.graph = self.build_graph(state)
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            api_key=os.getenv("GOOGLE_API_KEY"))
        self.structured_llm = self.llm.with_structured_output(SummaryOutput)

    def build_graph(self, state: GraphState):
        """
        """
    
    def summary_node(self, state: GraphState) -> SummaryOutput:
        """
        @brief Node which calls the LLM to generate a summary for a single file
        @details Pops a file from the stack, reads its contents, and prompts the LLM to generate a summary. Should loop back to this
        node until the stack is empty
        @param state The current state of the graph
        """
        file = state['files'].pop()
        with open(file, 'r') as f:
            contents = f.read()
        
        messages = [
            ("system", "You are a helpful assistant that creates concise, accurate summaries of code files."),
            ("user", f"Summarize the following code file: \n\n{contents}")
        ]

        return self.structured_llm.invoke(messages)