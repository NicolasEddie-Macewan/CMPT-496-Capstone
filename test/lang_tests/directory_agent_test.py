import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import pytest
from agent.directory_agent import DirectoryAgent
from agent.structured_output.directory_output import ContextAnalysisOutput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from collections import deque

class FakeCollection:
    def __init__(self, documents, metadatas):
        self._documents = documents
        self._metadatas = metadatas

    def query(self, query_texts, n_results):
        return {
            "documents": [self._documents[:n_results]],
            "metadatas": [self._metadatas[:n_results]],
        }
    
class FakeStructuredLLM:
    def __init__(self, output):
        self.output = output

    def invoke(self, prompt):
        return self.output

class FakeLLMWithStructuredOutput:
    def __init__(self, output):
        self.output = output

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self.output)

@pytest.fixture
def directory_agent():
    responses = [
        AIMessage(content="Summary for directory 1..."),
        AIMessage(content="Summary for directory 2...")]
    return DirectoryAgent(model=GenericFakeChatModel(messages=iter(responses)))

def test_crawler_node(directory_agent):
    current_dir = Path(__file__).parent
    test_codebase_path = str(current_dir / "TestCodebase")

    initial_state = {
        "directory_path": test_codebase_path,
        "files": deque()
    }

    result = directory_agent.crawler_node(initial_state)
    
    # Check that crawler found 5 dirs, so excluded the obj, bin, .vscode in TestCodebase
    assert "directories" in result
    assert result["total_number_of_directories"] == 3 # ignore obj, bin

def test_retriever_node(directory_agent):
    root_dir = "/repo"
    current_dir = "/repo/src/parser"

    code_collection = FakeCollection(
        documents=[
            "def parse(): pass",
            "def unrelated(): pass",
        ],
        metadatas=[
            {
                "file": "src/parser/main.py",
                "container": "function",
                "name": "parse",
                "type": "function",
                "namespace": "src.parser.main",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "file": "src/other/utils.py",
                "container": "function",
                "name": "unrelated",
                "type": "function",
                "namespace": "src.other.utils",
                "start_line": 1,
                "end_line": 2,
            },
        ],
    )

    summary_collection = FakeCollection(
        documents=[
            "Parser module handles parsing tokens into AST.",
            "Other module utility summary.",
        ],
        metadatas=[
            {
                "path": "src/parser/main.py",
                "type": "file",
                "name": "main.py",
                "parent": "src/parser",
            },
            {
                "path": "src/other/utils.py",
                "type": "file",
                "name": "utils.py",
                "parent": "src/other",
            },
        ],
    )

    state = {
        "directory_path": root_dir,
        "directories": deque([current_dir]),
        "code_context": [],
        "summary_context": [],
        "sufficient_code_context": False,
        "sufficient_summary_context": False,
        "codebase_k": 10,
        "file_summary_k": 10,
        "current_directory": "",
        "codebase_name": "repo",
        "total_number_of_directories": 1,
        "code_collection": code_collection,
        "summary_collection": summary_collection,
    }

    result = directory_agent.retriever_node(state)

    assert result["current_directory"] == current_dir
    assert len(result["code_context"]) == 2
    assert len(result["summary_context"]) == 2

    # The first code item should be the in-directory one
    assert "File: src/parser/main.py" in result["code_context"][0]

    # The first summary item should be the in-directory one
    assert "Path: src/parser/main.py" in result["summary_context"][0]

def test_context_analyser_node_sufficient(directory_agent):
    directory_agent.llm = FakeLLMWithStructuredOutput(
        ContextAnalysisOutput(
            sufficient_code_context=True,
            sufficient_summary_context=True,
            recommended_codebase_k_increase=0,
            recommended_file_summary_k_increase=0,
        )
    )

    state = {
        "directory_path": "/repo",
        "directories": deque(["/repo/src/parser"]),
        "code_context": ["[CODE CHUNK]\nFile: src/parser/main.py\nContent:\ndef parse(): pass"],
        "summary_context": ["[SUMMARY NODE]\nPath: src/parser/main.py\nContent:\nParser summary"],
        "sufficient_code_context": False,
        "sufficient_summary_context": False,
        "codebase_k": 10,
        "file_summary_k": 10,
        "current_directory": "/repo/src/parser",
        "codebase_name": "repo",
        "total_number_of_directories": 1,
        "code_collection": None,
        "summary_collection": None,
    }

    result = directory_agent.context_analyser_node(state)

    assert result["sufficient_code_context"] is True
    assert result["sufficient_summary_context"] is True
    assert result["codebase_k"] == 10
    assert result["file_summary_k"] == 10
