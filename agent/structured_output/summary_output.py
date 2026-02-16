"""
@file summary_output.py
@brief This file defines the SummaryOutput class, which is used to represent the structured output of a summary of a file's contents.
@details This file defines a nested structured output format for LLM summarization of a code file's contents, including separate classes for
function and class summaries
"""

from pydantic import BaseModel, Field

class FunctionSummary(BaseModel):
    """
    @brief A pydantic BaseModel representing a summary of a function defined in a code file.
    """
    name: str = Field(..., description="The name of the function.")
    parameters: list[str] = Field(..., description="A list of parameters that the function takes, if applicable.")
    return_type: str = Field(..., description="The return type of the function, if applicable.")
    description: str = Field(..., description="A concise description of what the function does.")
    calls: list[str] = Field(..., description="A list of other functions that this function calls, if applicable.")

class ClassSummary(BaseModel):
    """
    @brief A pydantic BaseModel representing a summary of a class defined in a code file.
    """
    name: str = Field(..., description="The name of the class.")
    description: str = Field(..., description="A concise description of what the class does.")
    methods: list[FunctionSummary] = Field(..., description="A list of methods defined in this class, if applicable.")

class SummaryOutput(BaseModel):
    """
    @brief A pydantic BaseModel representing a summary of a file's contents.
    """
    summary: str = Field(..., description="A concise summary of the file contents.")
    path: str = Field(..., description="The path of the file being summarized.")
    dependencies: list[str] = Field(..., description="A list of external libraries and imports used in this file, if applicable.")
    external_calls: str = Field(..., description="A list of function calls made in this file, but not within a function body, if applicable.")
    functions: list[FunctionSummary] = Field(..., description="A list of functions defined in this file, but not within a class, if applicable.")
    classes: list[ClassSummary] = Field(..., description="A list of classes defined in this file, if applicable.")