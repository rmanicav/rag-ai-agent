
from dataclasses import dataclass
from typing import Optional

from tools import calculate, search_documents


@dataclass
class ToolCall:
    tool: str
    argument: Optional[str] = None


def agent_decide(question):
    q = question.lower().strip()

    # Detect arithmetic
    has_operator = any(
        op in q for op in ["+", "*", "/", "%"]
    )

    has_subtraction = (
        "-" in q and any(char.isdigit() for char in q)
    )

    if has_operator or has_subtraction:
        return "CALCULATE"

    # Detect research-paper questions
    paper_keywords = [
        "retrieval",
        "rag",
        "dataset",
        "model",
        "paper",
        "generation",
        "knowledge",
        "architecture",
        "retriever",
        "document"
    ]

    if any(word in q for word in paper_keywords):
        return "SEARCH"

    return "DIRECT"


def structured_decision(question):
    decision = agent_decide(question)

    if decision == "SEARCH":
        return ToolCall(
            tool="search_documents",
            argument=question
        )

    if decision == "CALCULATE":
        expression = (
            question.lower()
            .replace("what is", "")
            .strip()
        )

        return ToolCall(
            tool="calculate",
            argument=expression
        )

    return ToolCall(
        tool="direct_llm",
        argument=question
    )


def execute_tool_call(tool_call, vectorstore):
    if tool_call.tool == "search_documents":
        return search_documents(
            vectorstore,
            tool_call.argument,
            k=3
        )

    if tool_call.tool == "calculate":
        return calculate(tool_call.argument)

    return None
