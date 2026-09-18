
import re


def calculate(expression):
    try:
        expression = re.sub(
            r"[^0-9+\-*/().% ]",
            "",
            expression
        )

        return eval(
            expression,
            {"__builtins__": {}},
            {}
        )

    except Exception:
        return "Unable to calculate expression."


def search_documents(vectorstore, query, k=3):
    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page")
        }
        for doc in results
    ]
