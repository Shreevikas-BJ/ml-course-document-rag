import os


def enable_langsmith(project_name: str = "ml-rag-docqa") -> None:
    """
    Enables LangSmith tracing via environment variables.
    Keep it simple: if keys aren't set, it quietly does nothing.
    """
    # Required for tracing
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project_name)

    # Optional but recommended: makes your traces linkable in the UI
    # LANGCHAIN_API_KEY should be set by you in env (we won't hardcode it)
