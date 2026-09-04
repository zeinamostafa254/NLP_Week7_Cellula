from pydantic import BaseModel, Field
import logging

from document_ai.llm.model import get_llm

logger = logging.getLogger(__name__)

class RewrittenQuery(BaseModel):
    """Schema for the rewritten query and extracted metadata."""
    rewritten_query: str = Field(
        description="A clear, concise, retrieval-friendly version of the user's question. Resolve pronouns using context, expand abbreviations, and preserve the original meaning."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="List of important technical terms and keywords extracted from the query for exact-match searching."
    )
    metadata_filters: dict = Field(
        default_factory=dict,
        description="Any explicit metadata filters requested by the user, such as specific document names, authors, or dates."
    )


SYSTEM_PROMPT = """
You are a query rewriting component in a document retrieval system.

Your job is NOT to answer the user's question.
Your job is to rewrite the user's question so that it is easier for a document retrieval system to search.

Rules:
- Preserve the original meaning.
- Resolve pronouns when possible using the conversation context.
- Expand abbreviations when useful.
- Include important technical terms.
- Do not invent facts.
- Do not answer the question.
- Keep the rewritten query concise.
"""


def rewrite_query(
    query: str,
    conversation_context: str = ""
) -> RewrittenQuery:

    client = get_llm()

    user_prompt = f"""
Conversation context:
{conversation_context}

User question:
{query}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # 1. Use LangChain's structured output parsing which handles JSON schema generation and validation
        # 2. Because RewrittenQuery has detailed Field(description="...") it acts as the prompt!
        structured_llm = client.with_structured_output(RewrittenQuery)
        result = structured_llm.invoke(messages)
        return result
    except Exception as e:
        logger.warning(f"Structured parsing failed, falling back: {e}")
        # Safe fallback if the model fails
        return RewrittenQuery(
            rewritten_query=query,
            keywords=query.split(),
            metadata_filters={},
        )


class QueryRewriter:
    """Class wrapper around rewrite_query for use by RetrieverAgent."""

    def rewrite(self, query: str, conversation_context: str = "") -> str:
        result = rewrite_query(query, conversation_context)
        return result.rewritten_query