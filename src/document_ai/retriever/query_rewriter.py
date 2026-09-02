import json

from pydantic import BaseModel, Field

from document_ai.llm.model import get_llm, get_model_name


class RewrittenQuery(BaseModel):
    rewritten_query: str
    keywords: list[str] = Field(default_factory=list)
    metadata_filters: dict = Field(default_factory=dict)


SYSTEM_PROMPT = """
You are a query rewriting component in a document retrieval system.

Your job is NOT to answer the user's question.

Your job is to rewrite the user's question so that it is easier
for a document retrieval system to search.

Return ONLY valid JSON with this structure:

{
    "rewritten_query": "clear retrieval-friendly query",
    "keywords": ["keyword1", "keyword2"],
    "metadata_filters": {}
}

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
    model = get_model_name()

    user_prompt = f"""
Conversation context:
{conversation_context}

User question:
{query}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        return RewrittenQuery.model_validate(data)

    except Exception:
        # Safe fallback if the model does not return valid JSON.
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