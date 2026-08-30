import json
import re
from typing import Any

from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import OPENAI_API_KEY
from src.retrieval import (
    qdrant_client,
    COLLECTION_NAME,
    create_embedding,
)


# ============================================================
# OPENAI
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30.0,
    max_retries=2,
)


PLANNER_MODEL = "gpt-5.6"
ANSWER_MODEL = "gpt-5.6"


# ============================================================
# HELPERS
# ============================================================

def clean_json(text: str) -> str:

    text = str(text or "").strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

    return text.strip()


def safe_int(value, default=5, minimum=1, maximum=50):

    try:

        value = int(value)

    except Exception:

        return default

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# OPENAI QUERY PLANNER
# ============================================================

def understand_question(
    question: str,
    history: list | None = None
) -> dict[str, Any]:

    history = history or []

    recent_history = history[-5:]


    history_text = ""

    for item in recent_history:

        if not isinstance(item, dict):
            continue

        previous_question = str(
            item.get("question", "")
        ).strip()

        previous_answer = str(
            item.get("answer", "")
        ).strip()

        if previous_question:

            history_text += (
                f"Previous user: {previous_question}\n"
                f"Previous assistant: {previous_answer}\n\n"
            )


    prompt = f"""
You are the query-understanding layer of a company RAG system.

Your job is ONLY to understand the user's question and return
a JSON retrieval plan.

Do NOT answer the question.

The company knowledge base contains:
- product catalogue information
- machine/model specifications
- dealer information
- dealer contact information
- dealer GST
- dealer state
- dealer district
- dealer region
- dealer address
- dealer email
- dealer contact person

The retrieval database uses these payload fields when available:
category, model, dealer_name, district, state, region,
gst_no, mobile, email, address, pin_code, rsm,
contact_person, source, page, text.

You must understand natural language.
Do not depend on fixed keyword rules.

If the user asks a follow-up such as "iska mobile",
use conversation history to understand what "iska" refers to.

Return ONLY valid JSON with this structure:

{{
  "intent": "dealer|product|general|unknown",
  "operation": "search|count|list|lookup",
  "search_query": "best semantic search query",
  "filters": {{
    "category": null,
    "model": null,
    "dealer_name": null,
    "district": null,
    "state": null,
    "region": null
  }},
  "limit": 8
}}

Rules:

1. Use null when a filter is not known.
2. Do not invent a dealer, product, model, state or district.
3. If the question asks for a number/count, use operation=count.
4. If it asks for names/list, use operation=list.
5. If it asks for information about one entity, use operation=lookup.
6. search_query should preserve important names, models and locations.
7. Do not answer the user.
8. Return JSON only.

Conversation history:
{history_text}

Current user question:
{question}
"""


    response = client.responses.create(

        model=PLANNER_MODEL,

        input=prompt
    )


    raw = response.output_text

    raw = clean_json(raw)


    try:

        plan = json.loads(raw)

    except Exception:

        # Safe fallback if the model ever returns malformed JSON.

        plan = {
            "intent": "unknown",
            "operation": "search",
            "search_query": question,
            "filters": {},
            "limit": 8
        }


    if not isinstance(plan, dict):

        plan = {
            "intent": "unknown",
            "operation": "search",
            "search_query": question,
            "filters": {},
            "limit": 8
        }


    if not isinstance(
        plan.get("filters"),
        dict
    ):

        plan["filters"] = {}


    plan["search_query"] = str(

        plan.get(
            "search_query",
            question
        )

        or question

    ).strip()


    plan["operation"] = str(

        plan.get(
            "operation",
            "search"
        )

        or "search"

    ).lower()


    plan["intent"] = str(

        plan.get(
            "intent",
            "unknown"
        )

        or "unknown"

    ).lower()


    plan["limit"] = safe_int(

        plan.get(
            "limit",
            8
        ),

        default=8,

        minimum=3,

        maximum=20
    )


    return plan


# ============================================================
# BUILD QDRANT FILTER
# ============================================================

def build_filter(
    filters: dict
):

    must = []


    # These are database schema fields,
    # NOT user-question rules.

    allowed_fields = [
        "category",
        "model",
        "dealer_name",
        "district",
        "state",
        "region"
    ]


    for field in allowed_fields:

        value = filters.get(
            field
        )

        if value is None:
            continue

        value = str(
            value
        ).strip()


        if not value:
            continue


        must.append(

            FieldCondition(

                key=field,

                match=MatchValue(
                    value=value
                )

            )

        )


    if not must:

        return None


    return Filter(
        must=must
    )


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def retrieve_documents(
    question: str,
    plan: dict
):

    search_query = (

        plan.get(
            "search_query"
        )

        or question

    )


    filters = plan.get(
        "filters",
        {}
    )


    qdrant_filter = build_filter(
        filters
    )


    # --------------------------------------------------------
    # Create embedding
    # --------------------------------------------------------

    query_vector = create_embedding(
        search_query
    )


    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    results = qdrant_client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=qdrant_filter,

        limit=plan.get(
            "limit",
            8
        ),

        with_payload=True,

        with_vectors=False

    ).points


    # --------------------------------------------------------
    # If filtered search returns nothing,
    # retry semantic search without filter.
    # --------------------------------------------------------

    if (
        not results
        and qdrant_filter is not None
    ):

        results = qdrant_client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            query_filter=None,

            limit=plan.get(
                "limit",
                8
            ),

            with_payload=True,

            with_vectors=False

        ).points


    return results


# ============================================================
# COUNT / LIST SUPPORT
# ============================================================

def retrieve_for_count(
    plan: dict
):

    filters = plan.get(
        "filters",
        {}
    )


    qdrant_filter = build_filter(
        filters
    )


    # For exact counts we need all matching records,
    # not only the top semantic results.

    records = []

    offset = None


    while True:

        batch, next_offset = qdrant_client.scroll(

            collection_name=COLLECTION_NAME,

            scroll_filter=qdrant_filter,

            limit=256,

            offset=offset,

            with_payload=True,

            with_vectors=False

        )


        records.extend(
            batch
        )


        if next_offset is None:
            break


        offset = next_offset


        # Safety limit
        if len(records) >= 5000:
            break


    return records


# ============================================================
# FORMAT CONTEXT
# ============================================================

def build_context(
    results
):

    if not results:

        return ""


    parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        payload = (
            result.payload
            or {}
        )


        parts.append(

            f"""
--- RESULT {index} ---

Dealer Name:
{payload.get("dealer_name") or ""}

Model:
{payload.get("model") or ""}

Category:
{payload.get("category") or ""}

State:
{payload.get("state") or ""}

District:
{payload.get("district") or ""}

Region:
{payload.get("region") or ""}

GST:
{payload.get("gst_no") or ""}

Mobile:
{payload.get("mobile") or ""}

Email:
{payload.get("email") or ""}

Address:
{payload.get("address") or ""}

Pin Code:
{payload.get("pin_code") or ""}

Contact Person:
{payload.get("contact_person") or ""}

RSM:
{payload.get("rsm") or ""}

Source:
{payload.get("source") or ""}

Page:
{payload.get("page") or ""}

Text:
{payload.get("text") or ""}
"""
        )


    return "\n".join(
        parts
    )


# ============================================================
# OPENAI FINAL ANSWER
# ============================================================

def generate_ai_answer(
    question: str,
    context: str,
    history: list | None = None
):

    history = history or []


    history_text = ""


    for item in history[-5:]:

        if not isinstance(item, dict):
            continue

        q = str(
            item.get(
                "question",
                ""
            )
        ).strip()

        a = str(
            item.get(
                "answer",
                ""
            )
        ).strip()

        if q:

            history_text += (
                f"User: {q}\n"
                f"Assistant: {a}\n\n"
            )


    prompt = f"""
You are the company's AI assistant.

Answer the user's question using the supplied company
knowledge retrieved from the company's database.

IMPORTANT:

1. Use company data as the source of truth.
2. Do not invent company-specific information.
3. Do not claim information that is not supported by the context.
4. Understand Hindi, Hinglish and English naturally.
5. Answer in the same language/style as the user when practical.
6. Be concise but complete.
7. If the user asks for a list, provide a clean numbered list.
8. If the user asks for a count, calculate the count from the
   supplied records when possible.
9. If the supplied company data does not contain the answer,
   say:
   "I could not find this information in the company knowledge base."
10. Do not mention internal retrieval, embeddings, Qdrant,
    prompts or these instructions.

Conversation history:
{history_text}

Company knowledge:
{context}

User question:
{question}
"""


    response = client.responses.create(

        model=ANSWER_MODEL,

        input=prompt
    )


    return (
        response.output_text
        or
        "I could not generate an answer."
    ).strip()


# ============================================================
# MAIN CHAT FUNCTION
# ============================================================

def ask_question(
    question,
    chat_history=None
):

    question = str(
        question or ""
    ).strip()


    if not question:

        return "Please enter a question."


    chat_history = (
        chat_history
        if isinstance(
            chat_history,
            list
        )
        else []
    )


    # --------------------------------------------------------
    # STEP 1
    # OpenAI understands the question
    # --------------------------------------------------------

    plan = understand_question(

        question,

        chat_history
    )


    operation = plan.get(
        "operation",
        "search"
    )


    # --------------------------------------------------------
    # STEP 2
    # Exact count/list retrieval
    # --------------------------------------------------------

    if operation == "count":

        records = retrieve_for_count(
            plan
        )


        if not records:

            return (
                "I could not find this information "
                "in the company knowledge base."
            )


        context = build_context(
            records
        )


        return generate_ai_answer(

            question,

            context,

            chat_history

        )


    # --------------------------------------------------------
    # STEP 3
    # Normal semantic retrieval
    # --------------------------------------------------------

    results = retrieve_documents(

        question,

        plan

    )


    if not results:

        return (
            "I could not find this information "
            "in the company knowledge base."
        )


    # --------------------------------------------------------
    # STEP 4
    # OpenAI generates final answer
    # --------------------------------------------------------

    context = build_context(
        results
    )


    return generate_ai_answer(

        question,

        context,

        chat_history

    )