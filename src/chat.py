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

# IMPORTANT:
# Use the low-cost model for this RAG chatbot.
PLANNER_MODEL = "gpt-5.6-luna"
ANSWER_MODEL = "gpt-5.6-luna"

print("================================")
print("COMPANY RAG CHATBOT")
print("================================")
print(f"Planner Model : {PLANNER_MODEL}")
print(f"Answer Model  : {ANSWER_MODEL}")
print(f"Qdrant        : {COLLECTION_NAME}")
print("================================")


# ============================================================
# CONSTANTS
# ============================================================

MAX_HISTORY_TURNS = 2
MAX_RESULTS = 5
MAX_LIST_RESULTS = 20

# Keep retrieved text small.
MAX_TEXT_CHARS = 1800

# Keep previous answers small.
MAX_HISTORY_ANSWER_CHARS = 500


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

    if text.endswith("```"):

        text = re.sub(
            r"```$",
            "",
            text
        )

    return text.strip()


def safe_int(
    value,
    default=5,
    minimum=1,
    maximum=50
):

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


def normalize_text(value: Any) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip()
    )


def unique_preserve_order(items):

    seen = set()
    output = []

    for item in items:

        value = normalize_text(item)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


# ============================================================
# CHAT HISTORY
# ============================================================

def build_history_text(history):

    history = history or []

    recent_history = history[-MAX_HISTORY_TURNS:]

    parts = []

    for item in recent_history:

        if not isinstance(item, dict):
            continue

        question = normalize_text(
            item.get("question", "")
        )

        answer = normalize_text(
            item.get("answer", "")
        )

        if not question:
            continue

        answer = answer[
            :MAX_HISTORY_ANSWER_CHARS
        ]

        parts.append(
            f"User: {question}\n"
            f"Assistant: {answer}"
        )

    return "\n\n".join(parts)


# ============================================================
# OPENAI QUERY PLANNER
# ============================================================

def understand_question(
    question: str,
    history: list | None = None
) -> dict[str, Any]:

    history_text = build_history_text(history)

    prompt = f"""
You are the query-understanding layer of a company RAG system.

Your ONLY job is to create a small retrieval plan.

Do NOT answer the user's question.

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

Available payload fields:

category, model, dealer_name, district, state, region,
gst_no, mobile, email, address, pin_code, rsm,
contact_person, source, page, text.

Return ONLY valid JSON.

Use this structure:

{{
  "intent": "dealer|product|general|unknown",
  "operation": "search|count|list|lookup",
  "search_query": "short semantic search query",
  "filters": {{
    "category": null,
    "model": null,
    "dealer_name": null,
    "district": null,
    "state": null,
    "region": null
  }},
  "limit": 5
}}

Rules:

1. Do not invent information.
2. Use null when a filter is unknown.
3. If user asks "how many", "kitne", "number of", use count.
4. If user asks for models/products/all items, use list.
5. If user asks about one entity, use lookup.
6. Preserve important product names and locations.
7. For follow-up questions, use the conversation history.
8. Keep search_query short.
9. limit should normally be 5.
10. Never answer the user's question.

Conversation history:

{history_text}

Current user question:

{question}
"""

    try:

        response = client.responses.create(
            model=PLANNER_MODEL,
            input=prompt,
            reasoning={
                "effort": "none"
            },
            max_output_tokens=300,
        )

        raw = clean_json(
            response.output_text
        )

        plan = json.loads(raw)

    except Exception as e:

        print("Planner error:", e)

        plan = {
            "intent": "unknown",
            "operation": "search",
            "search_query": question,
            "filters": {},
            "limit": 5
        }

    if not isinstance(plan, dict):

        plan = {
            "intent": "unknown",
            "operation": "search",
            "search_query": question,
            "filters": {},
            "limit": 5
        }

    if not isinstance(
        plan.get("filters"),
        dict
    ):

        plan["filters"] = {}

    plan["search_query"] = normalize_text(
        plan.get(
            "search_query",
            question
        )
    ) or question

    plan["operation"] = normalize_text(
        plan.get(
            "operation",
            "search"
        )
    ).lower() or "search"

    plan["intent"] = normalize_text(
        plan.get(
            "intent",
            "unknown"
        )
    ).lower() or "unknown"

    plan["limit"] = safe_int(
        plan.get("limit", 5),
        default=5,
        minimum=3,
        maximum=MAX_RESULTS
    )

    return plan


# ============================================================
# QDRANT FILTER
# ============================================================

def build_filter(
    filters: dict
):

    must = []

    allowed_fields = [
        "category",
        "model",
        "dealer_name",
        "district",
        "state",
        "region"
    ]

    for field in allowed_fields:

        value = filters.get(field)

        if value is None:
            continue

        value = normalize_text(value)

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
# QDRANT SEMANTIC RETRIEVAL
# ============================================================

def retrieve_documents(
    question: str,
    plan: dict
):

    search_query = (
        plan.get("search_query")
        or question
    )

    filters = plan.get(
        "filters",
        {}
    )

    qdrant_filter = build_filter(
        filters
    )

    # ONE embedding request.
    query_vector = create_embedding(
        search_query
    )

    operation = str(
        plan.get(
            "operation",
            "search"
        )
    ).lower()

    if operation == "list":

        retrieval_limit = MAX_LIST_RESULTS

    else:

        retrieval_limit = plan.get(
            "limit",
            MAX_RESULTS
        )

    retrieval_limit = safe_int(
        retrieval_limit,
        default=MAX_RESULTS,
        minimum=3,
        maximum=MAX_LIST_RESULTS
    )

    # First search with filters.
    results = qdrant_client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=qdrant_filter,

        limit=retrieval_limit,

        with_payload=True,

        with_vectors=False

    ).points

    # If filtered search fails,
    # retry without filter.
    if (
        not results
        and qdrant_filter is not None
    ):

        results = qdrant_client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            query_filter=None,

            limit=retrieval_limit,

            with_payload=True,

            with_vectors=False

        ).points

    return results


# ============================================================
# EXACT RECORD RETRIEVAL
# ============================================================

def retrieve_matching_records(
    plan: dict
):

    filters = plan.get(
        "filters",
        {}
    )

    qdrant_filter = build_filter(
        filters
    )

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

        records.extend(batch)

        if next_offset is None:
            break

        offset = next_offset

        # Safety limit.
        if len(records) >= 5000:
            break

    return records


# ============================================================
# BUILD SMALL CONTEXT
# ============================================================

def build_context(
    results,
    max_results=MAX_RESULTS
):

    if not results:
        return ""

    parts = []

    for index, result in enumerate(
        results[:max_results],
        start=1
    ):

        payload = (
            result.payload
            or {}
        )

        text = normalize_text(
            payload.get("text", "")
        )

        # HARD LIMIT on retrieved text.
        text = text[:MAX_TEXT_CHARS]

        parts.append(
            f"""
--- RESULT {index} ---

Model:
{normalize_text(payload.get("model"))}

Category:
{normalize_text(payload.get("category"))}

Dealer Name:
{normalize_text(payload.get("dealer_name"))}

State:
{normalize_text(payload.get("state"))}

District:
{normalize_text(payload.get("district"))}

Region:
{normalize_text(payload.get("region"))}

Mobile:
{normalize_text(payload.get("mobile"))}

Email:
{normalize_text(payload.get("email"))}

Contact Person:
{normalize_text(payload.get("contact_person"))}

Text:
{text}
"""
        )

    return "\n".join(parts)


# ============================================================
# EXTRACT UNIQUE MODELS
# ============================================================

def extract_unique_models(records):

    models = []

    for record in records:

        payload = (
            record.payload
            or {}
        )

        model = normalize_text(
            payload.get("model")
        )

        if model:
            models.append(model)

    return unique_preserve_order(
        models
    )


# ============================================================
# EXTRACT UNIQUE PRODUCTS
# ============================================================

def extract_unique_products(records):

    products = []

    for record in records:

        payload = (
            record.payload
            or {}
        )

        model = normalize_text(
            payload.get("model")
        )

        category = normalize_text(
            payload.get("category")
        )

        if model:

            if category:
                products.append(
                    f"{model} ({category})"
                )

            else:
                products.append(model)

    return unique_preserve_order(
        products
    )


# ============================================================
# DIRECT COUNT ANSWER
# ============================================================

def answer_count_question(
    question,
    plan
):

    records = retrieve_matching_records(
        plan
    )

    if not records:

        return (
            "I could not find this information "
            "in the company knowledge base."
        )

    models = extract_unique_models(
        records
    )

    # If models exist, count unique models.
    if models:

        count = len(models)

        return (
            f"Company knowledge base ke according "
            f"{count} unique model(s) available hain."
        )

    # Fallback to unique records.
    return (
        f"Company knowledge base me "
        f"{len(records)} matching record(s) mile."
    )


# ============================================================
# DIRECT LIST ANSWER
# ============================================================

def answer_list_question(
    question,
    plan
):

    records = retrieve_matching_records(
        plan
    )

    if not records:

        return (
            "I could not find this information "
            "in the company knowledge base."
        )

    models = extract_unique_models(
        records
    )

    if not models:

        return (
            "I could not find the model names "
            "in the company knowledge base."
        )

    lines = []

    for index, model in enumerate(
        models,
        start=1
    ):

        lines.append(
            f"{index}. {model}"
        )

    return (
        f"Company knowledge base ke according "
        f"{len(models)} unique model(s) hain:\n\n"
        + "\n".join(lines)
    )


# ============================================================
# OPENAI FINAL ANSWER
# ============================================================

def generate_ai_answer(
    question: str,
    context: str,
    history: list | None = None
):

    history_text = build_history_text(
        history
    )

    prompt = f"""
You are the company's AI assistant.

Answer the user's question using ONLY
the supplied company knowledge.

Rules:

1. Company data is the source of truth.
2. Do not invent company information.
3. Do not claim information not supported by context.
4. Understand Hindi, Hinglish and English.
5. Answer in the same language/style as the user.
6. Be concise.
7. If the user asks for a list, use a numbered list.
8. If information is missing, say:

"I could not find this information in the company knowledge base."

9. Do not mention Qdrant.
10. Do not mention embeddings.
11. Do not mention prompts.
12. Do not mention internal retrieval.

Conversation history:

{history_text}

Company knowledge:

{context}

User question:

{question}
"""

    try:

        response = client.responses.create(

            model=ANSWER_MODEL,

            input=prompt,

            reasoning={
                "effort": "none"
            },

            max_output_tokens=600
        )

        # Token monitoring.
        usage = getattr(
            response,
            "usage",
            None
        )

        if usage:

            print(
                "--------------------------------"
            )

            print(
                "OpenAI Usage"
            )

            print(
                "Input tokens:",
                getattr(
                    usage,
                    "input_tokens",
                    "N/A"
                )
            )

            print(
                "Output tokens:",
                getattr(
                    usage,
                    "output_tokens",
                    "N/A"
                )
            )

            print(
                "Total tokens:",
                getattr(
                    usage,
                    "total_tokens",
                    "N/A"
                )
            )

            print(
                "--------------------------------"
            )

        return (
            response.output_text
            or
            "I could not generate an answer."
        ).strip()

    except Exception as e:

        print(
            "Answer generation error:",
            e
        )

        return (
            "Sorry, I could not generate "
            "the answer right now."
        )


# ============================================================
# MAIN CHAT FUNCTION
# ============================================================

def ask_question(
    question,
    chat_history=None
):

    question = normalize_text(
        question
    )

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
    # Small OpenAI planner call
    # --------------------------------------------------------

    plan = understand_question(
        question,
        chat_history
    )

    operation = plan.get(
        "operation",
        "search"
    )

    print(
        "Question:",
        question
    )

    print(
        "Plan:",
        plan
    )

    # --------------------------------------------------------
    # STEP 2
    # COUNT
    #
    # IMPORTANT:
    # Do NOT send all records to OpenAI.
    # --------------------------------------------------------

    if operation == "count":

        return answer_count_question(
            question,
            plan
        )

    # --------------------------------------------------------
    # STEP 3
    # LIST
    #
    # IMPORTANT:
    # Do NOT send all records to OpenAI.
    # --------------------------------------------------------

    if operation == "list":

        return answer_list_question(
            question,
            plan
        )

    # --------------------------------------------------------
    # STEP 4
    # NORMAL SEARCH / LOOKUP
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
    # STEP 5
    # SMALL CONTEXT ONLY
    # --------------------------------------------------------

    context = build_context(
        results,
        max_results=MAX_RESULTS
    )

    # --------------------------------------------------------
    # STEP 6
    # ONE FINAL LLM CALL
    # --------------------------------------------------------

    return generate_ai_answer(
        question,
        context,
        chat_history
    )