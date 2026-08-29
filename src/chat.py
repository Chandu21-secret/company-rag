import re
import json

from openai import OpenAI

from src.config import OPENAI_API_KEY

from src.ingestion import (
    detect_category,
    detect_models
)

from src.retrieval import (
    search,
    search_bonhoeffer,
    filter_dealers,
    count_dealers,
    search_dealers
)

from src.llm import generate_answer


# ==================================================
# OPENAI
# ==================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

QUERY_MODEL = "gpt-4o-mini"


# ==================================================
# CONSTANTS
# ==================================================

COUNT_WORDS = [
    "kitne dealer",
    "kitna dealer",
    "kitni dealer",
    "kitne dealers",
    "kitna dealers",
    "kitni dealers",
    "number of dealers",
    "how many dealers",
    "total dealers",
    "total dealer",
    "dealer count",
    "dealers count",
    "dealer ki sankhya",
    "dealers ki sankhya"
]


LIST_WORDS = [
    "kaun kaun",
    "kaun-kaun",
    "kon kon",
    "kon-kon",
    "kaunse",
    "kaun se",
    "which dealers",
    "dealer batao",
    "dealers batao",
    "dealer dikhao",
    "dealers dikhao",
    "dealer list",
    "list of dealers",
    "naam batao",
    "naam dikhao",
    "names batao",
    "names dikhao"
]


FOLLOWUP_WORDS = [
    "iska",
    "iski",
    "isko",
    "isme",
    "iske",
    "is model",
    "is product",
    "uska",
    "uski",
    "usko",
    "usme",
    "uske",
    "unka",
    "unki",
    "unke"
]


BONHOEFFER_WORDS = [
    "bonhoeffer",
    "multi-tool",
    "multi tool",
    "multitool",
    "multiفtool",
    "euro-trim",
    "euro trim",
    "euroفtrim",
    "hedge trimmer",
    "hedge trim"
]


STATE_NAMES = [
    "jammu & kashmir",
    "jammu and kashmir",
    "himachal pradesh",
    "karnataka",
    "haryana",
    "punjab",
    "rajasthan",
    "uttar pradesh",
    "uttarakhand",
    "delhi",
    "maharashtra",
    "madhya pradesh",
    "gujarat",
    "bihar",
    "jharkhand",
    "odisha",
    "west bengal",
    "assam",
    "telangana",
    "andhra pradesh",
    "tamil nadu",
    "kerala",
    "goa",
    "chhattisgarh"
]


DISTRICT_NAMES = [
    "jammu",
    "mandya",
    "hamirpur",
    "gurugram",
    "faridabad",
    "panchkula",
    "fatehabad",
    "ambala",
    "karnal"
]


REGION_NAMES = [
    "north",
    "south",
    "east",
    "west",
    "central"
]


GENERAL_MESSAGES = [
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "namaste",
    "namaskar",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "how are u",
    "how r you",
    "how r u",
    "kaise ho",
    "kese ho",
    "kaisa ho",
    "kya haal hai",
    "kya hal hai",
    "thank you",
    "thanks",
    "bye",
    "goodbye"
]


# ==================================================
# NORMALIZE
# ==================================================

def normalize_text(text):

    text = str(text or "").lower()

    text = (
        text
        .replace("\xa0", " ")
        .replace("\u00a0", " ")
        .replace("\ufeff", "")
    )

    text = text.replace("-", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# CONTAINS ANY
# ==================================================

def contains_any(
    text,
    words
):

    return any(
        word in text
        for word in words
    )


# ==================================================
# GENERAL CONVERSATION
# ==================================================

def is_general_conversation(text):

    text = normalize_text(text)

    if text in GENERAL_MESSAGES:
        return True

    phrases = [
        "how are you",
        "how are u",
        "how r you",
        "how r u",
        "kaise ho",
        "kese ho",
        "kaisa ho",
        "kya haal hai",
        "kya hal hai",
        "what are you doing",
        "who are you",
        "what can you do"
    ]

    return contains_any(
        text,
        phrases
    )


# ==================================================
# GENERAL ANSWER
# ==================================================

def generate_general_answer(
    question,
    chat_history=None
):

    history = ""

    if chat_history:

        for item in chat_history[-3:]:

            history += (
                f"User: {item.get('question', '')}\n"
                f"AI: {item.get('answer', '')}\n"
            )

    try:

        response = openai_client.chat.completions.create(

            model=QUERY_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are a friendly company chatbot. "
                        "Answer casual conversation naturally "
                        "and briefly."
                    )
                },

                {
                    "role": "user",
                    "content": (
                        f"Previous conversation:\n{history}\n"
                        f"User: {question}"
                    )
                }
            ],

            temperature=0.3,

            max_tokens=100
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        print(
            "General answer error:",
            e
        )

        return "Hello! How can I help you?"


# ==================================================
# LOCATION
# ==================================================

def detect_location(text):

    text = normalize_text(text)

    district = None
    state = None
    region = None

    for name in STATE_NAMES:

        if name in text:

            state = name

            break


    for name in DISTRICT_NAMES:

        if name in text:

            district = name

            break


    if state in [
        "jammu & kashmir",
        "jammu and kashmir"
    ]:

        if (
            "jammu district" not in text
            and "jammu mein" not in text
            and "jammu me" not in text
            and "jammu ke" not in text
            and "jammu ka" not in text
        ):

            district = None


    for name in REGION_NAMES:

        if name in text:

            region = name

            break


    return (
        district,
        state,
        region
    )


# ==================================================
# FIELD
# ==================================================

def detect_field(text):

    text = normalize_text(text)

    if (
        "mobile number" in text
        or "mobile" in text
        or "phone number" in text
        or "phone" in text
        or "contact number" in text
    ):
        return "mobile"


    if (
        "gst number" in text
        or "gst no" in text
        or "gst" in text
    ):
        return "gst_no"


    if (
        "email address" in text
        or "email id" in text
        or "email" in text
        or "mail id" in text
        or "mail" in text
    ):
        return "email"


    if (
        "address" in text
        or "pata" in text
    ):
        return "address"


    if (
        "contact person" in text
        or "contact name" in text
    ):
        return "contact_person"


    if "district" in text:
        return "district"


    if "state" in text:
        return "state"


    if (
        "pin code" in text
        or "pincode" in text
    ):
        return "pin_code"


    if "region" in text:
        return "region"


    if "rsm" in text:
        return "rsm"


    return None


# ==================================================
# GET DEALER NAME
# ==================================================

def get_dealer_name(result):

    payload = result.payload or {}

    return str(
        payload.get(
            "dealer_name",
            ""
        ) or ""
    ).strip()


# ==================================================
# BUILD CONTEXT
# ==================================================

def build_context(results):

    parts = []

    for result in results:

        payload = result.payload or {}

        parts.append(
            f"""
Source: {payload.get('source', '')}
Page: {payload.get('page', '')}

{payload.get('text', '')}
"""
        )

    return "\n\n".join(parts)


# ==================================================
# BONHOEFFER EMAIL
# ==================================================

def get_bonhoeffer_email(results):

    pattern = (
        r"[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    for result in results:

        payload = result.payload or {}

        text = str(
            payload.get(
                "text",
                ""
            )
        )

        match = re.search(
            pattern,
            text
        )

        if match:

            return match.group(0)

    return None


# ==================================================
# BONHOEFFER PHONE
# ==================================================

def get_bonhoeffer_phone(results):

    for result in results:

        payload = result.payload or {}

        text = str(
            payload.get(
                "text",
                ""
            )
        )

        text = (
            text
            .replace("\xa0", " ")
            .replace("\u00a0", " ")
        )


        match = re.search(
            r"\+91\s*-\s*(\d{5})\s+(\d{5})",
            text
        )

        if match:

            return (
                "+91-"
                + match.group(1)
                + " "
                + match.group(2)
            )


        match = re.search(
            r"\+91\s*-?\s*(\d{10})",
            text
        )

        if match:

            number = match.group(1)

            return (
                "+91-"
                + number[:5]
                + " "
                + number[5:]
            )


    return None


# ==================================================
# BONHOEFFER PRODUCTS
# ==================================================

def get_bonhoeffer_products(results):

    products = []

    for result in results:

        payload = result.payload or {}

        text = normalize_text(
            payload.get(
                "text",
                ""
            )
        )


        if (
            "multi tool" in text
            or "multitool" in text
            or "multiفtool" in text
        ):

            if "Multi-Tool" not in products:

                products.append(
                    "Multi-Tool"
                )


        if (
            "euro trim" in text
            or "euroفtrim" in text
        ):

            if "Euro-Trim" not in products:

                products.append(
                    "Euro-Trim"
                )


        if (
            "hedge trimmer" in text
            or (
                "hedge" in text
                and "trimmer" in text
            )
        ):

            if "Hedge Trimmer" not in products:

                products.append(
                    "Hedge Trimmer"
                )


    return products


# ==================================================
# PRODUCT ANSWER
# ==================================================

def generate_product_answer(question, results):
    if not results:
        return "I could not find this information in the company knowledge base."

    q = normalize_text(question)
    context = build_context(results)

    # Fast direct answers: no LLM call.
    if "power" in q or "maximum power" in q or "horse power" in q or "hp" in q:
        match = re.search(r"(\d+(?:\.\d+)?)\s*HP", context, re.IGNORECASE)
        if match:
            return f"{match.group(1)} HP"

    if "cc" in q or "displacement" in q:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:CC|cc)", context)
        if match:
            return f"{match.group(1)} CC"

    if "rpm" in q:
        match = re.search(r"(\d+(?:,\d+)*)\s*RPM", context, re.IGNORECASE)
        if match:
            return f"{match.group(1)} RPM"

    if "weight" in q:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kgs)", context, re.IGNORECASE)
        if match:
            return f"{match.group(1)} kg"

    prompt = f"""Answer the user's product question using ONLY the company data below.

Rules:
- Give the exact information from the context.
- Do not invent information.
- Keep the answer short and direct.
- Never return dots such as ... or **...**.
- If the answer is not available, say exactly: I could not find this information in the company knowledge base.

Company data:
{context}

Question:
{question}
"""

    try:
        response = openai_client.chat.completions.create(
            model=QUERY_MODEL,
            messages=[
                {"role": "system", "content": "You answer company product questions using only supplied data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=80
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer or answer in ("...", "**...**", "…", "**…**"):
            return "I could not find this information in the company knowledge base."
        return answer
    except Exception as e:
        print("Product answer error:", e)
        return "I could not find this information in the company knowledge base."


# ==================================================
# MAIN CHAT FUNCTION
# ==================================================

def ask_question(
    question,
    chat_history=None
):

    question = str(
        question or ""
    ).strip()


    if not question:

        return "Please enter a question."


    if chat_history is None:

        chat_history = []


    q = normalize_text(
        question
    )


    # ==================================================
    # GENERAL
    # ==================================================

    if is_general_conversation(q):

        return generate_general_answer(
            question,
            chat_history
        )


    # ==================================================
    # LOCATION
    # ==================================================

    district, state, region = detect_location(
        q
    )


    # ==================================================
    # DEALER COUNT
    # ==================================================

    if contains_any(
        q,
        COUNT_WORDS
    ):

        total = count_dealers(

            district=district,

            state=state,

            region=region
        )


        if district:

            return (
                f"{district.title()} mein "
                f"{total} dealers hain."
            )


        if state:

            return (
                f"{state.title()} mein "
                f"{total} dealers hain."
            )


        if region:

            return (
                f"{region.title()} region mein "
                f"{total} dealers hain."
            )


        return (
            f"Company mein total "
            f"{total} dealers hain."
        )


    # ==================================================
    # DEALER LIST
    # ==================================================

    if contains_any(
        q,
        LIST_WORDS
    ):

        # ----------------------------------------------
        # If location missing, use previous question
        # ----------------------------------------------

        if (
            not district
            and not state
            and not region
            and chat_history
        ):

            previous = chat_history[-1].get(
                "question",
                ""
            )

            district, state, region = detect_location(
                previous
            )


        results = filter_dealers(

            district=district or None,

            state=state or None,

            region=region or None
        )


        if not results:

            return (
                "I could not find this information "
                "in the company knowledge base."
            )


        names = []


        for result in results:

            name = get_dealer_name(
                result
            )

            if name:

                names.append(
                    name
                )


        # Remove duplicates

        names = list(
            dict.fromkeys(
                names
            )
        )


        if not names:

            return (
                "I could not find this information "
                "in the company knowledge base."
            )


        if district:

            location = district.title()

        elif state:

            location = state.title()

        elif region:

            location = (
                region.title()
                + " Region"
            )

        else:

            location = "Company"


        lines = []


        for index, name in enumerate(
            names,
            start=1
        ):

            lines.append(
                f"{index}. {name}"
            )


        return (
            f"{location} mein "
            f"{len(names)} dealers hain:\n\n"
            + "\n".join(lines)
        )


    # ==================================================
    # DEALER FIELD
    # ==================================================

    field = detect_field(
        q
    )


    if field:

        results = []


        # ----------------------------------------------
        # Follow-up
        # ----------------------------------------------

        is_followup = contains_any(
            q,
            FOLLOWUP_WORDS
        )


        if (
            is_followup
            and chat_history
        ):

            previous_question = (
                chat_history[-1]
                .get(
                    "question",
                    ""
                )
            )


            results = search_dealers(
                previous_question
            )


        # ----------------------------------------------
        # Normal dealer search
        # ----------------------------------------------

        if not results:

            results = search_dealers(
                question
            )


        if not results:

            return (
                "I could not find this information "
                "in the company knowledge base."
            )


        values = []


        for result in results:

            payload = result.payload or {}


            value = str(
                payload.get(
                    field
                ) or ""
            ).strip()


            if value:

                values.append(
                    value
                )


        values = list(
            dict.fromkeys(
                values
            )
        )


        if len(values) == 1:

            return values[0]


        if values:

            return "\n".join(
                values
            )


        return (
            "I could not find this information "
            "in the company knowledge base."
        )


    # ==================================================
    # BONHOEFFER
    # ==================================================

    if contains_any(
        q,
        BONHOEFFER_WORDS
    ):

        results = search_bonhoeffer()


        if not results:

            return (
                "I could not find this information "
                "in the company knowledge base."
            )


        # ----------------------------------------------
        # Email
        # ----------------------------------------------

        if (
            "email" in q
            or "mail" in q
        ):

            email = get_bonhoeffer_email(
                results
            )

            if email:

                return email


        # ----------------------------------------------
        # Phone
        # ----------------------------------------------

        if (
            "mobile" in q
            or "phone" in q
            or "contact number" in q
        ):

            phone = get_bonhoeffer_phone(
                results
            )

            if phone:

                return phone


        # ----------------------------------------------
        # Product list
        # ----------------------------------------------

        if (
            "product" in q
            or "products" in q
            or "catalog" in q
            or "catalogue" in q
            or "kaun kaun" in q
            or "kaunse" in q
            or "kon kon" in q
        ):

            products = get_bonhoeffer_products(
                results
            )


            if products:

                return (
                    "Bonhoeffer ke catalogue mein "
                    "ye products hain:\n\n"
                    + "\n".join(
                        f"{i}. {product}"
                        for i, product
                        in enumerate(
                            products,
                            start=1
                        )
                    )
                )


        context = build_context(
            results
        )


        return generate_answer(
            question,
            context
        )


    # ==================================================
    # PRODUCT / MODEL
    # ==================================================

    models = detect_models(
        question
    )


    detected_model = (
        models[0]
        if models
        else None
    )


    category = detect_category(
        question
    )


    # ==================================================
    # PRODUCT SEARCH
    # ==================================================

    results = search(

        query=question,

        limit=3,

        category=category,

        model=detected_model
    )


    # ==================================================
    # PRODUCT ANSWER
    # ==================================================

    if results:

        return generate_product_answer(
            question,
            results
        )


    # ==================================================
    # LAST RESORT
    # ==================================================

    return (
        "I could not find this information "
        "in the company knowledge base."
    )
