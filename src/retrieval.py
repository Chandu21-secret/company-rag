import uuid
import re
import os

from difflib import SequenceMatcher

from dotenv import load_dotenv
from openai import OpenAI

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL .env mein nahi mila."
    )


if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY .env mein nahi mila."
    )


# ==================================================
# OPENAI
# ==================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==================================================
# QDRANT CLOUD
# ==================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)


COLLECTION_NAME = "company_knowledge"


# ==================================================
# CREATE PAYLOAD INDEXES
# ==================================================

def create_payload_indexes():

    indexed_fields = [
        "category",
        "model",
        "dealer_name",
        "district",
        "state",
        "region",
        "gst_no",
        "mobile",
        "email"
    ]

    for field in indexed_fields:

        try:

            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
                wait=True
            )

            print(
                f"Index ready: {field}"
            )

        except Exception as e:

            error_text = str(e).lower()

            if (
                "already exists" in error_text
                or "already indexed" in error_text
            ):

                print(
                    f"Index already exists: {field}"
                )

            else:

                print(
                    f"Index warning for {field}: {e}"
                )


# ==================================================
# CREATE COLLECTION
# ==================================================

def create_collection():

    collections = qdrant_client.get_collections()

    existing_names = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in existing_names:

        print(
            f"Collection already exists: {COLLECTION_NAME}"
        )

        # Collection already exists.
        # Make sure indexes also exist.
        create_payload_indexes()

        return

    qdrant_client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=1536,

            distance=Distance.COSINE
        )
    )

    print(
        f"Collection created: {COLLECTION_NAME}"
    )

    create_payload_indexes()


# ==================================================
# NORMALIZE TEXT
# ==================================================

def normalize_text(text):

    text = str(text or "").lower()

    text = (
        text
        .replace("\xa0", " ")
        .replace("\u00a0", " ")
        .replace("\ufeff", "")
    )

    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("_", " ")
    )

    text = re.sub(
        r"[^a-z0-9@.+&'\-\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# NORMALIZE MODEL
# ==================================================

def normalize_model(text):

    text = normalize_text(text)

    return re.sub(
        r"[\s\-]+",
        "",
        text
    )


# ==================================================
# SIMILARITY
# ==================================================

def similarity_score(
    text1,
    text2
):

    text1 = normalize_text(
        text1
    )

    text2 = normalize_text(
        text2
    )

    if not text1 or not text2:

        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# ==================================================
# SINGLE EMBEDDING
# ==================================================

def create_embedding(text):

    response = openai_client.embeddings.create(

        model=EMBEDDING_MODEL,

        input=text
    )

    return response.data[0].embedding


# ==================================================
# BATCH EMBEDDINGS
# ==================================================

def create_embeddings_batch(
    texts,
    batch_size=100
):

    all_embeddings = []

    total = len(texts)

    if total == 0:

        return []

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = texts[start:end]

        print(
            f"Embedding batch "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        response = openai_client.embeddings.create(

            model=EMBEDDING_MODEL,

            input=batch
        )

        sorted_data = sorted(

            response.data,

            key=lambda item: item.index
        )

        embeddings = [

            item.embedding

            for item in sorted_data
        ]

        all_embeddings.extend(
            embeddings
        )

    return all_embeddings


# ==================================================
# STORE CHUNKS
# ==================================================

def store_chunks(chunks):

    if not chunks:

        print(
            "No chunks to store."
        )

        return


    print()
    print("================================")
    print("CREATING BATCH EMBEDDINGS")
    print("================================")


    texts = [

        str(
            chunk.get(
                "text",
                ""
            )
        )

        for chunk in chunks
    ]


    embeddings = create_embeddings_batch(

        texts,

        batch_size=100
    )


    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Embedding count does not match "
            "chunk count."
        )


    points = []


    print()
    print(
        "Creating Qdrant points..."
    )


    for index, chunk in enumerate(chunks):

        vector = embeddings[index]


        unique_string = (

            f"{chunk.get('source', '')}-"

            f"{chunk.get('page', '')}-"

            f"{chunk.get('model', '')}-"

            f"{chunk.get('dealer_name', '')}-"

            f"{index}-"

            f"{chunk.get('text', '')}"
        )


        point_id = str(

            uuid.uuid5(

                uuid.NAMESPACE_URL,

                unique_string
            )
        )


        points.append(

            PointStruct(

                id=point_id,

                vector=vector,

                payload={

                    # ----------------------------------
                    # COMMON
                    # ----------------------------------

                    "text": chunk.get(
                        "text",
                        ""
                    ),

                    "source": chunk.get(
                        "source"
                    ),

                    "page": chunk.get(
                        "page"
                    ),

                    "category": chunk.get(
                        "category"
                    ),

                    "model": chunk.get(
                        "model"
                    ),


                    # ----------------------------------
                    # DEALER
                    # ----------------------------------

                    "dealer_name": chunk.get(
                        "dealer_name"
                    ),

                    "gst_no": chunk.get(
                        "gst_no"
                    ),

                    "address": chunk.get(
                        "address"
                    ),

                    "district": chunk.get(
                        "district"
                    ),

                    "state": chunk.get(
                        "state"
                    ),

                    "pin_code": chunk.get(
                        "pin_code"
                    ),

                    "region": chunk.get(
                        "region"
                    ),

                    "rsm": chunk.get(
                        "rsm"
                    ),

                    "contact_person": chunk.get(
                        "contact_person"
                    ),

                    "mobile": chunk.get(
                        "mobile"
                    ),

                    "email": chunk.get(
                        "email"
                    )
                }
            )
        )


    # --------------------------------------------------
    # UPLOAD
    # --------------------------------------------------

    print()
    print(
        "Uploading data to Qdrant Cloud..."
    )


    if points:

        qdrant_client.upsert(

            collection_name=COLLECTION_NAME,

            points=points,

            wait=True
        )


    print()
    print(
        f"{len(points)} chunks stored in Qdrant Cloud."
    )


    # --------------------------------------------------
    # MAKE SURE INDEXES EXIST
    # --------------------------------------------------

    create_payload_indexes()


# ==================================================
# GET ALL DEALERS
# ==================================================

def get_all_dealers():

    dealer_filter = Filter(

        must=[

            FieldCondition(

                key="category",

                match=MatchValue(
                    value="dealer"
                )
            )
        ]
    )


    records, _ = qdrant_client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=dealer_filter,

        limit=1000,

        with_payload=True,

        with_vectors=False
    )


    return records


# ==================================================
# FILTER DEALERS
# ==================================================

def filter_dealers(

    district=None,
    state=None,
    region=None
):

    records = get_all_dealers()

    filtered = []


    district = (

        normalize_text(district)

        if district

        else None
    )


    state = (

        normalize_text(state)

        if state

        else None
    )


    region = (

        normalize_text(region)

        if region

        else None
    )


    for record in records:

        payload = record.payload or {}


        record_district = normalize_text(

            payload.get(
                "district"
            ) or ""
        )


        record_state = normalize_text(

            payload.get(
                "state"
            ) or ""
        )


        record_region = normalize_text(

            payload.get(
                "region"
            ) or ""
        )


        if district:

            if record_district != district:

                continue


        if state:

            if record_state != state:

                continue


        if region:

            if record_region != region:

                continue


        filtered.append(
            record
        )


    return filtered


# ==================================================
# COUNT DEALERS
# ==================================================

def count_dealers(

    district=None,
    state=None,
    region=None
):

    records = filter_dealers(

        district=district,

        state=state,

        region=region
    )


    unique_dealers = set()


    for record in records:

        payload = record.payload or {}


        dealer_name = str(

            payload.get(
                "dealer_name"
            ) or ""

        ).strip()


        if dealer_name:

            unique_dealers.add(

                normalize_text(
                    dealer_name
                )
            )


    return len(unique_dealers)


# ==================================================
# GET DEALER SEARCH TEXT
# ==================================================

def get_dealer_search_text(payload):

    return normalize_text(

        " ".join([

            str(
                payload.get(
                    "dealer_name"
                ) or ""
            ),

            str(
                payload.get(
                    "district"
                ) or ""
            ),

            str(
                payload.get(
                    "state"
                ) or ""
            ),

            str(
                payload.get(
                    "region"
                ) or ""
            ),

            str(
                payload.get(
                    "contact_person"
                ) or ""
            ),

            str(
                payload.get(
                    "mobile"
                ) or ""
            ),

            str(
                payload.get(
                    "email"
                ) or ""
            ),

            str(
                payload.get(
                    "address"
                ) or ""
            ),

            str(
                payload.get(
                    "gst_no"
                ) or ""
            )
        ])
    )


# ==================================================
# EXTRACT SEARCH WORDS
# ==================================================

def extract_search_words(query):

    query = normalize_text(
        query
    )


    words = re.findall(

        r"[a-zA-Z0-9@.+&'\-]+",

        query
    )


    stop_words = {

        "the",

        "mein",
        "me",

        "ke",
        "ka",
        "ki",
        "ko",

        "se",

        "par",

        "kya",
        "hai",
        "hain",

        "kaun",
        "kaunse",

        "kitne",
        "kitna",
        "kitni",

        "kahan",

        "number",

        "mobile",
        "phone",

        "contact",

        "dealer",
        "dealers",

        "mujhe",

        "batao",
        "bataiye",

        "please",

        "email",
        "mail",

        "gst",

        "address",

        "district",
        "state",

        "region",

        "pin",
        "pincode",

        "iska",
        "iski",
        "isko",

        "isme",

        "iske",

        "uska",
        "uski",
        "usko",

        "usme",
        "uske",

        "unka",
        "unki",
        "unke"
    }


    return [

        word

        for word in words

        if word not in stop_words

        and len(word) >= 2
    ]


# ==================================================
# EXTRACT DEALER NAME
# ==================================================

def extract_dealer_candidate(query):

    words = extract_search_words(
        query
    )


    if not words:

        return ""


    extra_stop_words = {

        "batao",
        "bataiye",

        "maximum",
        "power",

        "model",
        "product",
        "products",

        "catalogue",
        "catalog",

        "displacement",

        "hp",
        "cc"
    }


    words = [

        word

        for word in words

        if word not in extra_stop_words
    ]


    return " ".join(
        words
    ).strip()


# ==================================================
# SEARCH DEALERS
# ==================================================

def search_dealers(query):

    query_original = str(
        query or ""
    ).strip()


    query_lower = normalize_text(
        query_original
    )


    records = get_all_dealers()


    if not records:

        return []


    # ==================================================
    # 1. EXACT DEALER NAME
    # ==================================================

    for record in records:

        payload = record.payload or {}


        dealer_name = str(

            payload.get(
                "dealer_name"
            ) or ""

        ).strip()


        if not dealer_name:

            continue


        dealer_normalized = normalize_text(
            dealer_name
        )


        if dealer_normalized in query_lower:

            return [
                record
            ]


    # ==================================================
    # 2. COMPACT MATCH
    # ==================================================

    query_compact = re.sub(

        r"[\s\-]+",

        "",

        query_lower
    )


    if query_compact:

        for record in records:

            payload = record.payload or {}


            dealer_name = str(

                payload.get(
                    "dealer_name"
                ) or ""

            ).strip()


            if not dealer_name:

                continue


            dealer_compact = re.sub(

                r"[\s\-]+",

                "",

                normalize_text(
                    dealer_name
                )
            )


            if (

                dealer_compact

                and dealer_compact in query_compact

            ):

                return [
                    record
                ]


    # ==================================================
    # 3. FUZZY FULL NAME
    # ==================================================

    dealer_candidate = extract_dealer_candidate(
        query_original
    )


    if not dealer_candidate:

        return []


    fuzzy_matches = []


    for record in records:

        payload = record.payload or {}


        dealer_name = str(

            payload.get(
                "dealer_name"
            ) or ""

        ).strip()


        if not dealer_name:

            continue


        dealer_normalized = normalize_text(
            dealer_name
        )


        full_score = similarity_score(

            dealer_candidate,

            dealer_normalized
        )


        query_words = dealer_candidate.split()

        dealer_words = dealer_normalized.split()


        matched_words = 0


        for query_word in query_words:

            best_word_score = 0.0


            for dealer_word in dealer_words:

                current_score = similarity_score(

                    query_word,

                    dealer_word
                )


                if current_score > best_word_score:

                    best_word_score = current_score


            if best_word_score >= 0.82:

                matched_words += 1


        if len(query_words) >= 2:

            if (

                full_score >= 0.78

                and matched_words >= 2

            ):

                fuzzy_matches.append(

                    (
                        full_score,
                        record
                    )
                )


        else:

            if full_score >= 0.90:

                fuzzy_matches.append(

                    (
                        full_score,
                        record
                    )
                )


    # ==================================================
    # BEST FUZZY MATCH ONLY
    # ==================================================

    if fuzzy_matches:

        fuzzy_matches.sort(

            key=lambda item: item[0],

            reverse=True
        )


        return [

            fuzzy_matches[0][1]

        ]


    return []


# ==================================================
# BONHOEFFER SEARCH
# ==================================================

def search_bonhoeffer():

    records, _ = qdrant_client.scroll(

        collection_name=COLLECTION_NAME,

        limit=1000,

        with_payload=True,

        with_vectors=False
    )


    results = []


    for record in records:

        payload = record.payload or {}


        source = str(

            payload.get(
                "source"
            ) or ""

        ).lower()


        text = str(

            payload.get(
                "text"
            ) or ""

        ).lower()


        if (

            "bonhoeffer" in source

            or "bonhoeffer" in text

        ):

            results.append(
                record
            )


    return results


# ==================================================
# SEARCH BONHOEFFER TEXT
# ==================================================

def search_bonhoeffer_text(query):

    records = search_bonhoeffer()


    if not records:

        return []


    query_normalized = normalize_text(
        query
    )


    direct_matches = []


    words = extract_search_words(
        query_normalized
    )


    for record in records:

        payload = record.payload or {}


        text = normalize_text(

            payload.get(
                "text"
            ) or ""
        )


        if not text:

            continue


        score = 0


        for word in words:

            if word in text:

                score += 1


        if score > 0:

            direct_matches.append(

                (
                    score,
                    record
                )
            )


    if direct_matches:

        direct_matches.sort(

            key=lambda item: item[0],

            reverse=True
        )


        return [

            item[1]

            for item in direct_matches
        ]


    return records


# ==================================================
# NORMAL PRODUCT SEARCH
# ==================================================

def search(

    query,

    limit=5,

    category=None,

    model=None
):

    # ==================================================
    # DEALER
    # ==================================================

    if category == "dealer":

        return search_dealers(
            query
        )


    # ==================================================
    # EMBEDDING
    # ==================================================

    query_vector = create_embedding(
        query
    )


    query_filter = None


    # ==================================================
    # MODEL FILTER
    # ==================================================

    if model:

        query_filter = Filter(

            must=[

                FieldCondition(

                    key="model",

                    match=MatchValue(

                        value=model
                    )
                )
            ]
        )


    # ==================================================
    # CATEGORY FILTER
    # ==================================================

    elif (

        category

        and category != "general"

    ):

        query_filter = Filter(

            must=[

                FieldCondition(

                    key="category",

                    match=MatchValue(

                        value=category
                    )
                )
            ]
        )


    # ==================================================
    # QDRANT CLOUD SEARCH
    # ==================================================

    results = qdrant_client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=query_filter,

        limit=limit

    ).points


    # ==================================================
    # MODEL FALLBACK
    # ==================================================

    if (

        model

        and not results

    ):

        results = qdrant_client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            query_filter=None,

            limit=limit

        ).points


    return results