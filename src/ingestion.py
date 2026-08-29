import pymupdf
from pathlib import Path
import re


def extract_text_from_pdf(pdf_path):
    """
    PDF se page-wise text extract karta hai.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append({
                "text": text,
                "page": page_number,
                "source": Path(pdf_path).name
            })

    document.close()

    return pages


def detect_category(text):
    """
    Text ke basis par product/dealer category detect karta hai.
    """

    text_lower = text.lower()

    # Dealer
    if (
        "dealer name" in text_lower
        or "gst number" in text_lower
        or "mobile number" in text_lower
        or "contact person" in text_lower
    ):
        return "dealer"

    # Products
    if "brushcutter" in text_lower or "bruchutter" in text_lower:
        return "brushcutter"

    if "water pump" in text_lower:
        return "water_pump"

    if "chainsaw" in text_lower:
        return "chainsaw"

    if "gasoline engine" in text_lower:
        return "gasoline_engine"

    if "gasoline tiller" in text_lower:
        return "tiller"

    return "general"


def detect_models(text):
    """
    Catalog text mein product model numbers detect karta hai.
    """

    patterns = [
        r"\bMBC[A-Z0-9-]+\b",
        r"\bMWP[A-Z0-9.-]+\b",
        r"\bMCS[A-Z0-9-]+\b",
        r"\bME[A-Z0-9-]+\b",
        r"\bMT[A-Z0-9-]+\b"
    ]

    models = []

    for pattern in patterns:

        found = re.findall(pattern, text.upper())

        for model in found:

            if model not in models:
                models.append(model)

    return models


def create_chunks(pages):
    """
    Product/model-aware chunks create karta hai.
    """

    chunks = []

    for page in pages:

        text = page["text"]

        models = detect_models(text)

        if models:

            for model in models:

                model_position = text.upper().find(model)

                if model_position == -1:
                    continue

                start = max(0, model_position - 500)
                end = min(len(text), model_position + 1200)

                chunk_text = text[start:end].strip()

                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "category": detect_category(chunk_text),
                    "model": model
                })

        else:

            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "category": detect_category(text),
                "model": None
            })

    return chunks