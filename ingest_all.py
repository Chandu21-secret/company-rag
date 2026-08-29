from pathlib import Path
import hashlib
import json

from src.ingestion import (
    extract_text_from_pdf,
    create_chunks
)

from src.excel_ingestion import (
    extract_dealers_from_excel
)

from src.retrieval import (
    create_collection,
    store_chunks
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "products"

EXCEL_DIR = DATA_DIR / "dealer"

STATE_FILE = DATA_DIR / "ingestion_state.json"


# ==================================================
# SUPPORTED FILE TYPES
# ==================================================

PDF_EXTENSIONS = {
    ".pdf"
}

EXCEL_EXTENSIONS = {
    ".xlsx",
    ".xls"
}


# ==================================================
# CREATE DIRECTORIES
# ==================================================

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXCEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# FILE HASH
# ==================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:

                break

            sha256.update(
                data
            )

    return sha256.hexdigest()


# ==================================================
# LOAD INGESTION STATE
# ==================================================

def load_state():

    if not STATE_FILE.exists():

        return {}


    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {}


# ==================================================
# SAVE INGESTION STATE
# ==================================================

def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            state,

            file,

            indent=4,

            ensure_ascii=False
        )


# ==================================================
# GET FILES
# ==================================================

def get_files(
    folder,
    extensions
):

    if not folder.exists():

        return []


    return [

        file

        for file in folder.iterdir()

        if (
            file.is_file()
            and
            file.suffix.lower()
            in extensions
        )
    ]


# ==================================================
# CHECK NEW / CHANGED FILE
# ==================================================

def is_new_file(
    file_path,
    state
):

    file_key = str(
        file_path.resolve()
    )

    current_hash = get_file_hash(
        file_path
    )

    old_hash = state.get(
        file_key
    )


    if old_hash == current_hash:

        return False


    return True


# ==================================================
# PROCESS PDF FILE
# ==================================================

def process_pdf(
    pdf_file
):

    print()

    print(
        f"Processing PDF: {pdf_file.name}"
    )


    pages = extract_text_from_pdf(
        str(pdf_file)
    )


    if not pages:

        print(
            "WARNING: No text found."
        )

        return []


    chunks = create_chunks(
        pages
    )


    print(
        f"Pages found: {len(pages)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )


    return chunks


# ==================================================
# PROCESS EXCEL FILE
# ==================================================

def process_excel(
    excel_file
):

    print()

    print(
        f"Processing Excel: {excel_file.name}"
    )


    dealers = extract_dealers_from_excel(

        str(excel_file)

    )


    print(
        f"Dealer records found: {len(dealers)}"
    )


    # Make source equal to actual filename

    for dealer in dealers:

        dealer["source"] = (
            excel_file.name
        )


    return dealers


# ==================================================
# MAIN
# ==================================================

def main():

    print()

    print(
        "================================"
    )

    print(
        "COMPANY RAG DATA INGESTION"
    )

    print(
        "================================"
    )


    # ----------------------------------------------
    # Qdrant
    # ----------------------------------------------

    print()

    print(
        "Checking Qdrant collection..."
    )


    create_collection()


    # ----------------------------------------------
    # Load state
    # ----------------------------------------------

    state = load_state()


    all_chunks = []

    processed_files = []

    skipped_files = []


    # ==================================================
    # PDF FILES
    # ==================================================

    print()

    print(
        "================================"
    )

    print(
        "CHECKING PDF FILES"
    )

    print(
        "================================"
    )


    pdf_files = get_files(

        PDF_DIR,

        PDF_EXTENSIONS
    )


    if not pdf_files:

        print()

        print(
            "No PDF files found."
        )

        print(
            f"Folder: {PDF_DIR}"
        )


    for pdf_file in pdf_files:

        if not is_new_file(
            pdf_file,
            state
        ):

            print()

            print(
                f"SKIP: {pdf_file.name}"
            )

            print(
                "Already ingested."
            )

            skipped_files.append(
                str(pdf_file)
            )

            continue


        try:

            chunks = process_pdf(
                pdf_file
            )


            if chunks:

                all_chunks.extend(
                    chunks
                )

                processed_files.append(
                    pdf_file
                )


        except Exception as e:

            print()

            print(
                f"ERROR: {pdf_file.name}"
            )

            print(e)


    # ==================================================
    # EXCEL FILES
    # ==================================================

    print()

    print(
        "================================"
    )

    print(
        "CHECKING EXCEL FILES"
    )

    print(
        "================================"
    )


    excel_files = get_files(

        EXCEL_DIR,

        EXCEL_EXTENSIONS
    )


    if not excel_files:

        print()

        print(
            "No Excel files found."
        )

        print(
            f"Folder: {EXCEL_DIR}"
        )


    for excel_file in excel_files:

        if not is_new_file(
            excel_file,
            state
        ):

            print()

            print(
                f"SKIP: {excel_file.name}"
            )

            print(
                "Already ingested."
            )

            skipped_files.append(
                str(excel_file)
            )

            continue


        try:

            dealers = process_excel(
                excel_file
            )


            if dealers:

                all_chunks.extend(
                    dealers
                )

                processed_files.append(
                    excel_file
                )


        except Exception as e:

            print()

            print(
                f"ERROR: {excel_file.name}"
            )

            print(e)


    # ==================================================
    # NOTHING NEW
    # ==================================================

    if not all_chunks:

        print()

        print(
            "================================"
        )

        print(
            "NO NEW DATA"
        )

        print(
            "================================"
        )

        print()

        print(
            "All files are already ingested,"
        )

        print(
            "or no files were found."
        )

        print()

        return


    # ==================================================
    # STORE DATA
    # ==================================================

    print()

    print(
        "================================"
    )

    print(
        "CREATING EMBEDDINGS"
    )

    print(
        "================================"
    )


    print()

    print(
        f"New records/chunks: {len(all_chunks)}"
    )


    store_chunks(
        all_chunks
    )


    # ==================================================
    # UPDATE STATE
    # ==================================================

    for file_path in processed_files:

        file_key = str(
            file_path.resolve()
        )

        state[file_key] = get_file_hash(
            file_path
        )


    save_state(
        state
    )


    # ==================================================
    # SUMMARY
    # ==================================================

    print()

    print(
        "================================"
    )

    print(
        "INGESTION COMPLETED"
    )

    print(
        "================================"
    )

    print()

    print(
        f"New files processed: "
        f"{len(processed_files)}"
    )

    print(
        f"Files skipped: "
        f"{len(skipped_files)}"
    )

    print(
        f"New records/chunks: "
        f"{len(all_chunks)}"
    )

    print()

    print(
        "Ingestion state saved to:"
    )

    print(
        STATE_FILE
    )

    print()


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()