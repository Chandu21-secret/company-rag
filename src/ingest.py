from pathlib import Path

from src.ingestion import extract_text_from_pdf, create_chunks
from src.retrieval import create_collection, store_chunks
from src.excel_ingestion import extract_dealers_from_excel


PRODUCTS_FOLDER = Path("data/products")
DEALER_EXCEL = Path("data/dealer/Dealers Database.xlsx")


def ingest_documents():

    pdf_files = list(PRODUCTS_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF file(s).")

    create_collection()

    total_chunks = 0

    # ==============================
    # PROCESS ALL PDF FILES
    # ==============================

    for pdf_path in pdf_files:

        print("\n================================")
        print(f"Processing: {pdf_path.name}")
        print("================================")

        pages = extract_text_from_pdf(str(pdf_path))

        print(f"Pages found: {len(pages)}")

        chunks = create_chunks(pages)

        print(f"Chunks created: {len(chunks)}")

        store_chunks(chunks)

        total_chunks += len(chunks)

    # ==============================
    # PROCESS DEALER EXCEL
    # ==============================

    if DEALER_EXCEL.exists():

        print("\n================================")
        print(f"Processing: {DEALER_EXCEL.name}")
        print("================================")

        dealer_records = extract_dealers_from_excel(
            str(DEALER_EXCEL)
        )

        print(
            f"Dealer records found: {len(dealer_records)}"
        )

        store_chunks(dealer_records)

        total_chunks += len(dealer_records)

    else:

        print("\nDealers Excel not found.")

    # ==============================
    # COMPLETE
    # ==============================

    print("\n================================")
    print("INGESTION COMPLETE")
    print("================================")

    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    ingest_documents()