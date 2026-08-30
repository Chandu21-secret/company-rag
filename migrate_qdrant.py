import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


load_dotenv()


COLLECTION = "company_knowledge"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL .env mein nahi mila")

if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY .env mein nahi mila")


print("=" * 50)
print("LOCAL QDRANT -> QDRANT CLOUD MIGRATION")
print("=" * 50)


# --------------------------------------------------
# LOCAL QDRANT
# --------------------------------------------------

print("\nConnecting to local Qdrant...")

local_client = QdrantClient(
    path="./qdrant_storage"
)


# --------------------------------------------------
# CLOUD QDRANT
# --------------------------------------------------

print("Connecting to Qdrant Cloud...")

cloud_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)


# --------------------------------------------------
# LOCAL COLLECTION
# --------------------------------------------------

print("\nChecking local collection...")

local_info = local_client.get_collection(
    COLLECTION
)

local_count = local_info.points_count or 0

print("Local collection:", COLLECTION)
print("Local points:", local_count)


# --------------------------------------------------
# DELETE CLOUD COLLECTION IF EXISTS
# --------------------------------------------------

print("\nPreparing cloud collection...")

try:

    if cloud_client.collection_exists(COLLECTION):

        cloud_client.delete_collection(
            COLLECTION
        )

        print("Existing cloud collection deleted.")

except Exception as e:

    print("Cloud collection check:", e)


# --------------------------------------------------
# CREATE CLOUD COLLECTION
# --------------------------------------------------

print("Creating cloud collection...")

cloud_client.create_collection(

    collection_name=COLLECTION,

    vectors_config=local_info.config.params.vectors
)

print("Cloud collection created.")


# --------------------------------------------------
# COPY POINTS
# --------------------------------------------------

print("\nCopying points...")

offset = None

total = 0

BATCH_SIZE = 100


while True:

    records, next_offset = local_client.scroll(

        collection_name=COLLECTION,

        limit=BATCH_SIZE,

        offset=offset,

        with_payload=True,

        with_vectors=True
    )


    if not records:

        break


    points = []


    for record in records:

        point = PointStruct(

            id=record.id,

            vector=record.vector,

            payload=record.payload or {}
        )

        points.append(point)


    cloud_client.upsert(

        collection_name=COLLECTION,

        points=points,

        wait=True
    )


    total += len(points)

    print(
        f"Transferred: {total}/{local_count}"
    )


    offset = next_offset


    if offset is None:

        break


# --------------------------------------------------
# VERIFY
# --------------------------------------------------

print("\nVerifying cloud collection...")

cloud_info = cloud_client.get_collection(
    COLLECTION
)


cloud_count = cloud_info.points_count or 0


print("\nLocal points :", local_count)
print("Cloud points :", cloud_count)


if cloud_count == local_count:

    print("\n" + "=" * 50)
    print("MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 50)

else:

    print("\n" + "=" * 50)
    print("WARNING: POINT COUNT DOES NOT MATCH")
    print("=" * 50)


# --------------------------------------------------
# CLOSE CLIENTS
# --------------------------------------------------

local_client.close()
cloud_client.close()