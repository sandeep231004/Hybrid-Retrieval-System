# pinecone_upload_local.py
"""
OFFLINE/LOCAL VERSION - No OpenAI API required!

Uses Sentence Transformers (Hugging Face) for FREE local embeddings.
- No rate limits
- No API costs
- Runs entirely on your machine
- Slightly lower quality than OpenAI, but perfectly fine for this assignment

Install first: pip install sentence-transformers
"""
import json
import time
import logging
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
import config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("\n❌ sentence-transformers not installed")
    print("\nPlease run: pip install sentence-transformers")
    print("Then run this script again.\n")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 100  # Can be much larger since it's local!

# Using all-mpnet-base-v2 for MAXIMUM QUALITY
# This is the best performing sentence-transformer model
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dimensions, best quality

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 768  # Must match model dimensions (all-mpnet-base-v2 = 768)

# -----------------------------
# Initialize
# -----------------------------
logger.info("Loading local embedding model (first run will download ~80MB)...")
model = SentenceTransformer(EMBEDDING_MODEL)
logger.info(f"✓ Model loaded: {EMBEDDING_MODEL} ({VECTOR_DIM} dimensions)")

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create index with correct dimensions
# -----------------------------
try:
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    # Check if we need to recreate with different dimensions
    if INDEX_NAME in existing_indexes:
        logger.warning(f"Index {INDEX_NAME} exists.")
        logger.warning(f"Checking if dimensions match ({VECTOR_DIM})...")

        index_info = pc.describe_index(INDEX_NAME)
        current_dim = index_info.dimension

        if current_dim != VECTOR_DIM:
            logger.error(f"\n❌ Dimension mismatch!")
            logger.error(f"   Existing index: {current_dim} dimensions")
            logger.error(f"   This model:     {VECTOR_DIM} dimensions")
            logger.error(f"\nOptions:")
            logger.error(f"1. Delete existing index and recreate:")
            logger.error(f"   pc.delete_index('{INDEX_NAME}')")
            logger.error(f"2. Use a different index name in .env")
            logger.error(f"3. Use a different model that matches {current_dim} dims")

            user_input = input(f"\nDelete '{INDEX_NAME}' and recreate? (yes/no): ")
            if user_input.lower() in ['yes', 'y']:
                logger.info(f"Deleting index {INDEX_NAME}...")
                pc.delete_index(INDEX_NAME)
                time.sleep(5)
                logger.info("Creating new index...")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=VECTOR_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=config.PINECONE_CLOUD,
                        region=config.PINECONE_REGION
                    )
                )
                time.sleep(10)
                logger.info("✓ New index created")
            else:
                logger.error("Aborting. Please resolve dimension mismatch first.")
                exit(1)
        else:
            logger.info(f"✓ Dimensions match ({VECTOR_DIM})")
    else:
        logger.info(f"Creating new index: {INDEX_NAME} ({VECTOR_DIM} dims)")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION
            )
        )
        time.sleep(10)
        logger.info("✓ Index created")

except Exception as e:
    logger.error(f"Error managing index: {e}")
    raise

index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings_local(texts):
    """
    Generate embeddings locally using Sentence Transformers.

    Returns:
        List of embedding vectors (384 or 768 dimensions)
    """
    logger.info(f"Generating embeddings for {len(texts)} texts (local, no API)...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    return [emb.tolist() for emb in embeddings]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"LOCAL EMBEDDINGS UPLOAD (No OpenAI API)")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {EMBEDDING_MODEL}")
        logger.info(f"Dimensions: {VECTOR_DIM}")
        logger.info(f"Rate limits: NONE (running locally)")
        logger.info(f"Cost: $0.00")
        logger.info(f"{'='*70}\n")

        # Load data
        logger.info(f"Loading data from {DATA_FILE}")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            nodes = json.load(f)

        logger.info(f"Loaded {len(nodes)} nodes")

        # Prepare items
        items = []
        skipped = 0

        for node in nodes:
            semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
            if not semantic_text.strip():
                skipped += 1
                continue

            meta = {
                "id": node.get("id"),
                "type": node.get("type"),
                "name": node.get("name"),
                "city": node.get("city", node.get("region", "")),
                "tags": node.get("tags", [])
            }
            items.append((node["id"], semantic_text, meta))

        logger.info(f"Prepared {len(items)} items (skipped {skipped})")

        batches = list(chunked(items, BATCH_SIZE))
        logger.info(f"Will upload in {len(batches)} batches of {BATCH_SIZE}")

        # Estimate time
        estimate_seconds = len(batches) * 5  # ~5 seconds per batch
        logger.info(f"Estimated time: ~{estimate_seconds}s ({estimate_seconds/60:.1f} minutes)\n")

        successful_batches = 0
        failed_batches = 0

        for batch_num, batch in enumerate(tqdm(batches, desc="Uploading"), 1):
            try:
                ids = [item[0] for item in batch]
                texts = [item[1] for item in batch]
                metas = [item[2] for item in batch]

                # Generate embeddings LOCALLY (no API call!)
                embeddings = get_embeddings_local(texts)

                # Prepare vectors
                vectors = [
                    {"id": _id, "values": emb, "metadata": meta}
                    for _id, emb, meta in zip(ids, embeddings, metas)
                ]

                # Upsert to Pinecone
                index.upsert(vectors=vectors)
                successful_batches += 1

                # Small delay to avoid overwhelming Pinecone
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed batch {batch_num}: {e}")
                failed_batches += 1
                time.sleep(1)

        # Results
        logger.info(f"\n{'='*70}")
        logger.info(f"UPLOAD COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"✓ Successful: {successful_batches}/{len(batches)}")
        logger.info(f"❌ Failed:     {failed_batches}/{len(batches)}")

        # Get stats
        try:
            stats = index.describe_index_stats()
            logger.info(f"\nPinecone Stats:")
            logger.info(f"  Vectors: {stats.get('total_vector_count', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")

        logger.info(f"{'='*70}\n")

        if successful_batches == len(batches):
            logger.info("✅ ALL BATCHES UPLOADED SUCCESSFULLY!")
            logger.info("\nNext steps:")
            logger.info("1. Update hybrid_chat.py to use local embeddings (see below)")
            logger.info("2. Run: python hybrid_chat.py")

            print("\n" + "="*70)
            print("IMPORTANT: Update hybrid_chat.py")
            print("="*70)
            print("\nAdd these lines at the top of hybrid_chat.py:")
            print("""
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Replace the embed_text function with:
def embed_text(text: str):
    embedding = embedding_model.encode([text])[0]
    return tuple(embedding.tolist())
""")
            print("="*70)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
