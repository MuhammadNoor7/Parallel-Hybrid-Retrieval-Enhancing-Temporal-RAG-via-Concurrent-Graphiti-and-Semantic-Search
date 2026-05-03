import json
import uuid
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import your custom embedder
from local_embedder import SentenceTransformerEmbedder

# --- Configuration ---
JSON_FILE_PATH = "data/longmemeval_s.json"
COLLECTION_NAME = "longmemeval_collection"
BATCH_SIZE = 32 # Adjust based on your VRAM (32-64 is usually safe for an 8GB GPU)

async def process_and_upload_batch(client, embedder, texts, metadata):
    """Helper function to embed and upload a chunk of data."""
    print(f"Embedding and uploading a batch of {len(texts)} sessions...")
    
    # 1. Use your embedder's create_batch method
    embeddings = await embedder.create_batch(texts)
    
    # 2. Structure the points for Qdrant
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=meta)
        for vector, meta in zip(embeddings, metadata)
    ]
    
    # 3. Upload to Qdrant asynchronously
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

async def main():
    # 1. Initialize your local embedder
    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-base-en-v1.5")
    vector_size = embedder.embedding_dim

    # 2. Initialize Async Qdrant Client (pointing to your Docker instance)
    client = AsyncQdrantClient(url="http://localhost:7333")

    # 3. Setup Qdrant Collection
    print(f"Setting up collection: {COLLECTION_NAME}...")
    if await client.collection_exists(collection_name=COLLECTION_NAME):
        await client.delete_collection(collection_name=COLLECTION_NAME)

    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # 4. Load JSON Data
    print("Reading dataset...")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts_batch = []
    metadata_batch = []

    # 5. Iterate through the data and batch it
    for row in data:
        question_id = row.get("question_id")
        haystack_session_ids = row.get("haystack_session_ids", [])
        haystack_sessions = row.get("haystack_sessions", [])
        
        for idx, session in enumerate(haystack_sessions):
            session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else f"unknown_{idx}"
            
            # Condense chat into a single string
            session_text = "\n".join([
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" 
                for msg in session
            ])
            
            texts_batch.append(session_text)
            metadata_batch.append({
                "question_id": question_id,
                "session_id": session_id,
                "text": session_text
            })
            
            # If our batch reaches the limit, process it
            if len(texts_batch) >= BATCH_SIZE:
                await process_and_upload_batch(client, embedder, texts_batch, metadata_batch)
                texts_batch.clear()
                metadata_batch.clear()

    # 6. Process any leftover items in the final batch
    if texts_batch:
        await process_and_upload_batch(client, embedder, texts_batch, metadata_batch)

    print("✅ All data successfully embedded and ingested into Qdrant!")

if __name__ == "__main__":
    # Run the async event loop
    asyncio.run(main())