import json
import re
import uuid
import datasets
from tqdm import tqdm # progress bar
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- 1. CONFIG & HELPER FUNCTIONS ---
def chunk_context(context_text, max_len=1000):
    """
    Splitting document into chunks
    """
    if not context_text:
        return []
    
    pattern = r'(?<=[a-zà-ỹ]{3}[.!?])\s+(?=[A-ZÀ-Ỵ""])'
    sentences = re.split(pattern, context_text)
    
    chunks = []
    chunk = ""
    total_len = 0
    
    for sentence in sentences:
        if total_len + len(sentence) >= max_len and chunk:
            chunks.append(chunk.strip())
            chunk = sentence + " "
            total_len = len(sentence)
        else:
            chunk += (sentence + " ")
            total_len += len(sentence)
    
    if chunk.strip():
        chunks.append(chunk.strip())
    
    return chunks

# MAIN PIPELINE
def main():
    # A. Configuration
    COLLECTION_NAME = "vifactcheck_chunks"
    VECTOR_SIZE = 1024      # BGE-M3 vector size
    BATCH_SIZE = 64         # Number of points to upload at a time to avoid RAM overload
    LIMIT_ARTICLES = 200    # Set a number for testing, set None to run all

    # B. Initialize Client & Model
    client = QdrantClient(path="vectordb")
    model = SentenceTransformer('BAAI/bge-m3')
    
    # Create new collection (reset if exists)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"-> Created collection '{COLLECTION_NAME}'")

    # C. Load Data
    print("-> Loading Dataset & Summaries...")
    # 1. Load Dataset
    ds = datasets.load_from_disk('../data/vifactcheck-normalized')
    
    # 2. Load Summaries
    with open("../data/summaries_train_result.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)
    
    # Filter dataset if needed for quick testing
    if LIMIT_ARTICLES:
        ds = ds.select(range(LIMIT_ARTICLES))
        print(f"-> Running test mode with {LIMIT_ARTICLES} articles.")

    print(f"-> Starting to process {len(ds)} articles... and {len(summaries)} summaries.")

    # D. Processing Loop
    points_buffer = []
    
    # Use tqdm to show progress bar
    for idx, item in tqdm(enumerate(ds), total=len(ds), desc="Processing"):
        try:
            # Get basic information
            context = item.get('Context', '')
            # If context is empty or None, skip
            if not context: 
                continue

            # Get summary from json file (use str(idx) as key)
            summarize_text = summaries.get(str(idx), "")
            
            # 1. CHUNKING
            chunks = chunk_context(context)
            
            # If no chunk can be split (text is too short or error), use the original context as 1 chunk
            if not chunks:
                chunks = [context]

            # 2. EMBEDDING
            # Embed list of chunks
            embeddings = model.encode(chunks, convert_to_tensor=False)

            # 3. PREPARE POINTS
            for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                
                # Payload contains all information to query back later
                payload = {
                    "article_id": idx,            # Original article ID to group
                    "chunk_index": i,             # Chunk index
                    "text": chunk_text,           # Main content of the chunk (for RAG)
                    "statement": item.get('Statement', ''),
                    "summarize": summarize_text,  # Summary load từ json
                    "evidence": item.get('Evidence', ''),
                    "label": item.get('labels', ''),
                    "topic": item.get('Merged Topic', ''),
                    "url": item.get('Url', '')
                }

                # Create Point for Qdrant
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Create random UUID for each chunk
                    vector=vector.tolist(),
                    payload=payload
                )
                points_buffer.append(point)

            # 4. BATCH UPSERT (Write to disk when enough)
            if len(points_buffer) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_buffer
                )
                points_buffer = [] # Reset buffer

        except Exception as e:
            print(f"\n[Error] Error at article index {idx}: {e}")
            continue

    # Upsert remaining points in the last buffer
    if points_buffer:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_buffer
        )

    # E. Finish
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n>> FINISH! Total {info.points_count} vectors (chunks) have been saved to folder 'vectordb'.")
    client.close()

if __name__ == "__main__":
    main()