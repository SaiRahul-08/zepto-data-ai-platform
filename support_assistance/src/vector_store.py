import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from .ingest import load_documents, chunk_documents

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def build_index():
    chunks = chunk_documents(load_documents())
    model = get_model()
    collection = get_collection()
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection

def retrieve(query, top_k=3):
    collection = get_collection()
    if collection.count() == 0:
        build_index()
    query_embedding = get_model().encode([query], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for i, doc in enumerate(result["documents"][0]):
        items.append({
            "id": result["ids"][0][i],
            "text": doc,
            "source": result["metadatas"][0][i]["source"],
            "distance": result["distances"][0][i],
        })
    return items

if __name__ == "__main__":
    c = build_index()
    print(f"ChromaDB collection '{COLLECTION_NAME}' contains {c.count()} chunks.")
