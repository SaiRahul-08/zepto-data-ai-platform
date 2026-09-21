from support_assistance.src.vector_store import build_index


if __name__ == "__main__":
    collection = build_index()
    print(f"Indexed {collection.count()} chunks in ChromaDB.")