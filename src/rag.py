"""Retrieval over course notes using sentence embeddings and a Chroma vector store."""

EMBED_MODEL = "all-MiniLM-L6-v2"


class NotesIndex:
    """Embeds note passages once, then finds the most relevant ones for a topic."""

    def __init__(self, passages, embed_model=EMBED_MODEL):
        import chromadb
        from chromadb.utils import embedding_functions

        self.passages = passages
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        client = chromadb.Client()  # in-memory store; rebuilt each session in seconds
        self.collection = client.get_or_create_collection("notes", embedding_function=embed_fn)
        if self.collection.count() == 0:
            self.collection.add(
                ids=[str(i) for i in range(len(passages))],
                documents=[f"{p['section']}: {p['text']}" for p in passages],
                metadatas=[{"week": p["week"], "section": p["section"]} for p in passages],
            )

    def search(self, topic, k=3):
        """Return the k passages most relevant to the topic, best match first."""
        k = min(k, len(self.passages))
        result = self.collection.query(query_texts=[topic], n_results=k)
        return [self.passages[int(i)] for i in result["ids"][0]]