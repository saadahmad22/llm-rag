from langchain.chains import RetrievalQA
from src.retrievers.vector_store import VectorStore
from src.models.llama import LlamaModel
from langchain_core.documents.base import Document
from langchain_core.chat_history import InMemoryChatMessageHistory

class RAGChain:
    def __init__(self, vector_store: VectorStore, model: LlamaModel):
        self.store = vector_store
        self.llm = model

    def run(self, query: str, history: InMemoryChatMessageHistory) -> str:
        # Retrieve relevant documents
        documents: list[Document] = self.store.retrieve(query)
        
        # Combine documents into a single context
        context = " ".join((doc.page_content for doc in documents))
        
        # Generate response using the Llama model
        response = self.llm.generate_response(f"Question: {query}\nContext: {context}", history)
        
        return response
    
    def generate(self, query: str, history: InMemoryChatMessageHistory) -> str:
        return self.llm.generate_response(f"Question: {query}", history)
    
    def embed(self, query: str) -> list[float]:
        return self.store.embeddings.embed_documents([query])[0]
    
    def get_user_history(self, user_id: str) -> InMemoryChatMessageHistory:
        return self.llm.get_user_history(user_id)