import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.retrievers.vector_store import VectorStore
from src.models.llama import LlamaModel
from src.chains.rag_chain import RAGChain

vector_store = VectorStore()
llm = LlamaModel(model_path="/Users/saad/Documents/Work/RaWee/poc/python/model/llama-3.2-3b-instruct-q8_0.gguf")
rag_chain = RAGChain(vector_store, llm)