from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
import tqdm
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",temperature=0.3)

vector_store = Chroma(
    persist_directory="study_materials",
    embedding_function=embeddings
)

class EmbeddingBM25RerankerRetriever:
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        reranker: BaseDocumentCompressor,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def invoke(self, query: str):
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        combined_docs = vector_docs + [
            doc for doc in bm25_docs if doc not in vector_docs
        ]

        reranked_docs = self.reranker.compress_documents(combined_docs, query)
        return reranked_docs
    
docs = vector_store.get()['documents']


docs = [Document(page_content=text) for text in docs]

bm25_retriever = BM25Retriever.from_documents(docs, language="english")

reranker = CohereRerank(top_n=10, model="rerank-english-v2.0")

contextual_embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

contextual_bm25_retriever = bm25_retriever

contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    vector_retriever=contextual_embedding_retriever,
    bm25_retriever=contextual_bm25_retriever,
    reranker=reranker,
)
