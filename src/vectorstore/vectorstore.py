"""
vector_store.py

This module is responsible for:

1. Converting document chunks into embeddings
2. Storing embeddings inside a FAISS vector database
3. Searching for the most relevant document chunks

This is the "Knowledge Storage" stage of a RAG pipeline.
"""

from typing import List

from langchain_core.documents import Document


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Handles document embeddings and semantic search.

    Responsibilities:
    • Create embeddings
    • Build a FAISS vector database
    • Retrieve relevant document chunks
    """

    def __init__(self):
        """
        Initialize the embedding model.

        OpenAIEmbeddings converts text into vectors
        (lists of numbers) that capture semantic meaning.
        """

        # Embedding model
        self.embedding_model = OpenAIEmbeddings()

        # FAISS vector database
        self.vectorstore = None

        # Search interface
        self.retriever = None

    ####################################################################
    # Build Vector Store
    ####################################################################

    def create_vector_store(
        self,
        documents: List[Document],
    ):
        """
        Create a FAISS vector database.

        Steps:
            1. Generate embeddings
            2. Store embeddings in FAISS
            3. Create a retriever

        Parameters
        ----------
        documents:
            List of document chunks
        """

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_model,
        )

        self.retriever = self.vectorstore.as_retriever()

    ####################################################################
    # Get Retriever
    ####################################################################

    def get_retriever(self):
        """
        Return the retriever object.

        The retriever will later be used by the RAG pipeline
        to search for relevant chunks.
        """

        if self.retriever is None:
            raise ValueError(
                "Vector store has not been created yet."
            )

        return self.retriever

    ####################################################################
    # Search Documents
    ####################################################################

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """
        Search for the most relevant document chunks.

        Parameters
        ----------
        query:
            User question

        k:
            Number of chunks to return

        Returns
        -------
        List of matching Document objects
        """

        if self.retriever is None:
            raise ValueError(
                "Vector store has not been created yet."
            )

        return self.retriever.invoke(
            query,
            search_kwargs={"k": k},
        )