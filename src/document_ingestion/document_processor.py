"""
document_processor.py

This module is responsible for:

1. Loading documents
    - Website (URL)
    - PDF file
    - Folder of PDF files
    - Text file

2. Splitting large documents into smaller chunks

These chunks are later converted into embeddings
and stored inside the vector database.
"""

from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document


from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles loading and splitting documents.

    Think of this class as the "Document Preparation" stage
    of a RAG pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Create a text splitter.

        Parameters
        ----------
        chunk_size:
            Maximum number of characters inside one chunk.

        chunk_overlap:
            Number of characters repeated between chunks.
            This helps preserve context.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    ###################################################################
    # Loading Documents
    ###################################################################

    def load_from_url(self, url: str) -> List[Document]:
        """
        Load a webpage.

        Example:
            https://python.langchain.com/docs/

        Returns:
            List of LangChain Document objects
        """

        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf(self, pdf_path: Union[str, Path]) -> List[Document]:
        """
        Load a single PDF.

        Example:
            data/python.pdf
        """

        loader = PyPDFLoader(str(pdf_path))
        return loader.load()

    def load_from_pdf_directory(
        self,
        folder: Union[str, Path],
    ) -> List[Document]:
        """
        Load every PDF inside a folder.

        Example:

            data/
                python.pdf
                ai.pdf
                machine_learning.pdf
        """

        loader = PyPDFDirectoryLoader(str(folder))
        return loader.load()

    def load_from_text(
        self,
        text_file: Union[str, Path],
    ) -> List[Document]:
        """
        Load a text (.txt) file.
        """

        loader = TextLoader(
            str(text_file),
            encoding="utf-8",
        )
        return loader.load()


 ###################################################################
    # Automatically Detect File Type
###################################################################

    def load_documents(
        self,
        sources: List[Union[str, Path]],
    ) -> List[Document]:

        documents: List[Document] = []

        for source in sources:

            # URL
            if isinstance(source, str) and source.startswith(
                ("http://", "https://")
            ):
                documents.extend(
                    self.load_from_url(source)
                )
                continue

            source = Path(source)

            # PDF folder
            if source.is_dir():
                documents.extend(
                    self.load_from_pdf_directory(source)
                )

            # PDF file
            elif source.suffix.lower() == ".pdf":
                documents.extend(
                    self.load_from_pdf(source)
                )

            # Text file
            elif source.suffix.lower() == ".txt":
                documents.extend(
                    self.load_from_text(source)
                )

            else:
                raise ValueError(
                    f"Unsupported file type: {source}"
                )

        return documents

    ###################################################################
    # Split Documents
    ###################################################################

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into smaller chunks.

        Example:

        20-page PDF
                ↓

        Chunk 1
        Chunk 2
        Chunk 3
        Chunk 4
        """

        return self.splitter.split_documents(documents)

    ###################################################################
    # Complete Pipeline
    ###################################################################

    def process_documents(
        self,
        sources: List[Union[str, Path]],
    ) -> List[Document]:
        """
        Complete document processing pipeline.

        Step 1:
            Load documents

        Step 2:
            Split them into chunks

        Step 3:
            Return the chunks
        """

        documents = self.load_documents(sources)

        chunks = self.split_documents(documents)

        return chunks