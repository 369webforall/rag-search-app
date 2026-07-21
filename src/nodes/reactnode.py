"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent


# Wikipedia imports
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun



class RAGNodes:
    """
    LangGraph nodes for RAG workflow.

    Flow:

    User Question
          |
          v
    ReAct Agent
          |
          +---- Retriever Tool
          |
          +---- Wikipedia Tool
          |
          v
       Final Answer
    """



    def __init__(
        self,
        retriever,
        llm
    ):

        self.retriever = retriever
        self.llm = llm

        # lazy loading
        self._agent = None



    # ==================================================
    # Retriever Node
    # ==================================================

    def retrieve_docs(
        self,
        state: RAGState
    ) -> RAGState:

        docs = self.retriever.invoke(
            state.question
        )


        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )



    # ==================================================
    # Build Agent Tools
    # ==================================================

    def _build_tools(
        self
    ) -> List[Tool]:


        # ----------------------------------------------
        # Internal PDF Retriever Tool
        # ----------------------------------------------

        def retriever_tool_fn(
            query: str
        ) -> str:
            """
            Search uploaded documents.
            """

            try:

                docs: List[Document] = (
                    self.retriever.invoke(query)
                )


                if not docs:
                    return (
                        "No relevant documents found."
                    )


                results = []


                for index, doc in enumerate(
                    docs[:8],
                    start=1
                ):

                    metadata = (
                        doc.metadata
                        if hasattr(doc, "metadata")
                        else {}
                    )


                    source = (
                        metadata.get("source")
                        or metadata.get("title")
                        or f"document_{index}"
                    )


                    results.append(
                        f"""
Document {index}
Source: {source}

{doc.page_content}
"""
                    )


                return "\n\n".join(results)


            except Exception as e:

                return (
                    f"Retriever error: {str(e)}"
                )



        retriever_tool = Tool(
            name="retriever",
            description="""
Search the user's uploaded PDF documents.

ALWAYS use this tool first for:
- PDF questions
- uploaded files
- course notes
- documents
- internal knowledge

""",
            func=retriever_tool_fn
        )



        # ----------------------------------------------
        # Wikipedia Tool
        # ----------------------------------------------


        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=3,
                lang="en"
            )
        )


        def wikipedia_tool_fn(
            query: str
        ) -> str:
            """
            Safe Wikipedia search.
            """

            try:

                # prevent very large requests
                query = query[:300]


                result = wiki.run(
                    query
                )


                if not result:

                    return (
                        "No Wikipedia information found."
                    )


                return result


            except Exception:

                return (
                    "Wikipedia is currently unavailable. "
                    "Please use uploaded documents."
                )



        wikipedia_tool = Tool(
            name="wikipedia",
            description="""
Search Wikipedia.

Use ONLY for general knowledge questions.

Do NOT use this for uploaded PDFs.
""",
            func=wikipedia_tool_fn
        )



        return [
            retriever_tool,
            wikipedia_tool
        ]



    # ==================================================
    # Build ReAct Agent
    # ==================================================

    def _build_agent(
        self
    ):


        tools = self._build_tools()



        system_prompt = """
You are a helpful RAG assistant.

Rules:

1. For PDF or uploaded document questions:
   ALWAYS call retriever first.

2. Never use Wikipedia for PDF questions.

3. Use Wikipedia only for general knowledge.

4. If retriever provides information,
   answer using that information.

5. Give only the final answer.

6. Do not mention tools.
"""



        self._agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )



    # ==================================================
    # Generate Answer Node
    # ==================================================

    def generate_answer(
        self,
        state: RAGState
    ) -> RAGState:


        if self._agent is None:

            self._build_agent()



        result = self._agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=state.question
                    )
                ]
            }
        )



        messages = result.get(
            "messages",
            []
        )


        answer: Optional[str] = None



        if messages:

            last_message = messages[-1]


            answer = getattr(
                last_message,
                "content",
                None
            )



        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=(
                answer
                if answer
                else "No answer generated."
            )
        )