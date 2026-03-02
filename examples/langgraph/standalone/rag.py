"""Standalone Agentic RAG graph used by LangGraph examples."""

from __future__ import annotations

from typing import Literal, cast

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field


def create_rag_graph():
    """Create a compiled RAG graph with retrieval, grading, rewrite, and answer nodes."""
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Retrieve passages from Lilian Weng blog posts about AI topics."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    response_model = ChatOpenAI(model="gpt-4o", temperature=0)
    grader_model = ChatOpenAI(model="gpt-4o", temperature=0)

    def generate_query_or_respond(state: MessagesState):
        system_prompt = (
            "You answer AI-topic questions. Use retrieve_blog_posts when factual "
            "blog context is needed; otherwise answer directly."
        )

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = response_model.bind_tools([retrieve_blog_posts]).invoke(messages)
        return {"messages": [response]}

    class GradeDocuments(BaseModel):
        """Binary relevance score."""

        binary_score: str = Field(description="'yes' if relevant, otherwise 'no'")

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content

        grade_prompt = (
            "Grade document relevance for the user question. Return 'yes' when "
            "the retrieved context is semantically relevant, otherwise 'no'.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )

        response = cast(
            GradeDocuments,
            grader_model.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": grade_prompt}]
            ),
        )
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(state: MessagesState):
        question = state["messages"][0].content
        rewrite_prompt = (
            "Rewrite this question to improve retrieval while keeping intent.\n\n"
            f"Question: {question}"
        )
        response = response_model.invoke([{"role": "user", "content": rewrite_prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def generate_answer(state: MessagesState):
        messages = state["messages"]

        context = None
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name == "retrieve_blog_posts":
                context = msg.content
                break

        if not context:
            return {"messages": [response_model.invoke(messages)]}

        question = messages[0].content
        answer_prompt = (
            "Answer the question using the retrieved context. Keep the answer to "
            "3 sentences max.\n\n"
            f"Context: {context}\n\nQuestion: {question}"
        )
        response = response_model.invoke([{"role": "user", "content": answer_prompt}])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retrieve_blog_posts]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer=InMemorySaver())
