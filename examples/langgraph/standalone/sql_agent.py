"""Standalone SQL agent graph for LangGraph examples."""

from __future__ import annotations

from typing import Annotated, Literal
import logging
import os
import sqlite3
import urllib.request

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class MessagesState(TypedDict):
    """State container for the SQL agent."""

    messages: Annotated[list, add_messages]


def create_sql_agent(db_path: str = "Chinook.db"):
    """Create a SQL agent that can answer natural language database questions."""
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return END
        return "tools"

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=InMemorySaver())


def download_chinook_db() -> str:
    """Download the Chinook sample database, or create a minimal fallback DB."""
    db_path = "Chinook.db"
    if os.path.exists(db_path):
        logger.info("Database already exists at %s", db_path)
        return db_path

    url = (
        "https://github.com/lerocha/chinook-database/raw/master/"
        "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    )
    logger.info("Downloading Chinook sample database...")

    try:
        urllib.request.urlretrieve(url, db_path)
        logger.info("Downloaded database to %s", db_path)
        return db_path
    except Exception as error:
        logger.warning("Download failed (%s); creating fallback database", error)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                title TEXT,
                salary INTEGER
            )
            """
        )
        cursor.executemany(
            "INSERT INTO employees (name, title, salary) VALUES (?, ?, ?)",
            [
                ("Alice Johnson", "Software Engineer", 95000),
                ("Bob Smith", "Product Manager", 110000),
                ("Carol Davis", "Data Scientist", 105000),
                ("David Brown", "Designer", 85000),
            ],
        )
        conn.commit()
        conn.close()
        logger.info("Created fallback database at %s", db_path)
        return db_path


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test() -> None:
        db_path = download_chinook_db()
        sql_agent = create_sql_agent(db_path)

        questions = [
            "List all the tables in the database",
            "What are the column names in the Employee table?",
            "How many employees are there?",
        ]

        logger.info("Testing SQL Agent")
        for question in questions:
            logger.info("Question: %s", question)
            result = await sql_agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                {"configurable": {"thread_id": "test-session"}},
            )
            final_message = result["messages"][-1]
            if isinstance(final_message, AIMessage):
                logger.info("Answer: %s", final_message.content)
            else:
                logger.info("Response: %s", final_message)

    asyncio.run(test())
