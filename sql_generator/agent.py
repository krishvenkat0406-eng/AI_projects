import duckdb
import pandas as pd
import plotly.express as px
from typing import TypedDict, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    sql_query: str
    error: Optional[str]
    iteration: int
    data: Optional[pd.DataFrame]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
db_path = "sales_data.duckdb"

DB_SCHEMA = """
Tables in DuckDB:
1. customers (customer_id INT, customer_name VARCHAR, region VARCHAR)
2. orders (order_id INT, customer_id INT, order_date TIMESTAMP, total_amount DOUBLE)

Note: Q3 corresponds to order_date between '2024-07-01' and '2024-09-30'.
"""

def generate_sql(state: AgentState) -> AgentState:
    question = state["question"]
    error = state.get("error")
    iteration = state.get("iteration", 0)

    system_prompt = f"You are an expert DuckDB SQL developer. Return ONLY valid executable SQL without markdown formatting.\nSchema:\n{DB_SCHEMA}"
    user_prompt = f"Fix this query:\nQuestion: {question}\nQuery: {state['sql_query']}\nError: {error}" if error else f"Convert to SQL: {question}"

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return {**state, "sql_query": sql, "iteration": iteration + 1}

def execute_sql(state: AgentState) -> AgentState:
    try:
        conn = duckdb.connect(db_path)
        df = conn.execute(state["sql_query"]).df()
        conn.close()
        return {**state, "data": df, "error": None}
    except Exception as e:
        return {**state, "error": str(e), "data": None}

def should_continue(state: AgentState) -> str:
    return "retry" if state["error"] and state["iteration"] < 4 else "finish"

workflow = StateGraph(AgentState)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.set_entry_point("generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", should_continue, {"retry": "generate_sql", "finish": END})
sql_agent_app = workflow.compile()

def run_sql_agent(question: str) -> Dict[str, Any]:
    result = sql_agent_app.invoke({"question": question, "sql_query": "", "error": None, "iteration": 0, "data": None})
    
    chart_html = ""
    table_html = ""
    if result["data"] is not None and not result["data"].empty:
        df = result["data"]
        table_html = df.to_html(classes="table table-striped table-hover", index=False)
        if len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualized Query Results")
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return {
        "sql_query": result["sql_query"],
        "error": result["error"] or "Query executed successfully.",
        "iterations": result["iteration"],
        "table_html": table_html,
        "chart_html": chart_html
    }
