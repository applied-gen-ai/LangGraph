import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

mixtral = 'mixtral-8x7b-32768'
llama = 'llama-3.3-70b-versatile'

llm = ChatGroq(temperature=0, model_name=llama)

# --------- Base conversation chain ---------
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that assists humans on any topic. Answer in 30 words",
        ),
        ("human", "{user_input}"),
    ]
)

chain = template | llm

# --------- Summarization chain ---------
templat_summarize = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant that summarizes the conversation 
            without losing key topics. Make sure every topic is included. 
            Just summarize the discussion around that topic in compact form.""",
        ),
        ("human", "{user_input}"),
    ]
)

chain_summarize = templat_summarize | llm


from typing import List, Any
def append_messages_reducer(current_messages: List[Any], new_messages: List[Any]) -> List[Any]:
    """Appends new messages to the existing list of messages."""

    if current_messages is None or len(current_messages)>5:
        return new_messages
    return current_messages + new_messages

# --------- State ---------

class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], append_messages_reducer]
    current_msg: str

# --------- Nodes ---------
def node1(state: CustomMessagesState):

    print(state['messages'])
    print("=====================================================")
    x = chain.invoke({"user_input": state["messages"]})
    msg = AIMessage(content=x.content)
    return {"messages": [msg], "current_msg": msg}  # appended due to reducer

def summarize_chain(state: CustomMessagesState):
    # last message in conversation
    rsp = state["messages"][-1]

    # create summary of full conversation
    summary = chain_summarize.invoke({"user_input": state["messages"]})

    # 🚨 overwrite memory with summary + last msg

    return {
        "messages": [
            AIMessage(content=f"Summary so far: {summary.content}"), rsp
        ],
        "current_msg": rsp
    }

# --------- Conditional edge ---------
def check_summary(state: CustomMessagesState):
    if len(state["messages"]) > 4:
        
        return "summarize_convo"
    else:
        return "normal_memory"

# --------- Build graph ---------
builder = StateGraph(CustomMessagesState)

builder.add_node("llm_response", node1)
builder.add_node("summarize_chain", summarize_chain)

builder.add_edge(START, "llm_response")

builder.add_conditional_edges(
    "llm_response",
    check_summary,
    {
        "summarize_convo": "summarize_chain",
        "normal_memory": END,
    },
)

builder.add_edge("summarize_chain", END)

# --------- Memory ---------
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

#graph.add_reducer("messages", append_messages_reducer)

# --------- Run loop ---------
thread_id = "user_123"   # each conversation has its own thread

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    res = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"AI: {res['current_msg']}")
