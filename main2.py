from uuid import uuid4
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model 
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

llm = init_chat_model(model='openai:gpt-5.5')

class IngtentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user just wants to chat, ask for knowledge or change code in the project.')


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent:str|None


def classify_intent(state:State):
    structured_llm = llm.with_structured_output(IngtentClassifier)
    result = structured_llm.invoke([
        {
            'role': 'system', 'content': 'Determine / classify whether the user wants to chat "chat", retrieve knowledge "knowledge" or change code "code"'
        }, 
        {
            'role': 'user', 'content': state['messages'][-1].content
        }
    ])
    return {'message_intent': result.message_intent}

def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'you are a talkative chatbot for fun. be nice'}] + state['messages']
    response = llm.invoke(messages) 
    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_rag(state: State):
    messages = [{'role': 'system', 'content': 'No matter what the user says always and always say that. "I am the RAG agent." do not give any other answer'}] + state['messages']
    response = llm.invoke(messages) 
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

def prompt_llm_code(state: State):
    messages = [{'role': 'system', 'content': 'No matter what the user says always and always say "I am the Coding Agent". do not give any other answer'}] + state['messages']
    response = llm.invoke(messages) 
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

graph_builder = StateGraph(State) 

graph_builder.add_node('classifier', classify_intent) 
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {
    'chat': 'chat_agent',
    'knowledge': 'rag_agent', 
    'code': 'coding_agent'
})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
graph.get_graph().draw_mermaid_png(output_file_path='basic-nodes.png')

config = {'configurable': {'thread_id': uuid4()}}

while True:
    user_message = input('Enter message:')
    result = graph.invoke({'messages': [{'role': 'user',  'content': user_message}]}, config=config) 
    
    print(result['messages'][-1].content)