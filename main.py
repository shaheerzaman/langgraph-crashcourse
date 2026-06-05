from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model 
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv(find_dotenv(raise_error_if_not_found=True))

llm = init_chat_model(model='openai:gpt-5.5') 

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


graph_builder = StateGraph(MessagesState) 

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm') 
graph_builder.add_edge('prompt_llm', END) 

graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='basic.png')

user_message = input('Enter message:')
print(graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}))

    