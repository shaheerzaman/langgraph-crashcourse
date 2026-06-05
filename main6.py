from uuid import uuid4
import os
import subprocess
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

load_dotenv(find_dotenv())

llm = init_chat_model(model='openai:gpt-5.5')

KNOWLEDGE = [
    'Quantum mechanics talks mechanical forces at the sub atomic level', 
    'Traditonal mechanics talks about forces about which are not sub atomic level', 
    ''
    'Neural Networks form the basis of the Large language models.',
    'A stategraph in Langgraph defines nodes and edges that operate on a shared typed state', 
    'Checkpointers like InMemorySaver let Langgraph persist conversation state across invocations using thread id.', 
    'RAG (Retrieval-Augment Generation) combines a retriever over a knowledge base with an LLM to ground answers in source documents', 
]

vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings(model='text-embedding-3-small'))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IngtentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user just wants to chat, askfor knowledge or change code in the project.')


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent:str | None = None
    next_node: str | None = None


def classify_intent(state:State):
    structured_llm = llm.with_structured_output(IngtentClassifier)
    result = structured_llm.invoke([
        {
            'role': 'system', 'content': 'Determine / classify whether the user wants to chat"chat", retrieve knowledge "knowledge" or change code "code"'
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
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3) 
    context = '\n'.join(f'- {doc.page_content}' for doc in documents)
    messages = [{'role': 'system', 'content': f'You are a RAG agent. Answer the user only using the context below. If the answer is not in the context, say you don\'t know. \n\n Context: \n{context}'}] + state['messages']
    response = llm.invoke(messages) 
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

def prompt_llm_code(state: State):
    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
    result = subprocess.run(
        [
            '/Users/mohammad.shaheer.zaman/.vscode/extensions/openai.chatgpt-26.527.31454-darwin-arm64/bin/macos-aarch64/codex',
            'exec',
            '--sandbox', 'workspace-write',
            '--color', 'never',
            '--skip-git-repo-check',
            '-',
        ],
        input=user_prompt,
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return {'messages': [{'role': 'assistant', 'content': output}]}

def accept_coding(state: State):
    user_prompt = state['messages'][-1].content 
    decision = interrupt(f'About to run Codex with request: \n\n{user_prompt} \n\n Approve? (yes/no or type a revised request)')
    text = str(decision).strip().lower()
    
    if text in ('y', 'yes', 'approve', 'ok'):
        return {'next_node': 'coding_agent'}
    if text in ('n', 'no', 'deny', 'cancel'):
        return {'messages':[{'role': 'assistant', 'content': 'Coding request was denied by the user'}], 'next_node': 'denied'}
    
    return {'messages': [{'role': 'user', 'content': text}], 'next_node': 'accept_coding'}

def prepare_coding_request(state: State):
    messages = [
        {'role': 'system', 'content': 'Rewrite the latest use coding request into a clear instruction for Codex. Use the conversation history as context. Only output the instruction, no explanation.'}
    ] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'user', 'content': response.content}]}

graph_builder = StateGraph(State) 

graph_builder.add_node('classifier', classify_intent) 
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node') == 'denied' else state.get('next_node'), {
    'end': END, 
    'coding_agent': 'coding_agent', 
    'accept_coding':'prepare_coding_request'
})
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {
    'chat': 'chat_agent',
    'knowledge': 'rag_agent', 
    'code': 'prepare_coding_request',
})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='output_graph.png')

config = {'configurable': {'thread_id': uuid4()}}

while True:
    user_message = input('Enter message:')
    result = graph.invoke({'messages': [{'role': 'user',  'content': user_message}]}, config=config) 
    
    while '__interrupt__' in result:
        prompt = result['__interrupt__'][0].value 
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision), config=config)
    
    print(result['messages'][-1].content)
