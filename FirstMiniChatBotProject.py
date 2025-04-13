import os
import tiktoken
import uuid
from typing import Dict
from openai import BaseModel
from fastapi import FastAPI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from starlette.responses import JSONResponse
from secret_key import openapi_key

os.environ["OPENAI_API_KEY"] = openapi_key

memory_store: Dict[str, ConversationBufferMemory] = {}
session_titles: Dict[str, str] = {}

class CompletionRequest(BaseModel):
    session_id: str
    prompt: str
    title: str = None
    max_tokens: int = 1000

def count_tokens(text: str, model_name="gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

class TokenGuard:
    def __init__(self, total_limit=4096, min_response_buffer=500):

        self.total_limit = total_limit
        self.min_response_buffer = min_response_buffer

    def get_safe_max_tokens(self, prompt_text: str, history_text: str, user_input: str, model_name: str) -> int:
        full_prompt = prompt_text + history_text + user_input
        used_tokens = count_tokens(full_prompt, model_name)
        remaining_tokens = self.total_limit - used_tokens

        if remaining_tokens <= self.min_response_buffer:
            raise ValueError("Prompt too long, token limit approaching.")

        return remaining_tokens - self.min_response_buffer

token_guard = TokenGuard()
app = FastAPI()

class SessionManager:
    def __init__(self):
        self.current_session = str(uuid.uuid4())

    def get_current_session(self):
        if self.current_session not in memory_store:
            memory_store[self.current_session] = ConversationBufferMemory(
                memory_key="chat_history",
                chat_memory=ChatMessageHistory(),
                return_messages=True
            )
        return memory_store[self.current_session]

class ModelSelector:
    def __init__(self):
        self.code_keywords = [
            "code", "coding", "python", "bug", "algorithm", "function", "class",
            "error", "exception", "compile", "debug", "framework", "library",
            "help me write", "how to write", "write a function", "write a class"
        ]
        self.simple_chat_keywords = [
            "hi", "hello", "how are you", "what’s up", "let’s talk", "chat", "conversation"
        ]
        self.max_question_length_for_simple_model = 15

    def is_coding_question(self, question: str) -> bool:
        return any(kw in question.lower() for kw in self.code_keywords)

    def is_simple_chat(self, question: str) -> bool:
        return any(kw in question.lower() for kw in self.simple_chat_keywords)

    def select_model(self, question: str) -> str:
        question = question.lower()
        word_count = len(question.strip().split())

        if self.is_coding_question(question):
            return "gpt-4"

        elif word_count > 20:
            return "gpt-4"

        elif self.is_simple_chat(question) and word_count <= self.max_question_length_for_simple_model:
            return "gpt-3.5-turbo-1106"

        return "gpt-3.5-turbo"

@app.post("/chat/")
def chat_management(request: CompletionRequest):

    """
    ConversationBufferMemory doğrudan belleği yönetiyor ve otomatik olarak yanıtları belleğe kaydediyor.
    """

    if request.session_id not in memory_store:
        memory_store[request.session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=ChatMessageHistory(),
            return_messages=True
        )
        if request.title:
            session_titles[request.session_id] = request.title

    memory_update = SessionManager()
    memory = memory_update.get_current_session()
    chat_history = memory.load_memory_variables({})["chat_history"]

    system_prompt = "You are an angry chatbot having a conversation with me."

    history_text = "\n".join([str(msg) for msg in chat_history])
    user_input = request.prompt

    model_selector = ModelSelector()
    selected_model = model_selector.select_model(user_input)

    try:
        safe_max = token_guard.get_safe_max_tokens(system_prompt, history_text, user_input, model_name=selected_model)
    except ValueError as ve:
        return {"error": str(ve)}

    llm = ChatOpenAI(model_name=selected_model, temperature=0.6, max_tokens=safe_max)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    conversation = prompt | llm

    response = conversation.invoke({"question": request.prompt, "chat_history": chat_history})  # AIMessage objesi.

    memory.save_context({"question": request.prompt}, {"response": str(response)})

    return {"response": response.content}

@app.get("/chat/{session_id}")
def get_chat_history(session_id: str):
    if session_id not in memory_store:
        return JSONResponse(status_code=404, content={"error": "Session not found."})

    memory = memory_store[session_id]
    chat_history = memory.load_memory_variables({})["chat_history"]

    history_list = []
    for msg in chat_history:
        history_list.append({
            "type": msg.type,
            "content": msg.content,
        })

    return {"session_id": session_id, "chat_history": history_list}

@app.get("/sessions")
def list_sessions():
    sessions = []
    for session_id, title in session_titles.items():
        print(session_id)
        sessions.append({
            "session_id": session_id,
            "title": title,
        })
    return {"sessions": sessions}