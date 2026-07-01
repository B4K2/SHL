from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import Agent
from app.catalog import load_records
from app.config import get_settings
from app.index import build_vector_store
from app.llm import build_client
from app.schemas import ChatRequest, ChatResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    records = load_records(settings.catalog_path)
    store = build_vector_store(settings, records)
    client = build_client(settings)
    app.state.agent = Agent(client=client, settings=settings, store=store)
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="SHL", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    agent: Agent = app.state.agent
    return await agent.respond(request.messages)
