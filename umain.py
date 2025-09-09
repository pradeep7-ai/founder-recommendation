# umain.py
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from supabase import create_client, Client
from dotenv import load_dotenv

# ---- Import embedding functions from uembeddings.py ---- #
from uembeddings import generate_embedding_from_userobj, store_embedding

# ---------------- Environment ---------------- #
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_URL or SUPABASE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- FastAPI Models ---------------- #
class UserCreate(BaseModel):
    name: str
    role: str
    industry: str
    stage: str
    years_experience: int
    funding_amount: float
    skills: str
    vision_statement: str
    education_background: str
    location: str

    @field_validator("funding_amount")
    def funding_amount_positive(cls, v):
        if v < 0:
            raise ValueError("Funding amount must be positive")
        return v

    @field_validator("years_experience")
    def years_experience_positive(cls, v):
        if v < 0:
            raise ValueError("Years of experience must be positive")
        return v


class UserResponse(BaseModel):
    id: int
    name: str
    role: str
    industry: str
    stage: str
    years_experience: int
    funding_amount: float
    skills: str
    vision_statement: str
    education_background: str
    location: str
    created_at: datetime


class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5


class RecommendationResponse(BaseModel):
    user: UserResponse
    recommendations: List[Dict[str, Any]]

# ---------------- Database Functions ---------------- #
async def create_user(user_data: UserCreate) -> int:
    response = supabase.table("users").insert(user_data.dict()).execute()
    if not response.data:
        raise Exception("Insert returned no data")
    return response.data[0]["id"]


async def get_user_by_id(user_id: int) -> Optional[UserResponse]:
    response = supabase.table("users").select("*").eq("id", user_id).single().execute()
    if not response.data:
        return None
    return UserResponse(**response.data)


async def generate_and_store_embedding(user_id: int, user_data: UserCreate):
    try:
        embedding = generate_embedding_from_userobj(user_data.dict())
        store_embedding(user_id, embedding)
    except Exception as e:
        print(f"Embedding generation failed for user {user_id}: {e}")



async def embedding_exists(user_id: int) -> bool:
    response = supabase.table("embeddings").select("user_id").eq("user_id", user_id).execute()
    return bool(response.data)


async def find_similar_users(user_id: int, top_n: int = 5):
    """Use Supabase RPC to fetch nearest neighbors"""
    response = supabase.rpc("match_users", {
        "query_user_id": user_id,
        "top_n": top_n,
    }).execute()
    return response.data or []


async def generate_missing_embeddings():
    users = supabase.table("users").select("*").execute().data or []
    existing = supabase.table("embeddings").select("user_id").execute().data or []
    existing_ids = {e["user_id"] for e in existing}
    missing = [u for u in users if u["id"] not in existing_ids]

    print(f"⚡ Generating embeddings for {len(missing)} users")

    for user in missing:
        try:
            emb = generate_embedding_from_userobj(user)
            store_embedding(user["id"], emb)
            print(f"Generated embedding for user {user['id']} - {user['name']}")
        except Exception as e:
            print(f"Failed to process user {user['id']}: {e}")


async def regenerate_all_embeddings():
    users = supabase.table("users").select("*").execute().data or []

    print(f"⚡ Regenerating embeddings for {len(users)} users")

    for user in users:
        try:
            emb = generate_embedding_from_userobj(user)
            store_embedding(user["id"], emb)
            print(f"Regenerated embedding for user {user['id']} - {user['name']}")
        except Exception as e:
            print(f"Failed to process user {user['id']}: {e}")


# ---------------- FastAPI App ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Embedding system ready")
    try:
        await generate_missing_embeddings()
    except Exception as e:
        print(f"Startup embedding generation failed: {e}")
    yield

app = FastAPI(title="Founder Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- API Endpoints ---------------- #
@app.post("/users", response_model=UserResponse)
async def create_user_endpoint(user_data: UserCreate, background_tasks: BackgroundTasks):
    user_id = await create_user(user_data)
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=500, detail="Failed to fetch created user")
    
    background_tasks.add_task(generate_and_store_embedding, user_id, user_data)
    return user

@app.post("/admin/regenerate-all-embeddings")
async def admin_regenerate_all_embeddings():
    await regenerate_all_embeddings()
    return {"message": "Embeddings regenerated for all users"}


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_endpoint(user_id: int):
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users/{user_id}/has_embedding")
async def has_embedding(user_id: int):
    return {"user_id": user_id, "has_embedding": await embedding_exists(user_id)}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    user = await get_user_by_id(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not await embedding_exists(request.user_id):
        raise HTTPException(status_code=400, detail="Embedding not ready")
    recs = await find_similar_users(request.user_id, request.top_n)
    return RecommendationResponse(user=user, recommendations=recs)


@app.post("/admin/generate-missing-embeddings")
async def admin_generate_missing_embeddings():
    await generate_missing_embeddings()
    return {"message": "Embeddings generated for all missing users"}


@app.get("/admin/embedding-status")
async def admin_embedding_status():
    users = supabase.table("users").select("id").execute().data or []
    embeddings = supabase.table("embeddings").select("user_id").execute().data or []
    with_emb = {e["user_id"] for e in embeddings}
    without = [u for u in users if u["id"] not in with_emb]

    return {
        "total_users": len(users),
        "users_with_embeddings": len(with_emb),
        "users_without_embeddings": len(without),
        "completion_percentage": f"{(len(with_emb) / len(users) * 100):.1f}%" if users else "0%",
    }


@app.get("/health")
async def health_check():
    try:
        supabase.table("users").select("id").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {e}"
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now(timezone.utc),
    }

# ---------------- Run ---------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
