# umain.py
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
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
    email: EmailStr
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
    email: EmailStr
    id: Optional[int] = None
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
    email: EmailStr
    top_n: int = 5


class RecommendationResponse(BaseModel):
    user: UserResponse
    recommendations: List[Dict[str, Any]]


# ---------------- Database Functions ---------------- #
async def create_user(user_data: UserCreate) -> str:
    payload = user_data.model_dump()  # pydantic v2-friendly
    response = supabase.table("users").insert(payload).execute()
    if not response.data:
        raise Exception("Insert returned no data")
    # return the email (primary key)
    return response.data[0].get("email")


async def get_user_by_email(email: str) -> Optional[UserResponse]:
    response = supabase.table("users").select("*").eq("email", email).single().execute()
    if not response.data:
        return None
    return UserResponse(**response.data)


async def generate_and_store_embedding(user_email: str, user_data: UserCreate):
    try:
        emb = generate_embedding_from_userobj(user_data.model_dump())
        store_embedding(user_email, emb)
    except Exception as e:
        print(f"Embedding generation failed for user {user_email}: {e}")


async def embedding_exists(email: str) -> bool:
    response = supabase.table("embeddings").select("email").eq("email", email).execute()
    return bool(response.data)


async def find_similar_users(email: str, top_n: int = 5):
    # 1. Lookup user_id from email
    user_record = supabase.table("users").select("id").eq("email", email).single().execute()
    if not user_record.data:
        raise HTTPException(status_code=404, detail=f"User {email} not found")
    user_id = user_record.data["id"]

    # 2. Call match_users RPC using user_id
    try:
        resp = supabase.rpc("match_users", {"query_user_id": user_id, "top_n": top_n}).execute()
        return resp.data or []
    except Exception as e:
        print(f"RPC call failed for user_id {user_id}: {e}")
        return []



async def generate_missing_embeddings():
    users = supabase.table("users").select("*").execute().data or []
    existing = supabase.table("embeddings").select("email", "user_id").execute().data or []

    existing_emails = {e.get("email") for e in existing if e.get("email")}
    existing_ids = {e.get("user_id") for e in existing if e.get("user_id")}

    missing = []
    for u in users:
        u_email = u.get("email")
        u_id = u.get("id")
        if (u_email and u_email not in existing_emails) and (u_id not in existing_ids):
            missing.append(u)

    print(f"⚡ Generating embeddings for {len(missing)} users")

    for user in missing:
        try:
            emb = generate_embedding_from_userobj(user)
            # prefer email if present
            email_val = user.get("email") or user.get("email")
            if not email_val:
                print(f"Skipping user (no email): {user}")
                continue
            store_embedding(email_val, emb)
            print(f"Generated embedding for user {email_val} - {user.get('name')}")
        except Exception as e:
            print(f"Failed to process user {user.get('email') or user.get('id')}: {e}")


async def regenerate_all_embeddings():
    users = supabase.table("users").select("*").execute().data or []
    print(f"⚡ Regenerating embeddings for {len(users)} users")
    for user in users:
        try:
            emb = generate_embedding_from_userobj(user)
            email_val = user.get("email")
            if not email_val:
                print("Skipping user without email:", user)
                continue
            store_embedding(email_val, emb)
            print(f"Regenerated embedding for user {email_val} - {user.get('name')}")
        except Exception as e:
            print(f"Failed to process user {user.get('email') or user.get('id')}: {e}")


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
    user_email = await create_user(user_data)
    user = await get_user_by_email(user_email)
    if not user:
        raise HTTPException(status_code=500, detail="Failed to fetch created user")
    # run embedding generation in background
    background_tasks.add_task(generate_and_store_embedding, user_email, user_data)
    return user


@app.get("/users/{email}", response_model=UserResponse)
async def get_user_endpoint(email: str):
    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users/{email}/has_embedding")
async def has_embedding(email: str):
    return {"email": email, "has_embedding": await embedding_exists(email)}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    user = await get_user_by_email(request.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not await embedding_exists(request.email):
        raise HTTPException(status_code=400, detail="Embedding not ready")
    recs = await find_similar_users(request.email, request.top_n)
    return RecommendationResponse(user=user, recommendations=recs)


@app.post("/admin/generate-missing-embeddings")
async def admin_generate_missing_embeddings():
    await generate_missing_embeddings()
    return {"message": "Embeddings generated for all missing users"}

@app.post("/admin/regenerate-all-embeddings")
async def admin_regenerate_all_embeddings():
    await regenerate_all_embeddings()
    return {"message": "Embeddings regenerated for all users"}


@app.get("/admin/embedding-status")
async def admin_embedding_status():
    users = supabase.table("users").select("id", "email").execute().data or []
    embeddings = supabase.table("embeddings").select("email", "user_id").execute().data or []
    with_emb = {e.get("email") for e in embeddings if e.get("email")}
    without = [u for u in users if u.get("email") not in with_emb]
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
