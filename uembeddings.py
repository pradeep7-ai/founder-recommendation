# uembeddings.py
import os
import re
import numpy as np
from sklearn.preprocessing import normalize
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Environment ---------------- #
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise Exception("Missing SUPABASE_URL, SUPABASE_KEY or OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

TARGET_DIM = 2306  # your Supabase schema expects this

# ---------------- Text Cleaning ---------------- #
def clean_text(text: str) -> str:
    """Clean text for embedding"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^\w\s,.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------- Generate Embedding ---------------- #
def generate_embedding_from_userobj(user_dict: dict) -> np.ndarray:
    """Generate combined semantic embeddings using OpenAI API"""

    def get_embedding(text: str) -> np.ndarray:
        text = clean_text(text)
        if not text or text == "UNKNOWN":
            return np.zeros(1536, dtype=np.float32)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    # 1. Industry + Skills
    industry_skills_text = f"{user_dict.get('industry', '')} {user_dict.get('skills', '')}"
    industry_skills_emb = get_embedding(industry_skills_text) * 0.25

    # 2. Vision + Role + Stage + Education + Location
    vision_text = user_dict.get("vision_statement", "")
    role_stage_text = f"{user_dict.get('role', '')} {user_dict.get('stage', '')}"
    education_text = user_dict.get("education_background", "")
    location_text = user_dict.get("location", "")
    combined_text = f"{vision_text} {role_stage_text} {education_text} {location_text}"
    combined_emb = get_embedding(combined_text) * 0.20

    # 3. Numeric features
    years_exp = user_dict.get("years_experience", 0)
    funding = user_dict.get("funding_amount", 0)
    numeric_features = np.array([
        np.tanh(years_exp / 20.0),
        np.tanh(funding / 1_000_000.0)
    ], dtype=np.float32) * 0.20

    # 4. Combine embeddings + numeric features
    final_emb = np.concatenate([industry_skills_emb, combined_emb, numeric_features])

    # Normalize
    final_emb = normalize(final_emb.reshape(1, -1))[0]

    # 5. Truncate to first TARGET_DIM dimensions (to match existing schema)
    if len(final_emb) < TARGET_DIM:
        # pad if shorter (unlikely with 1536+1536+2=3074) — but keep defensive
        final_emb = np.pad(final_emb, (0, TARGET_DIM - len(final_emb)), "constant")
    else:
        final_emb = final_emb[:TARGET_DIM]

    print(f"Generated semantic embedding with dimension: {final_emb.shape[0]}")
    return final_emb

# ---------------- Supabase Functions ---------------- #
def convert_embedding_to_list(embedding: np.ndarray) -> list:
    """Convert numpy embedding into list for Supabase storage"""
    return embedding.tolist()

def store_embedding(user_email: str, embedding: np.ndarray):
    """Store or update embedding in Supabase using user_id as FK"""
    try:
        # Step 1: Get user_id from users table
        user_response = supabase.table("users").select("id").eq("email", user_email).single().execute()
        if not user_response.data:
            raise ValueError(f"No user found with email {user_email}")

        user_id = user_response.data["id"]

        # Step 2: Convert embedding to list
        embedding_list = convert_embedding_to_list(embedding)

        # Step 3: Upsert into embeddings table
        response = supabase.table("embeddings").upsert({
            "user_id": user_id,
            "email": user_email,
            "embedding": embedding_list
        }).execute()

        if response.data:
            print(f" Semantic embedding stored for user {user_email} (user_id={user_id})")
        else:
            print(f"ℹEmbedding upsert executed for user {user_email} (no return data)")

    except Exception as e:
        print(f" Error storing embedding for user {user_email}: {e}")
        raise

