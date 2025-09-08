import numpy as np
from sentence_transformers import SentenceTransformer
import re
from sklearn.preprocessing import normalize
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import torch
# ---------------- Environment ---------------- #
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global model
model = None


# ---------------- Embedding Model ---------------- #
def init_model():
    """Initialize the sentence transformer model"""
    global model
    if model is None:
        torch.set_num_threads(1)  # reduce CPU memory usage
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
        print("Sentence transformer model loaded successfully")


def clean_text(text: str) -> str:
    """Clean text for embedding"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s,.-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def generate_embedding_from_userobj(user_dict: dict) -> np.ndarray:
    """Generate semantic embeddings using sentence transformers with weighted components"""
    global model
    try:
        if model is None:
            init_model()

        # Your specified weights
        weights = {
            'industry': 0.25,
            'skills': 0.20,
            'vision': 0.20,
            'role_stage': 0.18,
            'experience': 0.07,
            'education': 0.05,
            'funding': 0.03,
            'location': 0.02
        }

        # Generate semantic embeddings for each component
        def get_weighted_embedding(text, weight):
            text = clean_text(text)
            if not text or text == "UNKNOWN":
                return np.zeros(384) * weight
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding * weight

        industry_emb = get_weighted_embedding(user_dict.get('industry', ''), weights['industry'])
        skills_emb = get_weighted_embedding(user_dict.get('skills', ''), weights['skills'])
        vision_emb = get_weighted_embedding(user_dict.get('vision_statement', ''), weights['vision'])

        role_stage_text = f"{user_dict.get('role', '')} {user_dict.get('stage', '')}"
        role_stage_emb = get_weighted_embedding(role_stage_text, weights['role_stage'])

        education_emb = get_weighted_embedding(user_dict.get('education_background', ''), weights['education'])
        location_emb = get_weighted_embedding(user_dict.get('location', ''), weights['location'])

        # Numerical features
        years_exp = user_dict.get('years_experience', 0)
        funding = user_dict.get('funding_amount', 0)

        years_norm = np.tanh(years_exp / 20.0) * weights['experience'] if years_exp > 0 else 0
        funding_norm = np.tanh(funding / 1_000_000.0) * weights['funding'] if funding > 0 else 0
        numeric_features = np.array([years_norm, funding_norm])

        # Combine all components
        final_emb = np.concatenate([
            industry_emb,
            skills_emb,
            vision_emb,
            role_stage_emb,
            education_emb,
            location_emb,
            numeric_features
        ])

        # Normalize
        final_emb = normalize(final_emb.reshape(1, -1))[0]

        print(f"Generated semantic embedding with dimension: {final_emb.shape[0]}")
        return final_emb

    except Exception as e:
        print(f"Error in semantic embedding generation: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(384)  # fallback


# ---------------- Supabase Functions ---------------- #
def convert_embedding_to_list(embedding: np.ndarray) -> list:
    """Convert numpy embedding into list for Supabase storage"""
    return embedding.tolist()


def store_embedding(user_id: int, embedding: np.ndarray):
    """Store or update embedding in Supabase"""
    try:
        embedding_list = convert_embedding_to_list(embedding)

        response = supabase.table("embeddings").upsert({
            "user_id": user_id,
            "embedding": embedding_list
        }).execute()

        if response.data:
            print(f" Semantic embedding stored for user {user_id}")
        else:
            print(f"Failed to store embedding for user {user_id}: {response}")

    except Exception as e:
        print(f"Error storing embedding for user {user_id}: {e}")
        raise
