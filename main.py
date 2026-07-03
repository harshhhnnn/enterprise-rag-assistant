import os
import math
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# CONFIGURATION & API KEY SETUP
# -----------------------------------------------------------------------------
# This secretly loads the API key from your .env file
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

app = FastAPI(title="DocuMind Clean UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL IN-MEMORY DATABASE
# -----------------------------------------------------------------------------
DOCUMENT_STORE = []
CURRENT_FILE_INFO = {
    "filename": "No document loaded",
    "pages": 0,
    "chunks": 0
}

def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2: return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

# -----------------------------------------------------------------------------
# FASTAPI ROUTES (The Backend Endpoints)
# -----------------------------------------------------------------------------

@app.get("/status")
async def get_system_status():
    has_key = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY"
    return {
        "has_key": has_key,
        "file_loaded": bool(DOCUMENT_STORE),
        "file_info": CURRENT_FILE_INFO
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global DOCUMENT_STORE, CURRENT_FILE_INFO
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    client = get_gemini_client()
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Key is invalid.")

    try:
        import pypdf
    except ImportError:
        raise HTTPException(status_code=500, detail="Missing package. Run: pip install pypdf")

    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = pypdf.PdfReader(pdf_file)
        
        total_pages = len(reader.pages)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or scanned.")

        chunk_size = 1000
        overlap = 200
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)

        embed_response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=chunks
        )

        new_store = []
        for index, emb in enumerate(embed_response.embeddings):
            new_store.append({
                "text": chunks[index],
                "embedding": emb.values
            })

        DOCUMENT_STORE = new_store
        CURRENT_FILE_INFO = {
            "filename": file.filename,
            "pages": total_pages,
            "chunks": len(chunks)
        }

        return {"message": "Success", "file_info": CURRENT_FILE_INFO}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload Failed: {str(e)}")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def ask_question(request: ChatRequest):
    client = get_gemini_client()
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing.")

    if not DOCUMENT_STORE:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    try:
        query_emb_res = client.models.embed_content(
            model='gemini-embedding-001',
            contents=request.question
        )
        query_embedding = query_emb_res.embeddings[0].values

        ranked_chunks = []
        for item in DOCUMENT_STORE:
            sim = cosine_similarity(query_embedding, item["embedding"])
            ranked_chunks.append((sim, item["text"]))

        ranked_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [text for sim, text in ranked_chunks[:5]]

        context_block = "\n---\n".join(top_chunks)
        system_instruction = (
            "You are a helpful AI assistant. Answer the user's question "
            "using ONLY the retrieved document text chunks provided below.\n\n"
            f"CONTEXT:\n{context_block}"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )

        return {"answer": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")

# -----------------------------------------------------------------------------
# MOUNT STATIC FILES (Must be at the very bottom)
# -----------------------------------------------------------------------------
# This single line replaces the massive HTML_PAGE string. It tells FastAPI to 
# look in the "static" folder and serve "index.html" whenever a user visits the site!
app.mount("/", StaticFiles(directory="static", html=True), name="static")