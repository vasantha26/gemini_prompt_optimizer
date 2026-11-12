import os
import traceback
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# Load GEMINI_API_KEY from a .env file next to this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = FastAPI(title="Gemini Prompt Optimizer API")

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Client, created on first request using GEMINI_API_KEY (from .env or the environment)
client = None

# gemini-3.8-flash is often overloaded (503/504, 1+ min responses); flash-lite answers in ~1s
MODEL = "gemini-3.5-flash-lite"


# Request Schema
class PromptRequest(BaseModel):
    user_id: str
    raw_prompt: str


# Database Setup
def init_db():
    conn = sqlite3.connect("prompts_database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            raw_prompt TEXT,
            optimized_prompt TEXT,
            final_output TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/optimize-and-run")
def optimize_and_run(data: PromptRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable not set!"
        )

    global client
    if client is None:
        # Fail after 60s instead of hanging if Gemini is slow
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=60_000),
        )

    try:
        # 1. Gemini System Instruction to optimize the prompt
        optimizer_system_instruction = """
        You are an expert AI Prompt Engineer. 
        Analyze the user's raw prompt, strip away all conversational fluff, greetings, and heavy tokens, 
        and rewrite it into a crisp, highly structured, token-efficient prompt (using Role, Task, and Constraints).
        Return ONLY the optimized prompt text.
        """

        # Call Gemini to optimize the prompt
        optimizer_response = client.models.generate_content(
            model=MODEL,
            contents=f"Optimize this raw prompt: {data.raw_prompt}",
            config=types.GenerateContentConfig(
                system_instruction=optimizer_system_instruction,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        optimized_prompt = optimizer_response.text.strip()

        # 2. Run the Optimized Prompt back to Gemini to get the final clean output
        # (Disabled. To enable: uncomment this block and the final_output lines below)
        # final_response = client.models.generate_content(
        #     model=MODEL,
        #     contents=optimized_prompt,
        #     config=types.GenerateContentConfig(
        #         thinking_config=types.ThinkingConfig(thinking_level="low")
        #     ),
        # )
        # final_output = final_response.text
        final_output = None

        # 3. Save everything to SQLite Database
        conn = sqlite3.connect("prompts_database.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prompt_logs (user_id, raw_prompt, optimized_prompt, final_output)
            VALUES (?, ?, ?, ?)
        """,
            (data.user_id, data.raw_prompt, optimized_prompt, final_output),
        )
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "raw_prompt": data.raw_prompt,
            "optimized_prompt": optimized_prompt,
            # "final_output": final_output,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))