import gradio as gr
from fastapi import FastAPI
import sys
import os

# Ensure we can import from backend/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main FastAPI app
from backend.main import app as fastapi_app

# Define a simple Gradio dashboard to show backend status and API documentation links
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Voice-RAG Backend (MSMARCO-XI)
    
    ### Deployed successfully on HuggingFace Spaces (16GB RAM Free Tier)
    
    The FastAPI backend is fully active and mounted here.
    
    * **Interactive API Docs (Swagger):** [Click here to view /docs](/docs)
    * **Health Check Endpoint:** [Click here to view /health](/health)
    * **Frontend connection URL:** Use this Space's base URL for `NEXT_PUBLIC_API_URL`.
    """)

# Mount our FastAPI app directly onto Gradio
# Gradio runs on port 7860, so this makes all our FastAPI routes (/api/query, /api/voice, etc.)
# accessible at the Space's public URL!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
