from fastapi import FastAPI
import subprocess

app = FastAPI()

MODEL_PATH = "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf"

@app.post("/inference")
async def run_inference(prompt: str):
    command = [
        "./build/bin/llama-cli",
        "-m", MODEL_PATH,
        "-temp", "0",
        "-p", prompt
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    return {"result": result.stdout}

@app.get("/")
async def root():
    return {"message": "Welcome to the BitNet inference API"}
