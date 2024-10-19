from fastapi import FastAPI, HTTPException
import subprocess
import logging
import shlex

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf"

@app.post("/inference")
def run_inference(prompt: str):
    logger.info(f"Received inference request with prompt: {prompt}")
    
    command = [
        "./build/bin/llama-cli",
        "-m", MODEL_PATH,
        "-p", prompt
    ]
    
    try:
        # Use shlex.join to properly quote the command arguments
        quoted_command = shlex.join(command)
        logger.info(f"Executing command: {quoted_command}")
        
        # Use shell=True to ensure the command is interpreted correctly
        result = subprocess.run(quoted_command, shell=True, capture_output=True, text=True, check=True)
        
        logger.info("Inference completed successfully")
        return {"result": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during inference: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail="Inference failed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.get("/")
def root():
    return {"message": "Welcome to the BitNet inference API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
