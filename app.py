from flask import Flask, request, jsonify
import subprocess
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf"

@app.route('/inference', methods=['POST'])
def run_inference():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info(f"Received inference request with prompt: {prompt}")
    
    command = [
        "./build/bin/llama-cli",
        "-m", MODEL_PATH,
        "-p", prompt
    ]
    
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("Inference completed successfully")
        return jsonify({"result": result.stdout})
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during inference: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return jsonify({"error": "Inference failed", "details": e.stderr}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error occurred"}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Welcome to the BitNet inference API"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
