from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

def capture_output(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output = []
    for stdout_line in iter(process.stdout.readline, ""):
        output.append(stdout_line.strip())
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        return f"Error: {process.stderr.read()}"
    return "\n".join(output)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    prompt = data.get('prompt', 'The meaning of life is')
    n = data.get('n', 6)  # Default to 6 if not provided
    temp = data.get('temp', 0)  # Default to 0 if not provided

    command = [
        "python3", "run_inference.py",
        "-m", "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf",
        "-p", prompt,
        "-n", str(n),
        "-temp", str(temp)
    ]

    result = capture_output(command)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
