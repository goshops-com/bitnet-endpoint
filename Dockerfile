FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake clang git && \
    rm -rf /var/lib/apt/lists/*

# Clone BitNet repository
RUN git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git && \
    rm -rf BitNet/.git

WORKDIR /BitNet

# Install Python dependencies
RUN pip install -r requirements.txt && \
    pip install fastapi uvicorn && \
    pip cache purge

# Generate code and build
RUN python3 utils/codegen_tl2.py --model Llama3-8B-1.58-100B-tokens --BM 256,128,256,128 --BK 96,96,96,96 --bm 32,32,32,32
RUN cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
RUN cmake --build build --target llama-cli --config Release

# Download model
ADD https://huggingface.co/brunopio/Llama3-8B-1.58-100B-tokens-GGUF/resolve/main/Llama3-8B-1.58-100B-tokens-TQ2_0.gguf .
RUN echo "2565559c82a1d03ecd1101f536c5e99418d07e55a88bd5e391ed734f6b3989ac  Llama3-8B-1.58-100B-tokens-TQ2_0.gguf" | shasum -ca 256

# Copy the FastAPI app
COPY app.py .

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
