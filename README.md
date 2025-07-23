# Ollama Flask Proxy API

A lightweight Flask API server that acts as a proxy for the Ollama local AI model server, providing a convenient endpoint for chat completions. This allows you to easily integrate local large language models (LLMs) powered by Ollama into your web applications, frontend projects, or other services, with built-in CORS support.

## Features

* **Ollama Proxy:** Seamlessly forwards chat completion requests to your locally running Ollama server.
* **CORS Enabled:** Configured with `flask_cors` to handle cross-origin requests, making it easy to use with frontend applications.
* **Streaming Support:** Supports both non-streaming and Server-Sent Events (SSE) for real-time chat responses.
* **Configurable LLM Parameters:** Allows setting `temperature`, `top_p`, `num_predict` (max tokens), `presence_penalty`, and `frequency_penalty` via request body.
* **Simple & Lightweight:** Built with Flask, keeping the footprint minimal.

## Prerequisites

Before you begin, ensure you have the following installed and set up:

* **Python 3.8+**
* **Ollama:** The Ollama application must be installed and running on your system.
    * Download from: [ollama.com](https://ollama.com/)
    * Ensure the Ollama server is running, typically on `http://localhost:11434`. You can start it from your terminal:
        ```bash
        ollama serve
        ```
* **Ollama Models:** You need to have at least one LLM model pulled via Ollama. For example, to pull `llama2`:
    ```bash
    ollama pull llama2
    ```
    Or if you prefer `qwen2.5:14b`:
    ```bash
    ollama pull qwen2.5:14b
    ```
    You can list your available models with `ollama list`.

## Installation

1.  **Clone the repository (or create the file):**
    ```bash
    git clone [https://github.com/your-username/ollama-flask-api.git](https://github.com/your-username/ollama-flask-api.git)
    cd ollama-flask-api
    ```
    (If you just have the `app.py` file, create a new directory and place it there.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat
        ```
    * **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

4.  **Install dependencies:**
    ```bash
    pip install Flask ollama Flask-Cors
    ```

## Usage

### Running the API Server

From the root of your project directory (with your virtual environment activated):
```bash
python app.py
 ```

### Request the API Server

```python
import ollama

client = ollama.Client(host='http://127.0.0.1:5000/', timeout=240.0)

llm = LLM(client=client)

message = [{"role": "user", "content": "hello"}]              
response = llm.get_chat_completion(message) 
print(response) 
```
### Response API Server
<img width="1235" height="239" alt="image" src="https://github.com/user-attachments/assets/97cb821c-b666-4053-b149-42800320d007" />



