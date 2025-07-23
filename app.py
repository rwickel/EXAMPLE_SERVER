from flask import Flask, request, jsonify
import ollama # The official Ollama Python client
# from ollama import ChatResponse # Not strictly needed here, as ollama.Client handles it
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Configure the Ollama client
# IMPORTANT: The host should be the base URL of your Ollama server.
# It should NOT include /api/chat or /v1/. The client adds that.
client = ollama.Client(host='http://localhost:11434', timeout=240.0)

# Define default values for LLM parameters or make them configurable outside the function
# These cannot be `self.` since `chat` is a standalone Flask route function.
# You could load these from a config file or environment variables in a real app.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9 # A common default for top_p
DEFAULT_MAX_COMPLETION_TOKENS = 1024 # Or a suitable default for your use case
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_THINK = False # As per your 'think' key

@app.route('/')
def home():
    """A simple welcome message for the API server."""
    return "Welcome to the Ollama API Server!"

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handles chat completion requests by proxying them to a local Ollama server.
    Expects a JSON payload with 'model', 'messages', and optional 'stream',
    'temperature', 'top_p', 'max_completion_tokens', 'presence_penalty',
    'frequency_penalty', 'think'.
    """
    try:
        data = request.json
    except Exception as e:
        return jsonify({"error": f"Invalid JSON in request body: {e}"}), 400

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    model = data.get('model')
    messages = data.get('messages')

    # Basic input validation
    if not model:
        return jsonify({"error": "Model name ('model') is required"}), 400
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Messages ('messages') must be a list of chat message objects"}), 400

    # Extract optional parameters with sensible defaults
    stream = data.get('stream', False)
    # Note: 'tools' is not a standard top-level parameter for ollama.chat()
    # If your specific Ollama model supports function calling/tools via a custom setup,
    # you'd integrate it differently, likely by adding a 'tools' key within the `options`
    # or by transforming `data.get('tools')` into the expected format for Ollama's API.
    # For now, commenting it out to avoid passing an unsupported parameter.
    # tools = data.get('tools')

    # Construct the parameters for the ollama.Client.chat() call
    chat_params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            # Use .get() with fallback to defaults, or ensure they are present if required
            "temperature": data.get('temperature', DEFAULT_TEMPERATURE),
            "top_p": data.get('top_p', DEFAULT_TOP_P),
            "num_predict": data.get('max_completion_tokens', DEFAULT_MAX_COMPLETION_TOKENS),
            "presence_penalty": data.get('presence_penalty', DEFAULT_PRESENCE_PENALTY),
            "frequency_penalty": data.get('frequency_penalty', DEFAULT_FREQUENCY_PENALTY),
        },
        # 'think' is not a standard Ollama parameter for `ollama.chat`
        # If this is for a custom wrapping layer (like from `repl`),
        # it needs to be handled by that layer, not passed directly to ollama.Client.chat.
        # Removing for direct Ollama client use.
        # "think": data.get('think', DEFAULT_THINK),
    }

    try:
        # Call the Ollama API with the constructed parameters
        # If 'stream' is True, the response will be an iterator.
        # For non-streaming, it's a dict.
        if stream:
            # Handle streaming response
            def generate_stream():
                for chunk in client.chat(**chat_params):
                    # Each chunk is a dictionary. You might want to yield a JSON string.
                    yield f"data: {json.dumps(chunk)}\n\n"
            # Flask's streaming response
            from flask import Response
            return Response(generate_stream(), mimetype='text/event-stream')
        else:
            response = client.chat(**chat_params)
            # The ollama-python client typically returns a dictionary directly for non-streaming
            # No need for .to_dict() unless it's a Pydantic model object, which it isn't always.
            print(response.message)
            return jsonify(response.model_dump())
        
    except ollama.ResponseError as e:
        # Specific error handling for Ollama API issues
        print(f"Ollama Response Error: Status {e.status_code}, Body: {e.response_body.decode('utf-8')}")
        return jsonify({"error": f"Ollama API error: {e.status_code} - {e.response_body.decode('utf-8')}"}), e.status_code
    except Exception as e:
        # General error handling
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure Ollama server is running on http://localhost:11434
    # and you have the desired models pulled (e.g., 'ollama pull llama2')
    app.run(debug=True, host='0.0.0.0', port=5000)