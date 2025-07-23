from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
from openai import OpenAI # Import the OpenAI client

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Configure the OpenAI client to point to your Ollama server's OpenAI-compatible API
# Ollama's OpenAI-compatible API is typically at /v1/
# The api_key can be anything non-empty for Ollama
client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama', timeout=240.0)

# Define default values for LLM parameters
# These align with typical OpenAI parameters, which Ollama also supports via its API
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0 # OpenAI default for top_p is 1.0
DEFAULT_MAX_TOKENS = None # OpenAI's `max_tokens` is often optional/None for full generation
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0

@app.route('/')
def home():
    """A simple welcome message for the API server."""
    return "Welcome to the Ollama (via OpenAI API) Server!"

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handles chat completion requests by proxying them to a local Ollama server
    using the OpenAI API client.
    Expects a JSON payload with 'model', 'messages', and optional 'stream',
    'temperature', 'top_p', 'max_tokens', 'presence_penalty',
    'frequency_penalty'.
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

    # Construct the parameters for the OpenAI.chat.completions.create() call
    # Note the change from 'num_predict' to 'max_tokens' for OpenAI compatibility
    chat_params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": data.get('temperature', DEFAULT_TEMPERATURE),
        "top_p": data.get('top_p', DEFAULT_TOP_P),
        "max_tokens": data.get('max_completion_tokens', DEFAULT_MAX_TOKENS), # Renamed for OpenAI client
        "presence_penalty": data.get('presence_penalty', DEFAULT_PRESENCE_PENALTY),
        "frequency_penalty": data.get('frequency_penalty', DEFAULT_FREQUENCY_PENALTY),
        # If your Ollama model supports tools/function calling via OpenAI spec,
        # you would pass them here:
        # "tools": data.get('tools'),
        # "tool_choice": data.get('tool_choice'),
    }

    try:
        # Call the OpenAI API client's chat completions create method
        if stream:
            # Handle streaming response
            def generate_stream():
                # The OpenAI client's stream is an iterator of ChatCompletionChunk objects
                for chunk in client.chat.completions.create(**chat_params):
                    # Each chunk.model_dump() is a dictionary suitable for JSON serialization
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            return Response(generate_stream(), mimetype='text/event-stream')
        else:
            response_obj = client.chat.completions.create(**chat_params)
            # OpenAI client's non-streaming response is a ChatCompletion object (Pydantic model)
            # Use .model_dump() to convert it to a dictionary for jsonify
            print(f"Response:\n{response_obj.choices[0].message.content}\n\n")
            return jsonify(response_obj.model_dump())

    except Exception as e:
        # The OpenAI client raises its own exceptions (e.g., openai.APIError)
        # We catch a general Exception here for simplicity, but you could
        # import openai.APIError and catch it specifically for more granular handling.
        print(f"An error occurred with the OpenAI client: {str(e)}")
        # Check if the error has a status code (typical for API errors)
        status_code = getattr(e, 'status_code', 500)
        return jsonify({"error": f"API call failed: {str(e)}"}), status_code

if __name__ == '__main__':
    # --- IMPORTANT ---
    # 1. Ensure 'ollama serve' is running in a separate terminal.
    # 2. Ensure you have the desired Ollama model pulled (e.g., 'ollama pull llama2').
    #    The `base_url` for the OpenAI client correctly points to Ollama's
    #    OpenAI-compatible endpoint.

    app.run(debug=True, host='0.0.0.0', port=5000)