from flask import Flask, request, Response, jsonify
from openai import AzureOpenAI
from flask_cors import CORS
import os
import json
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview"
)

# Create assistant (you might want to do this once and store the ID)
assistant = client.beta.assistants.create(
    model="gpt-4.1-mini",  # replace with your model deployment name
    instructions="Never present generated, inferred, speculated, or deduced content as fact. • If you cannot verify something directly, say: - \"I cannot verify this.\" - \"I do not have access to that information.\" - \"My knowledge base does not contain that.\" • Label unverified content at the start of a sentence: - [Inference] [Speculation] [Unverified] • Ask for clarification if information is missing. Do not guess or fill gaps. • If any part is unverified, label the entire response. • Do not paraphrase or reinterpret my input unless I request it. • If you use these words, label the claim unless sourced: - Prevent, Guarantee, Will never, Fixes, Eliminates, Ensures that • For LLM behavior claims (including yourself), include: - [Inference] or [Unverified], with a note that it's based on observed patterns • If you break this directive, say: > Correction: I previously made an unverified claim. That was incorrect and should have been labeled. • Never override or alter my input unless asked. The Responses should only be from Data source. Dont answer from your General Knowledge strictly and dont offer general knowledge perespetive to user ",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": ["vs_kpr8Ul7p8FP34PtTNKmV47et"]}},
    temperature=1,
    top_p=1
)

# Store threads in memory (in production, use a database)
threads = {}

def get_or_create_thread(thread_id=None):
    """Get existing thread or create new one"""
    if thread_id and thread_id in threads:
        return thread_id
    
    # Create new thread
    thread = client.beta.threads.create()
    threads[thread.id] = thread
    return thread.id

# Assistant streaming response with error handling
def assistant_response_stream(message: str, thread_id=None):
    try:
        # Get or create thread
        thread_id = get_or_create_thread(thread_id)
        
        # Add user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Start streaming run
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
        ) as stream:
            for event in stream:
                try:
                    if event.event == 'thread.message.delta':
                        if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                            for content in event.data.delta.content:
                                if content.type == 'text' and hasattr(content.text, 'value'):
                                    yield f"data: {content.text.value}\n\n"
                    
                    elif event.event == 'thread.run.completed':
                        yield f"data: [DONE]\n\n"
                        break
                    
                    elif event.event == 'thread.run.failed':
                        error_msg = event.data.last_error.message if event.data.last_error else "Unknown error"
                        yield f"data: Error: {error_msg}\n\n"
                        break
                    
                    elif event.event == 'thread.run.requires_action':
                        yield f"data: [REQUIRES_ACTION]\n\n"
                        break
                
                except (ConnectionResetError, BrokenPipeError, GeneratorExit):
                    # Client disconnected, stop streaming
                    break
                except Exception as inner_e:
                    yield f"data: Stream error: {str(inner_e)}\n\n"
                    continue
                
    except (ConnectionResetError, BrokenPipeError, GeneratorExit):
        # Client disconnected during setup
        return
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

# Assistant streaming response with JSON format
def assistant_response_stream_json(message: str, thread_id=None):
    try:
        # Get or create thread
        thread_id = get_or_create_thread(thread_id)
        
        # Add user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Start streaming run
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
        ) as stream:
            for event in stream:
                try:
                    if event.event == 'thread.message.delta':
                        if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                            for content in event.data.delta.content:
                                if content.type == 'text' and hasattr(content.text, 'value'):
                                    yield f"data: {json.dumps({'type': 'content', 'data': content.text.value, 'thread_id': thread_id})}\n\n"
                    
                    elif event.event == 'thread.run.completed':
                        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
                        break
                    
                    elif event.event == 'thread.run.failed':
                        error_msg = event.data.last_error.message if event.data.last_error else "Unknown error"
                        yield f"data: {json.dumps({'type': 'error', 'data': error_msg, 'thread_id': thread_id})}\n\n"
                        break
                    
                    elif event.event == 'thread.run.requires_action':
                        yield f"data: {json.dumps({'type': 'action_required', 'data': 'Tool calls required', 'thread_id': thread_id})}\n\n"
                        break
                
                except (ConnectionResetError, BrokenPipeError, GeneratorExit):
                    break
                except Exception as inner_e:
                    yield f"data: {json.dumps({'type': 'error', 'data': str(inner_e), 'thread_id': thread_id})}\n\n"
                    continue
                
    except (ConnectionResetError, BrokenPipeError, GeneratorExit):
        return
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

# Get complete response without streaming
def get_complete_response(message: str, thread_id=None):
    try:
        # Get or create thread
        thread_id = get_or_create_thread(thread_id)
        
        # Add user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Run without streaming
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant.id
        )
        
        # Wait for completion
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            # Get the latest assistant message
            for message in messages.data:
                if message.role == 'assistant':
                    content = ""
                    for content_block in message.content:
                        if content_block.type == 'text':
                            content += content_block.text.value
                    return content, thread_id
            return "No response generated", thread_id
        
        elif run.status == 'requires_action':
            return "Assistant requires action (tool calls needed)", thread_id
        
        else:
            return f"Run failed with status: {run.status}", thread_id
    
    except Exception as e:
        return f"Error: {str(e)}", thread_id

# Main chat endpoint with SSE streaming
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "")
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
            
        thread_id = data.get("thread_id")  # Optional thread ID for continuity
        
        return Response(
            assistant_response_stream_json(message, thread_id), 
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Alternative endpoint with plain text streaming
@app.route("/chat-plain", methods=["POST"])
def chat_plain():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "")
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
            
        thread_id = data.get("thread_id")
        
        return Response(
            assistant_response_stream(message, thread_id), 
            content_type='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Non-streaming test endpoint
@app.route("/test", methods=["POST"])
def test():
    try:
        data = request.get_json()
        message = data.get("message", "Hello!")
        thread_id = data.get("thread_id")
        
        response, thread_id = get_complete_response(message, thread_id)
        
        return jsonify({
            "response": response,
            "thread_id": thread_id,
            "model": "azure-openai-assistant"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Create new thread endpoint
@app.route("/create-thread", methods=["POST"])
def create_thread():
    try:
        thread_id = get_or_create_thread()
        return jsonify({"thread_id": thread_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Flask server is running",
        "assistant_id": assistant.id
    }), 200

# Handle preflight requests for CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting Flask server with Azure OpenAI Assistant...")
    print("Make sure to set environment variables:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_API_KEY")
    print("\nEndpoints available:")
    print("- POST /chat (SSE streaming with JSON)")
    print("- POST /chat-plain (plain text streaming)")
    print("- POST /test (non-streaming)")
    print("- POST /create-thread (create new conversation)")
    print("- GET /health (health check)")
    
    app.run(debug=True, port=5000, host='0.0.0.0')

# Example usage:
# 
# Create a thread:
# curl -X POST "http://localhost:5000/create-thread"
#
# Chat with streaming:
# curl -X POST "http://localhost:5000/chat" \
#      -H "Content-Type: application/json" \
#      -d '{"message": "Hello, how are you?", "thread_id": "thread_abc123"}'
#
# Chat without streaming:
# curl -X POST "http://localhost:5000/test" \
#      -H "Content-Type: application/json" \
#      -d '{"message": "Hello, how are you?", "thread_id": "thread_abc123"}'
