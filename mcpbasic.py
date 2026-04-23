import subprocess
import json
import threading
import queue
import os
import sys
import webbrowser
from dotenv import load_dotenv
from groq import Groq
from mcp.server.fastmcp import FastMCP

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in .env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# MCP SERVER
# ==============================
mcp = FastMCP("Groq LLM Server")

@mcp.tool(description="Generate response using Groq LLaMA 3.1 model")
def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


# ==============================
# MCP CLIENT
# ==============================
class MCPClient:
    def __init__(self, script_name):
        self.process = subprocess.Popen(
            [sys.executable, script_name, "server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self.response_queue = queue.Queue()

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        self._initialize()

    def _read_stdout(self):
        while True:
            line = self.process.stdout.readline()
            if line:
                self.response_queue.put(line.strip())

    def _read_stderr(self):
        while True:
            line = self.process.stderr.readline()
            if line:
                print("SERVER LOG:", line.strip())

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def receive(self):
        while True:
            msg = self.response_queue.get()
            try:
                return json.loads(msg)
            except:
                continue

    def _initialize(self):
        self.send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {"name": "streamlit-client", "version": "1.0"}
            }
        })
        self.receive()

        self.send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })

    def ask(self, prompt):
        self.send({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ask_llm",
                "arguments": {"prompt": prompt}
            }
        })

        resp = self.receive()

        content = resp.get("result", {}).get("content", [])
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]

        return "\n".join(texts) if texts else str(resp)


# ==============================
# STREAMLIT APP
# ==============================
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="MCP Chat", layout="centered")
    st.title("🚀 MCP Chat (Groq + LLaMA)")

    # Initialize MCP client once
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient(sys.argv[0])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    prompt = st.chat_input("Type your message...")

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        response = st.session_state.mcp_client.ask(prompt)

        # Show assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # MCP SERVER MODE
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print("MCP Server started", file=sys.stderr)
        mcp.run()

    # STREAMLIT MODE
    elif len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()

    # DEFAULT: launch streamlit + open browser
    else:
        print("🚀 Launching Streamlit UI...")

        url = "http://localhost:8501"

        # Open browser automatically
    
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", sys.argv[0], "--", "streamlit"
        ])