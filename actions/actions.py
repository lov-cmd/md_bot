# actions/actions.py
import os
import requests
import json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"  # Ensure this matches the model you pulled in Ollama

# Load Knowledge Base
try:
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
    print("Knowledge base loaded successfully.")
except FileNotFoundError:
    print("Error: knowledge_base.txt not found.")
    KNOWLEDGE_BASE = "No knowledge base available."

class ActionQueryKnowledgeBase(Action):

    def name(self) -> Text:
        return "action_query_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1. Get User Query
        user_query = tracker.latest_message.get('text')
        print(f"Received user query: {user_query}")

        # 2. Construct the System Prompt (RAG)
        # We instruct the model to act as a tutor and use the provided context.
        system_instruction = f"""
        You are an expert AI assistant specializing in Prompt Engineering.
        Use the following Context to answer the user's question.
        If the answer is not in the context, say "Sorry, I don't have information on that topic."
        Answer in Traditional Chinese (繁體中文).

        Context:
        ---
        {KNOWLEDGE_BASE}
        ---
        """

        # 3. Prepare the Payload for Ollama
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            "stream": False, # We want the full response at once, not streaming
            "options": {
                "temperature": 0.3 # Low temperature for factual accuracy
            }
        }

        # 4. Call Ollama API
        print("Sending request to Ollama...")
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status() # Check for HTTP errors
            
            # Parse JSON response
            result = response.json()
            generated_text = result.get("message", {}).get("content", "")
            
            print(f"Ollama Response: {generated_text}")
            
            # 5. Send response to user
            dispatcher.utter_message(text=generated_text)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            dispatcher.utter_message(text="抱歉，我目前無法連接到大語言模型 (Ollama)。請檢查 Ollama 是否正在運行。")

        return []