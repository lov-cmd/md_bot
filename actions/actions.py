import os
import requests
import csv
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"  # Ensure you have pulled this model: `ollama pull qwen3:4b`

class ActionQueryMcDonalds(Action):

    def name(self) -> Text:
        return "action_query_mcdonalds"

    def get_knowledge_base_content(self) -> Text:
        """
        Reads both the TXT and CSV files and combines them into a single string context.
        """
        combined_context = ""

        # 1. Read Text File (General Info)
        try:
            with open("knowledge_base.txt", "r", encoding="utf-8") as f:
                text_data = f.read()
                combined_context += f"--- GENERAL FOOD SAFETY & POLICIES ---\n{text_data}\n\n"
        except FileNotFoundError:
            print("Warning: knowledge_base.txt not found.")

        # 2. Read CSV File (Nutrition Data)
        try:
            with open("knowledge_nutrition.csv", "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                csv_text = "--- NUTRITION DATA ---\n"
                for row in reader:
                    # Convert CSV row to a readable sentence for the LLM
                    item_name = row.get("Item", "Unknown Item")
                    details = ", ".join([f"{k}: {v}" for k, v in row.items() if k != "Item"])
                    csv_text += f"Item: {item_name} | Nutritional Values: [{details}]\n"
                
                combined_context += csv_text
        except FileNotFoundError:
            print("Warning: knowledge_nutrition.csv not found.")

        return combined_context

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1. Get User Query
        user_query = tracker.latest_message.get('text')
        print(f"User Query: {user_query}")

        # 2. Load Data
        context_data = self.get_knowledge_base_content()

        # 3. Construct System Prompt
        system_instruction = f"""
        You are a helpful customer service assistant for McDonald's Hong Kong.
        Use the provided Context to answer the user's question about food safety or nutrition.
        
        Context:
        {context_data}
        
        Instructions:
        - If asked about nutrition (calories, protein, etc.), look at the 'NUTRITION DATA' section.
        - If asked about safety, allergens, or policies, look at the 'GENERAL FOOD SAFETY' section.
        - If the answer is not in the context, politely say you don't have that specific information.
        - Keep answers concise and friendly.
        """

        # 4. Prepare Payload
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2  # Low temperature for accurate data retrieval
            }
        }

        # 5. Call Ollama
        try:
            print(f"Sending request to Ollama ({MODEL_NAME})...")
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("message", {}).get("content", "")
            
            dispatcher.utter_message(text=generated_text)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            dispatcher.utter_message(text="Sorry, I am having trouble accessing the nutrition database right now.")

        return []