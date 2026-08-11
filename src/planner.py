import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Planner:
    @staticmethod
    def generate_plan(user_query: str, context_length: int) -> dict:
        """
        Root AI ko bhej kar JSON plan banwaye.
        MIT raw code generate karta tha (jisme syntax errors aate the).
        Hum JSON generate karwayenge (0% syntax error).
        """
        system_prompt = """
        You are a reasoning planner. Given a user query and a large document, 
        generate a JSON plan to answer it. 
        Rules:
        1. Break the query into sub-tasks.
        2. Use 'search' to find raw chunks.
        3. Use 'sub_call' to analyze chunks.
        4. Use 'verify' to check confidence (compare two sub-calls).
        5. Use 'final' to produce the answer.
        
        Output ONLY valid JSON. No markdown, no code fences.
        """
        
        user_prompt = f"""
        User Query: {user_query}
        Total document size: {context_length} characters.
        Generate a JSON plan to answer this query accurately and cheaply.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Sasta aur tez
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}  # FORCED JSON!
        )
        
        plan = json.loads(response.choices[0].message.content)
        return plan