"""
JSON Planner Module
Generates structured JSON plans using OpenAI instead of raw code.
This eliminates syntax errors (MIT's main weakness).
"""

import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
import openai

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Planner:
    """Generates JSON execution plans using OpenAI."""

    @staticmethod
    def generate_plan(user_query: str, context_length: int) -> Dict[str, Any]:
        """
        Send user query to OpenAI and return a structured JSON plan.

        MIT's approach: Generate raw Python code (40% syntax error rate).
        Our approach: Force JSON output (0% syntax error).

        Args:
            user_query: The user's question.
            context_length: Length of the document in characters.

        Returns:
            A JSON dictionary containing the execution plan.
        """
        system_prompt = """
        You are a reasoning planner. Given a user query and a large document,
        generate a JSON plan to answer it.

        IMPORTANT: Output ONLY valid JSON. No markdown, no code fences.

        Plan Structure:
        {
            "steps": [
                {
                    "id": "step_1",
                    "type": "search",
                    "keywords": ["keyword1", "keyword2"],
                    "chunk_size": 5000,
                    "description": "Search document for relevant sections"
                },
                {
                    "id": "step_2",
                    "type": "sub_call",
                    "prompt": "Specific question for AI",
                    "depends_on": ["step_1"],
                    "description": "Extract specific information"
                },
                {
                    "id": "step_3",
                    "type": "verify",
                    "depends_on": ["step_2"],
                    "description": "Verify answer confidence"
                },
                {
                    "id": "step_4",
                    "type": "final",
                    "depends_on": ["step_2", "step_3"],
                    "description": "Produce final answer"
                }
            ]
        }

        Rules:
        1. Always start with a "search" step to find relevant context.
        2. Use "sub_call" for AI analysis of specific chunks.
        3. Use "verify" to check confidence (compare two sub-calls).
        4. End with a "final" step that depends on all previous steps.
        5. Keep steps minimal (2-5 steps) for cost efficiency.
        6. Only include steps that are absolutely necessary.
        """

        user_prompt = f"""
        User Query: {user_query}
        Total document size: {context_length} characters.

        Generate a JSON plan to answer this query accurately and cost-efficiently.
        Plan should have 3-5 steps maximum.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},  # FORCES JSON!
                temperature=0.3,  # More deterministic output
            )

            content = response.choices[0].message.content
            plan = json.loads(content)

            # Validate plan has required fields
            if "steps" not in plan or not plan["steps"]:
                raise ValueError("Plan missing 'steps' array")

            return plan

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from OpenAI: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate plan: {e}")

    @staticmethod
    def validate_plan(plan: Dict[str, Any]) -> bool:
        """
        Validate that the plan has required structure.

        Args:
            plan: The JSON plan to validate.

        Returns:
            True if valid, raises ValueError if invalid.
        """
        if "steps" not in plan:
            raise ValueError("Plan missing 'steps' key")

        steps = plan["steps"]
        if not isinstance(steps, list):
            raise ValueError("'steps' must be a list")

        if len(steps) < 2:
            raise ValueError("Plan must have at least 2 steps")

        step_ids = set()
        for step in steps:
            if "id" not in step:
                raise ValueError("Step missing 'id'")
            if step["id"] in step_ids:
                raise ValueError(f"Duplicate step id: {step['id']}")
            step_ids.add(step["id"])

            if "type" not in step:
                raise ValueError(f"Step {step['id']} missing 'type'")

            valid_types = {"search", "sub_call", "verify", "final"}
            if step["type"] not in valid_types:
                raise ValueError(
                    f"Invalid type '{step['type']}' in step {step['id']}"
                )

        # Check that all dependencies exist
        for step in steps:
            for dep in step.get("depends_on", []):
                if dep not in step_ids:
                    raise ValueError(
                        f"Step {step['id']} depends on unknown step: {dep}"
                    )

        # Check that final step exists
        has_final = any(step["type"] == "final" for step in steps)
        if not has_final:
            raise ValueError("Plan must have a 'final' step")

        return True