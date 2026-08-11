from src.loader import DocumentLoader

class SafeInterpreter:
    def __init__(self, pdf_path: str):
        self.context = DocumentLoader.load_pdf(pdf_path)
        self.answers = {}  # Store results of each step
        self.sub_call_count = 0
        self.total_cost = 0.0
        self.confidence = 0.0
    
    def execute(self, plan: dict):
        for step in plan.get("steps", []):
            step_id = step["id"]
            step_type = step["type"]
            
            if step_type == "search":
                # Sirf substring search (safe)
                keyword = step.get("keywords", [""])[0]
                chunk = self.context[:5000]  # Simulate search
                self.answers[step_id] = chunk
                
            elif step_type == "sub_call":
                self.sub_call_count += 1
                # Yahan actual OpenAI/Claude call aayegi
                # Abhi simulate kar rahe hain
                self.answers[step_id] = f"Result for {step_id}"
                self.total_cost += 0.002  # Simulated cost
                
            elif step_type == "verify":
                deps = step.get("depends_on", [])
                if deps:
                    # Confidence calculate karne ke liye answers compare karo
                    self.confidence = 0.95  # Abhi dummy, baad mein actual logic
                    
            elif step_type == "final":
                return {
                    "answer": self.answers.get(step_id, "No answer"),
                    "cost": self.total_cost,
                    "sub_calls": self.sub_call_count,
                    "confidence": self.confidence
                }
        return {"error": "No final step found"}