import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AIAdvisor:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    def generate_hint(self, diagnosis_data: dict) -> str:
        leak_detected = diagnosis_data.get("leak_detected", False)
        zone = diagnosis_data.get("suspected_zone", "Healthy")
        residuals = diagnosis_data.get("residuals", {})
        
        if not leak_detected:
            return "All twin residuals within normal bounds. Energy field correlation manifold is stable. No action required."

        prompt = f"""
        You are an expert diesel engine diagnostic AI specialized in the Cat C18 engine.
        The LeakSense Twin system has detected a leak in the engine.
        
        DIAGNOSIS DETAILS:
        - Suspected Zone: {zone}
        - MAF Residual: {residuals.get('MAF', 0)}
        - Boost Residual: {residuals.get('Boost', 0)}
        - Exhaust Temp Residual: {residuals.get('Exhaust_Temp', 0)}
        
        TASK:
        Provide a concise, professional diagnostic hint (1-2 sentences) for a maintenance engineer.
        Focus on the most likely physical components to inspect in the {zone}.
        Output only the hint text in quotes.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior Cat C18 engine diagnostic specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating AI hint: {e}")
            return f"Inspect the components in {zone}. Check for loose connections or damaged seals."

if __name__ == "__main__":
    # Quick test
    advisor = AIAdvisor()
    test_data = {
        "leak_detected": True,
        "suspected_zone": "Zone 1 — Intake",
        "residuals": {"MAF": 12.5, "Boost": -2.1, "Exhaust_Temp": 5.0}
    }
    print(advisor.generate_hint(test_data))
