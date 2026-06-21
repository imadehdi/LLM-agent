from __future__ import annotations
import json
import ollama
from pydantic import ValidationError
from src.schemas import FormulatedOptimizationProblem
from src.prompts import build_custom_jinja_prompt

class FormulationAgent:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0, max_retries: int = 2, verbose: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.client = ollama.Client(host=host)

    def formulate(self, user_request: str, valid_columns: list[str], valid_matrices: list[str], ontology_dict: dict) -> FormulatedOptimizationProblem:
        last_error = None
        for attempt in range(self.max_retries + 1):
            if self.verbose: print(f"[FormulationAgent] Attempt {attempt + 1}/{self.max_retries + 1}")
            try:
                full_prompt = build_custom_jinja_prompt(valid_columns, valid_matrices, ontology_dict, user_request)
                messages = [{"role": "user", "content": full_prompt}]
                if last_error:
                    messages.append({"role": "user", "content": f"Your previous output failed validation with this error:\n{last_error}\nPlease regenerate."})

                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format=FormulatedOptimizationProblem.model_json_schema(),
                    options={"temperature": self.temperature, "num_predict": 2000, "num_ctx": 4096},
                )
                content = response["message"]["content"]
                data = json.loads(content)
                return FormulatedOptimizationProblem.model_validate(data)
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                last_error = f"Error: {str(e)}"
        raise RuntimeError(f"FormulationAgent failed. Last error: {last_error}")