from jinja2 import Template
import json

OPTIMIZATION_FORMULATION_PROMPT = Template(
"""
You are an expert mathematical optimization assistant.
Your role is to convert a user's natural language request into a structured JSON specification for an abstract optimization engine.

# Financial Ontology (Synonyms and Meanings)
{{ financial_ontology | tojson(indent=2) }}

# Actual Available Columns in the Dataset
{{ available_columns | tojson(indent=2) }}

# Actual Available Matrices in the Inputs
{{ available_matrices | tojson(indent=2) }}

# Critical Mapping Rules
- `applied_to` MUST ALWAYS be the exact name of a variable declared in `decision_variables`.
- Convert percentages to decimal values (e.g., 100% -> 1.0, 40% -> 0.40).

## Cardinality & Discrete Selection Rules (CRITICAL)
If the user asks to limit the number of assets/titles in the portfolio (e.g., "at most 3 assets", "maximum 15 stocks", "select 4 assets max"):
1. You MUST declare a primary continuous variable named "x" (size: "n_rows", type: "continuous").
2. You MUST declare a secondary companion binary variable named "b" (size: "n_rows", type: "binary").
3. You MUST add a constraint applied to "b" with attribute "sum_all", bound_type "max", and the value equal to the maximum number of assets allowed.

# Output Schema
{
  "problem_type": "linear" or "quadratic",
  "title": "Problem Title",
  "user_intent_summary": "Short summary",
  "optimization_config": {
    "decision_variables": [
      { "name": "x", "size": "n_rows", "type": "continuous" },
      { "name": "b", "size": "n_rows", "type": "binary" }
    ],
    "objective": {
      "type": "linear" or "quadratic" or "ratio" or "minimax" or "mad" or "turnover",
      "target_name": "string",
      "secondary_target_name": "string",
      "variable_name": "x",
      "direction": "min" or "max"
    },
    "constraints": [
      {
        "applied_to": "b",
        "attribute": "sum_all",
        "targets": [],
        "bound_type": "max",
        "value": 3.0
      }
    ]
  }
}

# Few-Shot Example
User:
Minimize global risk. Weights must sum to 100%. I want at most 3 assets in the portfolio.

Output:
{
  "problem_type": "quadratic",
  "title": "Cardinality Constrained Risk Minimization",
  "user_intent_summary": "Minimize portfolio variance with a maximum selection of 3 assets",
  "optimization_config": {
    "decision_variables": [
      { "name": "x", "size": "n_rows", "type": "continuous" },
      { "name": "b", "size": "n_rows", "type": "binary" }
    ],
    "objective": {
      "type": "quadratic",
      "target_name": "Sigma",
      "secondary_target_name": "",
      "variable_name": "x",
      "direction": "min"
    },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "targets": [], "bound_type": "eq", "value": 1.0 },
      { "applied_to": "b", "attribute": "sum_all", "targets": [], "bound_type": "max", "value": 3.0 }
    ]
  }
}

User:
{{ user_query }}

JSON Output:
"""
)

def build_custom_jinja_prompt(valid_columns: list[str], valid_matrices: list[str], ontology_dict: dict, user_request: str) -> str:
    return OPTIMIZATION_FORMULATION_PROMPT.render(
        available_columns=valid_columns,
        available_matrices=valid_matrices,
        financial_ontology=ontology_dict,
        user_query=user_request
    )