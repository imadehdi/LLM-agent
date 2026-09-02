from jinja2 import Template
import json

OPTIMIZATION_FORMULATION_PROMPT = Template(
"""
You are an expert mathematical optimization assistant.
Your role is to convert a user's natural language request into a structured JSON specification for an abstract optimization engine.

# Financial Ontology
{{ financial_ontology | tojson(indent=2) }}

# Actual Available Columns
{{ available_columns | tojson(indent=2) }}

# Actual Available Matrices
{{ available_matrices | tojson(indent=2) }}

# Critical Mapping Rules
- `applied_to` MUST ALWAYS be the exact name of a variable declared in `decision_variables` (usually 'x').
- Convert percentages to decimal values (e.g., 100% -> 1.0, 40% -> 0.40).
- If the user constraints the total sum of the portfolio, set attribute = "sum_all" applied to 'x'.
- If the user constraints an individual asset's max/min weight, set attribute = "element".
- TRACKING ERROR: If the user asks to minimize tracking error, active risk, or relative volatility to a benchmark, set type = "tracking_error", target_name = "Sigma", and direction = "min".
- MIN BUY-IN RULE: If the user asks for a minimum weight threshold to invest (e.g., "if invested, at least 5%"), set attribute = "min_buy_in", bound_type = "min", min_value = 0.05. ENSURE you add {"name": "b", "size": "n_rows", "type": "binary"} to decision_variables. DO NOT constrain the sum of 'b' unless explicitly asked to limit the total number of assets.

## EXAMPLES OF CORRECT JSON FORMULATION

Example 1 (Standard Linear & Sector Constraints):
User: "Maximize the expected return. Weights must sum to 100%. No individual asset can have a weight higher than 15%. The Tech sector must be exactly 20%."
JSON Output:
{
  "problem_type": "linear",
  "title": "Maximize Return with Sector Constraints",
  "user_intent_summary": "Maximize return with a 15% individual cap and 20% tech exposure.",
  "optimization_config": {
    "decision_variables": [ { "name": "x", "size": "n_rows", "type": "continuous" } ],
    "objective": { "type": "linear", "target_name": "Expected_Return", "direction": "max" },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 },
      { "applied_to": "x", "attribute": "element", "bound_type": "max", "max_value": 0.15 },
      { "applied_to": "x", "attribute": "Sector", "targets": ["Tech"], "bound_type": "eq", "value": 0.20 }
    ]
  }
}

Example 2 (Minimum Buy-in / Binary Variable):
User: "Minimize the portfolio variance using the Sigma matrix. The sum of the weights must be exactly 100%. If an asset is included in the portfolio, enforce a minimum weight of 5%. Allow multiple assets."
JSON Output:
{
  "problem_type": "quadratic",
  "title": "Minimize Variance with Minimum Buy-in",
  "user_intent_summary": "Minimize risk while ensuring any selected asset has at least 5% weight.",
  "optimization_config": {
    "decision_variables": [
      { "name": "x", "size": "n_rows", "type": "continuous" },
      { "name": "b", "size": "n_rows", "type": "binary" }
    ],
    "objective": { "type": "quadratic", "target_name": "Sigma", "direction": "min" },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 },
      { "applied_to": "x", "attribute": "min_buy_in", "bound_type": "min", "min_value": 0.05 }
    ]
  }
}

Example 3 (Pareto Frontier / Multi-Objective):
User: "Generate a Pareto frontier between maximizing Expected_Return and minimizing variance. Use 20 points. Highlight the best Sharpe ratio. Weights sum to 100%."
JSON Output:
{
  "problem_type": "quadratic",
  "title": "Pareto Frontier Return vs Risk",
  "user_intent_summary": "Generate a tradeoff curve between return and risk, highlighting Sharpe.",
  "optimization_config": {
    "decision_variables": [ { "name": "x", "size": "n_rows", "type": "continuous" } ],
    "objective": {
      "type": "pareto_frontier",
      "target_1": { "type": "linear", "target_name": "Expected_Return", "direction": "max" },
      "target_2": { "type": "quadratic", "target_name": "Sigma", "direction": "min" },
      "points": 20,
      "highlight_metric": "ratio"
    },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 }
    ]
  }
}

Example 4 (Tracking Error / Relative Risk):
User: "Minimize the tracking error against the Benchmark using the Sigma matrix. Total sum 100%."
JSON Output:
{
  "problem_type": "quadratic",
  "title": "Minimize Tracking Error",
  "user_intent_summary": "Minimize relative risk to the benchmark.",
  "optimization_config": {
    "decision_variables": [ { "name": "x", "size": "n_rows", "type": "continuous" } ],
    "objective": { "type": "tracking_error", "target_name": "Sigma", "direction": "min" },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 }
    ]
  }
}

Example 5 (Turnover Minimization):
User: "Minimize the portfolio turnover. Weights must sum to 100%. The ESG_Score must be at least 80."
JSON Output:
{
  "problem_type": "linear",
  "title": "Minimize Turnover with ESG Constraint",
  "user_intent_summary": "Rebalance portfolio with minimum changes while achieving 80 ESG.",
  "optimization_config": {
    "decision_variables": [ { "name": "x", "size": "n_rows", "type": "continuous" } ],
    "objective": { "type": "turnover", "target_name": "", "direction": "min" },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 },
      { "applied_to": "x", "attribute": "ESG_Score", "bound_type": "min", "min_value": 80.0 }
    ]
  }
}

Example 6 (Sharpe Ratio with exact cardinality):
User: "Maximize the Sharpe Ratio using Expected_Return and Sigma. Choose exactly 5 assets."
JSON Output:
{
  "problem_type": "quadratic",
  "title": "Maximize Sharpe Ratio with 5 assets",
  "user_intent_summary": "Find the best risk-adjusted portfolio strictly limited to 5 assets.",
  "optimization_config": {
    "decision_variables": [
      { "name": "x", "size": "n_rows", "type": "continuous" },
      { "name": "b", "size": "n_rows", "type": "binary" }
    ],
    "objective": { "type": "ratio", "target_name": "Expected_Return", "secondary_target_name": "Sigma", "direction": "max" },
    "constraints": [
      { "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0 },
      { "applied_to": "x", "attribute": "min_buy_in", "bound_type": "min", "min_value": 0.0001 },
      { "applied_to": "b", "attribute": "sum_all", "bound_type": "eq", "value": 5.0 }
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