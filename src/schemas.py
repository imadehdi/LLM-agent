from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class ProblemType(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    UNKNOWN = "unknown"

class ObjectiveConfig(BaseModel):
    type: Literal["linear", "quadratic", "ratio", "minimax", "mad", "turnover"] = Field(..., description="Mathematical type of the objective function.")
    target_name: str = Field(..., description="The exact name of the data column or input matrix to optimize.")
    secondary_target_name: str = Field(default="", description="The name of the secondary input matrix or column (required for ratio, else '').")
    variable_name: str = Field("x", description="Nom de la variable de décision à utiliser dans l'objectif")
    direction: Literal["min", "max"]

class DecisionVariable(BaseModel):
    name: str = Field(..., description="Nom symbolique de la variable (ex: 'x')")
    size: Literal["n_rows", "scalar"] = Field(..., description="Taille de la variable")
    # NOUVEAU : Spécification du domaine de la variable pour le monde discret
    type: Literal["continuous", "binary", "integer"] = Field(default="continuous", description="Domaine mathématique de la variable (continu, binaire ou entier)")

class GenericConstraint(BaseModel):
    applied_to: str = Field(..., description="Nom de la variable de décision impactée (ex: 'x' ou 'b')")
    attribute: str = Field(..., description="Nom exact de la colonne du tableau, ou 'sum_all', ou 'element'")
    targets: List[str] = Field(default_factory=list, description="Valeurs textuelles ciblées si colonne catégorielle. Sinon vide.")
    bound_type: Literal["eq", "max", "min", "range"]
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class OptimizationConfig(BaseModel):
    decision_variables: List[DecisionVariable]
    objective: ObjectiveConfig
    constraints: List[GenericConstraint]

class FormulatedOptimizationProblem(BaseModel):
    problem_type: ProblemType
    title: str
    user_intent_summary: str
    optimization_config: OptimizationConfig