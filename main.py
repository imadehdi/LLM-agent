import os
import numpy as np
import pandas as pd
import casadi as ca
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent_formulator import FormulationAgent
from src.casadi_builder import CasadiProblemBuilder
from src.solver import solve_optimization

def main():
    load_dotenv()
    console = Console()

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    user_request = """
    Minimize global risk.
    Weights must sum to 100%.
    Maximum weight for a single asset is 25%.
    The Tech sector must not exceed 40% of the portfolio.
    The portfolio expected return must be at least 8.5%.
    """

    console.print(Panel.fit("ÉTAPES 0 — Requête brute de l'utilisateur", style="bold blue"))
    console.print(user_request.strip())

    # Données spécifiques à l'application financière
    df_assets = pd.DataFrame({
        'ticker': ["AAPL", "MSFT", "JNJ", "PFE", "JPM", "GS"],
        'sector': ["Tech", "Tech", "Health", "Health", "Finance", "Finance"],
        'expected_return': [0.12, 0.11, 0.06, 0.05, 0.09, 0.08]
    })

    n = len(df_assets)
    np.random.seed(42)
    A = np.random.normal(size=(n, n))
    Sigma = (A @ A.T / n + np.eye(n) * 0.01).tolist()
    
    # Pack de matrices pures pour l'optimiseur
    matrix_inputs = {
        "Sigma": Sigma,
        "x0": [1.0 / n] * n
    }
    available_columns = df_assets.columns.tolist()

    # ============================================================
    # ÉTAPE 1 — LE LLM GÉNÈRE LE JSON ABSTRAIT
    # ============================================================
    console.print(Panel.fit("ÉTAPE 1 — Extraction du JSON abstrait (Agent 1 - LLM)", style="bold blue"))
    
    formulator = FormulationAgent(model=model, host=host, verbose=False)
    
    # 1. Le dictionnaire métier reçu
    ontology_dict = {
  "metric_zones": [
    {
      "name": "allocation",
      "description": "Mesures d'exposition et de pondération dans le portefeuille.",
      "examples": [
        "weight",
        "active_weight",
        "market_value"
      ]
    },
    {
      "name": "performance",
      "description": "Mesures de rendement et de performance.",
      "examples": [
        "expected_return",
        "realized_return",
        "yield",
        "yield_contribution"
      ]
    },
    {
      "name": "risk",
      "description": "Mesures globales de risque.",
      "examples": [
        "volatility",
        "tracking_error",
        "value_at_risk",
        "conditional_value_at_risk",
        "max_drawdown"
      ]
    },
    {
      "name": "risk_contribution",
      "description": "Contributions au risque.",
      "examples": [
        "risk_contribution",
        "tracking_error_contribution"
      ]
    },
    {
      "name": "fixed_income",
      "description": "Mesures obligataires.",
      "examples": [
        "duration",
        "duration_contribution",
        "spread_duration",
        "spread_duration_contribution"
      ]
    },
    {
      "name": "credit",
      "description": "Mesures liées au crédit.",
      "examples": [
        "spread",
        "rating_score"
      ]
    },
    {
      "name": "liquidity",
      "description": "Mesures de liquidité.",
      "examples": [
        "liquidity_score"
      ]
    },
    {
      "name": "cost",
      "description": "Mesures de coûts.",
      "examples": [
        "turnover",
        "transaction_cost"
      ]
    },
    {
      "name": "esg",
      "description": "Mesures ESG et climat.",
      "examples": [
        "esg_score",
        "esg_contribution",
        "carbon_intensity",
        "carbon_contribution"
      ]
    }
  ],
  "objective_functions": [
    {
      "name": "maximize_expected_return",
      "description": "Maximiser le rendement attendu du portefeuille."
    },
    {
      "name": "maximize_active_return",
      "description": "Maximiser la surperformance par rapport au benchmark."
    },
    {
      "name": "minimize_volatility",
      "description": "Minimiser la volatilité du portefeuille."
    },
    {
      "name": "minimize_variance",
      "description": "Minimiser la variance des rendements."
    },
    {
      "name": "minimize_tracking_error",
      "description": "Réduire au maximum l'écart par rapport au benchmark."
    },
    {
      "name": "maximize_sharpe_ratio",
      "description": "Maximiser le rendement ajusté du risque."
    },
    {
      "name": "maximize_sortino_ratio",
      "description": "Maximiser le rendement ajusté du risque baissier."
    },
    {
      "name": "maximize_information_ratio",
      "description": "Maximiser la surperformance ajustée du risque actif."
    },
    {
      "name": "minimize_value_at_risk",
      "description": "Réduire la perte potentielle estimée par la VaR."
    },
    {
      "name": "minimize_conditional_value_at_risk",
      "description": "Réduire les pertes extrêmes estimées par le CVaR."
    },
    {
      "name": "minimize_max_drawdown",
      "description": "Réduire la perte maximale historique."
    },
    {
      "name": "maximize_diversification",
      "description": "Améliorer la diversification du portefeuille."
    },
    {
      "name": "risk_parity",
      "description": "Équilibrer les contributions au risque."
    },
    {
      "name": "maximize_yield",
      "description": "Maximiser le rendement actuariel du portefeuille."
    },
    {
      "name": "minimize_duration",
      "description": "Réduire la sensibilité aux taux."
    },
    {
      "name": "maximize_esg_score",
      "description": "Maximiser la qualité ESG du portefeuille."
    },
    {
      "name": "minimize_carbon_intensity",
      "description": "Réduire l'empreinte carbone du portefeuille."
    },
    {
      "name": "minimize_turnover",
      "description": "Limiter les réallocations."
    },
    {
      "name": "minimize_transaction_cost",
      "description": "Réduire les coûts de transaction."
    },
    {
      "name": "maximize_liquidity",
      "description": "Améliorer la liquidité du portefeuille."
    }
  ]
}
    
    # Dans l'ÉTAPE 1 de ton main.py :
    available_columns = df_assets.columns.tolist()
    valid_matrices = list(matrix_inputs.keys()) # Renvoie ['Sigma'] de façon dynamique !

    formulator = FormulationAgent(model=model, host=host, verbose=False)
    
    problem = formulator.formulate(
        user_request=user_request, 
        valid_columns=available_columns,
        valid_matrices=valid_matrices, # Injecté dynamiquement
        ontology_dict=ontology_dict
    )


    config_dict = problem.optimization_config.model_dump()
    
    console.print(problem.optimization_config.model_dump_json(indent=2))

    # ============================================================
    # ÉTAPE 2 — EXÉCUTION DU PROBLÈME (MOTEUR ABSTRAIT)
    # ============================================================
    console.print(Panel.fit("ÉTAPE 2 & 3 — Traduction Algébrique & Résolution", style="bold yellow"))
    
    # On importe et on utilise l'exécuteur générique qu'on a codé !
    from src.agent_executor import OptimizationExecutor
    
    executor = OptimizationExecutor(verbose=True)
    
    # On lance la résolution de notre problème abstrait
    result = executor.run(
        optimization_config=config_dict,
        matrix_inputs=matrix_inputs,
        df_data=df_assets
    )
    
    console.print(f"Statut du solveur : [bold green]{result['status']}[/bold green]\n")
    
    if result["success"]:
        # Traduction métier des résultats pour l'affichage final
        optimal_weights = result["x_values"]
        
        res_table = Table(title="Résultat de l'Optimisation de Portefeuille")
        res_table.add_column("Ticker", style="cyan")
        res_table.add_column("Secteur", style="magenta")
        res_table.add_column("Rendement attendu", style="yellow")
        res_table.add_column("Poids Optimal (x)", style="bold green")
        
        for i, row in df_assets.iterrows():
            res_table.add_row(
                row['ticker'],
                row['sector'],
                f"{row['expected_return'] * 100:.1f}%",
                f"{optimal_weights[i] * 100:.2f}%"
            )
        console.print(res_table)
    else:
        console.print(f"[bold red]Erreur de résolution : {result['error']}[/bold red]")

if __name__ == "__main__":
    main()