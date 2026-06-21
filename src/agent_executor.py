import casadi as ca
import pandas as pd
import numpy as np

class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        n_rows = len(df_data)
        from src.casadi_builder import CasadiProblemBuilder
        from src.solver import solve_optimization
        
        builder = CasadiProblemBuilder(n_rows=n_rows)
        
        # 1. Création des variables et détection du monde discret
        decision_vars_list = optimization_config.get("decision_variables", [])
        builder.create_variables(decision_vars_list)
        
        # Construction du vecteur d'indicateurs discrets pour BONMIN
        discrete_flags = []
        for var_cfg in decision_vars_list:
            v_type = var_cfg.get("type", "continuous")
            v_size = n_rows if var_cfg.get("size") == "n_rows" else 1
            
            # True si binaire ou entier, False si continu
            is_discrete = True if v_type in ["binary", "integer"] else False
            discrete_flags.extend([is_discrete] * v_size)

        # Couplage automatique Big-M : Si 'x' et 'b' coexistent, on impose x <= b (c'est-à-dire x - b <= 0)
        # CORRECTION : On boucle sur n_rows pour bien ajouter 1 borne par actif !
        if "x" in builder.vars and "b" in builder.vars:
            x_var = builder.vars["x"]
            b_var = builder.vars["b"]
            for i in range(n_rows):
                builder.add_constraint(x_var[i] - b_var[i], -ca.inf, 0.0)

        # 2. Construction de la fonction objective
        obj_config = optimization_config.get("objective", {})
        obj_type = obj_config.get("type")                   
        target_name = obj_config.get("target_name")         
        secondary_target_name = obj_config.get("secondary_target_name") 
        var_name = obj_config.get("variable_name", "x")     
        direction = obj_config.get("direction", "min")       
        current_var = builder.vars[var_name]

        eps = 1e-5

        if obj_type == "quadratic":
            matrix_dm = ca.DM(matrix_inputs[target_name]) 
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            builder.set_objective(obj_expr, sense=direction)
            
        elif obj_type == "linear":
            objective_vector = df_data[target_name].values.reshape(1, -1)
            obj_expr = ca.mtimes(objective_vector, current_var)
            builder.set_objective(obj_expr, sense=direction)
            
        elif obj_type == "ratio":
            return_vector = df_data[target_name].values.reshape(1, -1)
            lin_return_expr = ca.mtimes(return_vector, current_var)
            matrix_dm = ca.DM(matrix_inputs[secondary_target_name])
            variance_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            obj_expr = lin_return_expr / ca.sqrt(variance_expr)
            builder.set_objective(obj_expr, sense=direction)
            
        elif obj_type == "minimax":
            scenarios_matrix = ca.DM(matrix_inputs[target_name])
            scenario_values = ca.mtimes(scenarios_matrix, current_var)
            obj_expr = 0.005 * ca.log(ca.sum1(ca.exp(scenario_values / 0.005)))
            builder.set_objective(obj_expr, sense=direction)
            
        elif obj_type == "mad":
            scenarios_np = np.array(matrix_inputs[target_name])
            T_scenarios, _ = scenarios_np.shape
            mean_vector = np.mean(scenarios_np, axis=0).reshape(1, -1)
            deviations_np = scenarios_np - mean_vector
            portfolio_deviations = ca.mtimes(ca.DM(deviations_np), current_var)
            obj_expr = ca.sum1(ca.sqrt(portfolio_deviations**2 + eps)) / T_scenarios
            builder.set_objective(obj_expr, sense=direction)
            
        elif obj_type == "turnover":
            x0_vector = ca.DM(matrix_inputs["x0"])
            diff = current_var - x0_vector
            obj_expr = ca.sum1(ca.sqrt(diff**2 + eps))
            builder.set_objective(obj_expr, sense=direction)

        else:
            raise ValueError(f"Type d'objectif inconnu : {obj_type}")

        # 3. Application des contraintes utilisateur
        builder.apply_smart_constraints(optimization_config.get("constraints", []), df_data)
        
        # 4. Préparation du vecteur initial x0
        x0_mesh = matrix_inputs.get("x0")
        if "b" in builder.vars:
            x0_mesh = x0_mesh + [1.0] * n_rows
            
        return solve_optimization(builder, x0=x0_mesh, discrete_vector=discrete_flags)