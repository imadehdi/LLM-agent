import casadi as ca
import pandas as pd
import numpy as np
import copy

class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        obj_type = optimization_config.get("objective", {}).get("type")
        
        # INTERCEPTION : Si on demande une Frontière de Pareto
        if obj_type == "pareto_frontier":
            return self._run_pareto_epsilon(optimization_config, matrix_inputs, df_data)
            
        # --- DÉBUT DE L'EXÉCUTION STANDARD (Un seul point) ---
        n_rows = len(df_data)
        from src.casadi_builder import CasadiProblemBuilder
        from src.solver import solve_optimization
        
        builder = CasadiProblemBuilder(n_rows=n_rows)
        
        decision_vars_list = optimization_config.get("decision_variables", [])
        builder.create_variables(decision_vars_list)
        
        discrete_flags = []
        for var_cfg in decision_vars_list:
            v_type = var_cfg.get("type", "continuous")
            v_size = n_rows if var_cfg.get("size") == "n_rows" else 1
            is_discrete = True if v_type in ["binary", "integer"] else False
            discrete_flags.extend([is_discrete] * v_size)

        if "x" in builder.vars and "b" in builder.vars:
            x_var = builder.vars["x"]
            b_var = builder.vars["b"]
            for i in range(n_rows):
                builder.add_constraint(x_var[i] - b_var[i], -ca.inf, 0.0)

        obj_config = optimization_config.get("objective", {})
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

        elif obj_type == "risk_budgeting":
            matrix_dm = ca.DM(matrix_inputs[target_name])
            budgets = ca.DM(matrix_inputs.get("Budgets", [1.0 / n_rows] * n_rows)) 
            variance_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            marginal_risk = ca.mtimes(matrix_dm, current_var)
            risk_contribution = current_var * marginal_risk
            target_contribution = budgets * variance_expr
            obj_expr = ca.sum1((risk_contribution - target_contribution)**2)
            builder.set_objective(obj_expr, sense="min")

        elif obj_type == "cvar":
            scenarios_matrix = ca.DM(matrix_inputs[target_name])
            T_scenarios = scenarios_matrix.shape[0]
            beta = 0.95
            
            alpha = ca.MX.sym('alpha', 1) 
            u = ca.MX.sym('u', T_scenarios) 
            
            builder.vars['alpha'] = alpha
            builder.vars['u'] = u
            builder.lbx.append(-ca.inf)
            builder.ubx.append(ca.inf)
            builder.lbx.extend([0.0] * T_scenarios) 
            builder.ubx.extend([ca.inf] * T_scenarios)
            
            portfolio_losses = ca.mtimes(scenarios_matrix, current_var)
            
            for t in range(T_scenarios):
                builder.add_constraint(u[t] + alpha - portfolio_losses[t], 0.0, ca.inf)
                
            obj_expr = alpha + (1.0 / (T_scenarios * (1.0 - beta))) * ca.sum1(u)
            builder.set_objective(obj_expr, sense=direction)

        else:
            raise ValueError(f"Type d'objectif inconnu : {obj_type}")

        builder.apply_smart_constraints(optimization_config.get("constraints", []), df_data)
        
        x0_mesh = list(matrix_inputs.get("x0", [1.0/n_rows]*n_rows))
        if "b" in builder.vars:
            x0_mesh.extend([1.0] * n_rows)
        if "alpha" in builder.vars:
            x0_mesh.append(0.0)
            discrete_flags.append(False)
        if "u" in builder.vars:
            T_scens = builder.vars["u"].shape[0]
            x0_mesh.extend([0.0] * T_scens)
            discrete_flags.extend([False] * T_scens)
            
        return solve_optimization(builder, x0=x0_mesh, discrete_vector=discrete_flags)


    # =========================================================================
    # LOGIQUE EXCLUSIVE : FRONTIÈRE DE PARETO (EPSILON-CONTRAINTE)
    # =========================================================================
    def _run_pareto_epsilon(self, config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        """Exécute l'algorithme d'Epsilon-Contrainte pour générer N portefeuilles."""
        
        obj_cfg = config["objective"]
        target_1 = obj_cfg["target_1"] # Ex: Rendement (max)
        target_2 = obj_cfg["target_2"] # Ex: Risque (min)
        points = obj_cfg.get("points", 20)

        # 1. ANCRE 1 : Portefeuille de Risque Minimum
        cfg_min_risk = copy.deepcopy(config)
        cfg_min_risk["objective"] = target_2
        res_min = self.run(cfg_min_risk, matrix_inputs, df_data)
        
        # 2. ANCRE 2 : Portefeuille de Rendement Maximum
        cfg_max_ret = copy.deepcopy(config)
        cfg_max_ret["objective"] = target_1
        res_max = self.run(cfg_max_ret, matrix_inputs, df_data)

        if not res_min["success"] or not res_max["success"]:
            return {"success": False, "error": "Impossible de trouver les ancres de la frontière."}

        # --- Calcul de l'intervalle des Variances pour découper l'Epsilon ---
        w_min = np.array(res_min["x_values"])
        w_max = np.array(res_max["x_values"])
        sigma = np.array(matrix_inputs["Sigma"])
        
        var_min = w_min.T @ sigma @ w_min
        var_max = w_max.T @ sigma @ w_max
        
        # Grille d'Epsilons (Plafonds de risque progressifs)
        epsilons = np.linspace(var_min, var_max, points)
        
        pareto_results = []
        
        # 3. LA BOUCLE DES EPSILONS
        for eps in epsilons:
            # On cherche à maximiser le Rendement
            cfg_eps = copy.deepcopy(config)
            cfg_eps["objective"] = target_1
            
            # On instancie le builder à la main pour cette boucle
            n_rows = len(df_data)
            from src.casadi_builder import CasadiProblemBuilder
            from src.solver import solve_optimization
            
            builder = CasadiProblemBuilder(n_rows=n_rows)
            builder.create_variables(config.get("decision_variables", []))
            
            # Application des contraintes de base du LLM
            builder.apply_smart_constraints(config.get("constraints", []), df_data)
            
            x_var = builder.vars["x"]
            
            # --- Objectif : Maximiser le Rendement ---
            ret_vec = df_data[target_1["target_name"]].values.reshape(1, -1)
            expr_return = ca.mtimes(ret_vec, x_var)
            builder.set_objective(expr_return, sense="max")
            
            # --- Le Mur d'Epsilon : Contrainte dure sur le Risque ---
            sigma_dm = ca.DM(matrix_inputs["Sigma"])
            expr_risk = ca.mtimes(ca.mtimes(x_var.T, sigma_dm), x_var)
            
            # Ajout de la contrainte : Variance <= Epsilon
            builder.add_constraint(expr_risk, -ca.inf, eps)
            
            # Résolution
            res = solve_optimization(builder, x0=[1.0/n_rows]*n_rows, discrete_vector=[False]*n_rows)
            
            if res["success"]:
                w_opt = np.array(res["x_values"][:n_rows])
                # On calcule les coordonnées finales du point pour le tracé
                realized_return = np.dot(w_opt, df_data[target_1["target_name"]].values)
                realized_risk = np.sqrt(w_opt.T @ sigma @ w_opt)
                
                pareto_results.append({
                    "risk": float(realized_risk),
                    "return": float(realized_return),
                    "weights": w_opt.tolist()
                })
        
        return {
            "success": True,
            "type": "pareto",
            "pareto_points": pareto_results # Retourne le tableau des 20 points !
        }