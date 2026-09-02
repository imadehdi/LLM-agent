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

        obj_config = optimization_config.get("objective", {})
        target_name = obj_config.get("target_name")         
        secondary_target_name = obj_config.get("secondary_target_name") 
        var_name = obj_config.get("variable_name", "x")     
        direction = obj_config.get("direction", "min")       
        
        # --- SÉCURITÉ ANTI-HALLUCINATION (OBJECTIF) ---
        if var_name not in builder.vars:
            var_name = "x"
        current_var = builder.vars[var_name]

        eps = 1e-5

        if obj_type == "quadratic":
            matrix_dm = ca.DM(matrix_inputs[target_name]) 
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            builder.set_objective(obj_expr, sense=direction)

        elif obj_type == "tracking_error":
            matrix_dm = ca.DM(matrix_inputs[target_name]) # La matrice Sigma
            # On récupère le benchmark (s'il n'y en a pas, on met 0 par défaut)
            xb_vector = ca.DM(matrix_inputs.get("Benchmark", [0.0] * n_rows))
            
            # La fameuse équation (x - x_b)
            active_weights = current_var - xb_vector
            
            # (x - x_b)^T * Sigma * (x - x_b)
            obj_expr = ca.mtimes(ca.mtimes(active_weights.T, matrix_dm), active_weights)
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

        elif obj_type == "tracking_error":
            # 1. On charge la matrice de covariance (Sigma)
            matrix_dm = ca.DM(matrix_inputs[target_name]) 
            
            # 2. On charge le vecteur des poids du Benchmark (ou 0 par défaut si non fourni)
            bench_list = matrix_inputs.get("Benchmark", [0.0] * n_rows)
            xb_vector = ca.DM(bench_list)
            
            # 3. Calcul des poids actifs : (x - x_b)
            active_weights = current_var - xb_vector
            
            # 4. Formule quadratique du Tracking Error : (x - x_b)^T * Sigma * (x - x_b)
            obj_expr = ca.mtimes(ca.mtimes(active_weights.T, matrix_dm), active_weights)
            builder.set_objective(obj_expr, sense=direction)

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

        # --- SÉCURITÉ ANTI-HALLUCINATION (CONTRAINTES) ---
        constraints = optimization_config.get("constraints", [])
        for c in constraints:
            if c.get("applied_to") not in builder.vars:
                c["applied_to"] = "x"  # Force le retour à la variable par défaut
                
        builder.apply_smart_constraints(constraints, df_data)
        
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
    # LOGIQUE EXCLUSIVE : FRONTIÈRE DE PARETO UNIVERSELLE
    # =========================================================================
    def _run_pareto_epsilon(self, config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        
        obj_cfg = config["objective"]
        target_1 = obj_cfg["target_1"] 
        target_2 = obj_cfg["target_2"] 
        points = obj_cfg.get("points", 20)

        def _build_target_expr_and_val(t_cfg, x_var_sym, x_val_num):
            t_type = t_cfg.get("type")
            t_name = t_cfg.get("target_name")
            if t_type == "linear":
                vec = df_data[t_name].values
                expr = ca.mtimes(vec.reshape(1, -1), x_var_sym) if x_var_sym is not None else None
                val = np.dot(x_val_num, vec) if x_val_num is not None else 0.0
                return expr, val
            elif t_type == "quadratic":
                mat = np.array(matrix_inputs[t_name])
                expr = ca.mtimes(ca.mtimes(x_var_sym.T, ca.DM(mat)), x_var_sym) if x_var_sym is not None else None
                val = np.sqrt(x_val_num.T @ mat @ x_val_num) if x_val_num is not None else 0.0
                return expr, val
            else:
                raise ValueError(f"Type de cible Pareto non supporté : {t_type}")

        # 1. ANCRE 1 : Optimiser target_2
        cfg_min_t2 = copy.deepcopy(config)
        cfg_min_t2["objective"] = target_2
        res_min = self.run(cfg_min_t2, matrix_inputs, df_data)
        
        # 2. ANCRE 2 : Optimiser target_1
        cfg_max_t1 = copy.deepcopy(config)
        cfg_max_t1["objective"] = target_1
        res_max = self.run(cfg_max_t1, matrix_inputs, df_data)

        if not res_min["success"] or not res_max["success"]:
            return {"success": False, "error": "Impossible de trouver les ancres de la frontière."}

        w_min = np.array(res_min["x_values"])
        w_max = np.array(res_max["x_values"])
        
        def _get_raw_t2_val(t_cfg, w):
            if t_cfg.get("type") == "linear":
                return np.dot(w, df_data[t_cfg.get("target_name")].values)
            elif t_cfg.get("type") == "quadratic":
                mat = np.array(matrix_inputs[t_cfg.get("target_name")])
                return w.T @ mat @ w
                
        raw_min = _get_raw_t2_val(target_2, w_min)
        raw_max = _get_raw_t2_val(target_2, w_max)
        epsilons = np.linspace(raw_min, raw_max, points)
        
        pareto_results = []
        n_rows = len(df_data)
        
        for eps in epsilons:
            from src.casadi_builder import CasadiProblemBuilder
            from src.solver import solve_optimization
            
            builder = CasadiProblemBuilder(n_rows=n_rows)
            builder.create_variables(config.get("decision_variables", []))
            
            # --- SÉCURITÉ ANTI-HALLUCINATION POUR PARETO ---
            constraints = config.get("constraints", [])
            for c in constraints:
                if c.get("applied_to") not in builder.vars:
                    c["applied_to"] = "x"
            builder.apply_smart_constraints(constraints, df_data)
            
            # On sécurise la récupération de la variable ici aussi
            var_name = target_1.get("variable_name", "x")
            if var_name not in builder.vars:
                var_name = "x"
            x_var = builder.vars[var_name]
            
            # Objectif : Target 1
            expr_t1, _ = _build_target_expr_and_val(target_1, x_var, None)
            builder.set_objective(expr_t1, sense=target_1.get("direction", "max"))
            
            # Contrainte Epsilon : Target 2 <= eps
            expr_t2, _ = _build_target_expr_and_val(target_2, x_var, None)
            builder.add_constraint(expr_t2, -ca.inf, eps)
            
            res = solve_optimization(builder, x0=[1.0/n_rows]*n_rows, discrete_vector=[False]*n_rows)
            
            if res["success"]:
                w_opt = np.array(res["x_values"][:n_rows])
                _, final_val_t1 = _build_target_expr_and_val(target_1, None, w_opt)
                _, final_val_t2 = _build_target_expr_and_val(target_2, None, w_opt)
                
                pareto_results.append({
                    "target_2_value": float(final_val_t2),
                    "target_1_value": float(final_val_t1),
                    "weights": w_opt.tolist()
                })

        highlight_metric = obj_cfg.get("highlight_metric", "")
        best_idx = None
        highlight_message = ""
        
        if pareto_results and highlight_metric:
            if highlight_metric == "ratio":
                ratios = [p["target_1_value"] / (p["target_2_value"] + 1e-9) for p in pareto_results]
                best_idx = int(np.argmax(ratios))
                highlight_message = f"AI Insight: Optimal Ratio (Target 1 / Target 2) at Portfolio #{best_idx + 1}"
                
            elif highlight_metric == "knee_point":
                x_vals = np.array([p["target_2_value"] for p in pareto_results])
                y_vals = np.array([p["target_1_value"] for p in pareto_results])
                x1, y1 = x_vals[0], y_vals[0]
                x2, y2 = x_vals[-1], y_vals[-1]
                max_dist = -1
                
                for i, (x0, y0) in enumerate(zip(x_vals, y_vals)):
                    dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / (np.sqrt((x2 - x1)**2 + (y2 - y1)**2) + 1e-9)
                    if dist > max_dist:
                        max_dist = dist
                        best_idx = i
                highlight_message = f"AI Insight: Best geometric compromise (Knee Point) at Portfolio #{best_idx + 1}"
                
            elif highlight_metric == "min_target_2":
                best_idx = 0
                highlight_message = "AI Insight: Minimum Target 2 portfolio."
                
            elif highlight_metric == "max_target_1":
                best_idx = len(pareto_results) - 1
                highlight_message = "AI Insight: Maximum Target 1 portfolio."

        return {
            "success": True,
            "type": "pareto",
            "pareto_points": pareto_results,
            "target_1_name": target_1.get("target_name", "Target_1"),
            "target_2_name": target_2.get("target_name", "Target_2"),
            "best_portfolio_index": best_idx,
            "highlight_message": highlight_message
        }