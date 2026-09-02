import casadi as ca
import numpy as np
import pandas as pd

def parse_casadi_bounds(constraint_dict):
    b_type = constraint_dict["bound_type"]
    v = constraint_dict.get("value")
    if b_type == "eq": return v, v
    elif b_type == "max": return -ca.inf, v 
    elif b_type == "min": return v, ca.inf
    elif b_type == "range":
        min_v = constraint_dict.get("min_value")
        max_v = constraint_dict.get("max_value")
        return (-ca.inf if min_v is None else min_v), (ca.inf if max_v is None else max_v)
    else: raise ValueError(f"Type de borne inconnu : {b_type}")

class CasadiProblemBuilder:
    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.vars = {} 
        self.objective = 0
        self.g_exprs, self.lbg, self.ubg = [], [], []
        self.lbx, self.ubx = [], [] 

    def create_variables(self, decision_variables_config: list):
        """Instancie les variables symboliques demandées par l'IA."""
        for var in decision_variables_config:
            name = var["name"]
            if var["size"] == "n_rows":
                self.vars[name] = ca.MX.sym(name, self.n_rows)
                self.lbx.extend([0.0] * self.n_rows) # Bornes par défaut (Positivité)
                self.ubx.extend([1.0] * self.n_rows) 
            elif var["size"] == "scalar":
                self.vars[name] = ca.MX.sym(name, 1)
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)

    def set_objective(self, expr, sense="min"):
        if sense == "max": self.objective = -expr
        else: self.objective = expr

    def add_constraint(self, expr, lb, ub):
        self.g_exprs.append(expr)
        self.lbg.append(lb)
        self.ubg.append(ub)

    def apply_smart_constraints(self, config_constraints: list, df_data: pd.DataFrame):
        col_map_case = {str(c).lower(): c for c in df_data.columns}

        for cstr in config_constraints:
            lb, ub = parse_casadi_bounds(cstr)
            attribute_raw = cstr.get("attribute", "")
            attribute_lower = attribute_raw.lower()
            
            var_name = cstr.get("applied_to", "x")
            if var_name not in self.vars: var_name = "x"
            current_var = self.vars[var_name]

            # 1. Contrainte de Somme
            if attribute_lower in ["sum_all", "sum", "somme", "total"]:
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue
                
            # 2. Contrainte Élémentaire (Plafond simple)
            if attribute_lower == "element":
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * self.n_rows)
                self.ubg.extend([ub] * self.n_rows)
                continue
            
            # 3. NOUVEAU : Minimum Buy-in (Big-M)
            if attribute_lower == "min_buy_in":
                # On suppose que 'b' existe dans self.vars
                if "b" not in self.vars:
                    print("Warning: min_buy_in demandé mais 'b' est absent.")
                    continue
                b_var = self.vars["b"]
                min_val = cstr.get("min_value", 0.0) # Le seuil L (ex: 0.03)
                # x - b <= 0  => x <= b
                # x - L*b >= 0 => x >= L*b
                for i in range(self.n_rows):
                    self.add_constraint(current_var[i] - b_var[i], -ca.inf, 0.0)
                    self.add_constraint(current_var[i] - min_val * b_var[i], 0.0, ca.inf)
                continue

            # 4. Vérification colonne
            if attribute_lower not in col_map_case:
                continue
            real_col_name = col_map_case[attribute_lower]

            # ... (laisser le reste de la fonction inchangé pour les autres contraintes)
            if pd.api.types.is_numeric_dtype(df_data[real_col_name]):
                exposure_vector = df_data[real_col_name].values.reshape(1, -1)
                exposure_expr = ca.mtimes(exposure_vector, current_var)
                self.add_constraint(exposure_expr, lb, ub)
            else:
                all_dummies = pd.get_dummies(df_data[real_col_name], dtype=float)
                val_map = {str(c).lower(): c for c in all_dummies.columns}
                targets = cstr.get("targets") or []
                targets_lower = [str(t).lower() for t in targets]
                
                if not targets or "all" in targets_lower:
                    exposure_matrix = all_dummies.values
                    n_c = exposure_matrix.shape[1]
                else:
                    valid_targets = [val_map[t_low] for t_low in targets_lower if t_low in val_map]
                    if not valid_targets: continue
                    exposure_matrix = all_dummies[valid_targets].values
                    n_c = len(valid_targets)
                    
                group_exposures = ca.mtimes(exposure_matrix.T, current_var)
                self.lbg.extend([lb] * n_c)
                self.ubg.extend([ub] * n_c)
                self.g_exprs.append(group_exposures)

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX()
        x = ca.vertcat(*list(self.vars.values()))
        return {"x": x, "f": self.objective, "g": g}