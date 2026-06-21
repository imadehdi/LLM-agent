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
        for cstr in config_constraints:
            lb, ub = parse_casadi_bounds(cstr)
            attribute = cstr["attribute"].lower()
            targets = cstr.get("targets") or []
            targets_lower = [str(t).lower() for t in targets]

            var_name = cstr.get("applied_to", "x")
            current_var = self.vars[var_name]

            # Contrainte sur la somme globale de la variable
            if attribute == "sum_all":
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue
                
            # 2. Contrainte sur CHAQUE ÉLÉMENT INDIVIDUEL
            if attribute == "element":
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * self.n_rows) # On duplique la borne pour chaque ligne
                self.ubg.extend([ub] * self.n_rows) # On duplique la borne pour chaque ligne
                continue

            if attribute not in df_data.columns:
                print(f"⚠️ Warning: L'attribut '{attribute}' n'existe pas.")
                continue

            # Multiplication par une colonne de données numériques (Moyenne pondérée linéaire)
            if pd.api.types.is_numeric_dtype(df_data[attribute]):
                exposure_vector = df_data[attribute].values.reshape(1, -1)
                exposure_expr = ca.mtimes(exposure_vector, current_var)
                self.add_constraint(exposure_expr, lb, ub)

            # Extraction via matrice binaire pour les variables textuelles/catégorielles
            else:
                all_dummies = pd.get_dummies(df_data[attribute], dtype=float)
                col_map = {str(c).lower(): c for c in all_dummies.columns}
                
                if not targets or "all" in targets_lower:
                    exposure_matrix = all_dummies.values
                    n_constraints = exposure_matrix.shape[1]
                else:
                    valid_targets = [col_map[t_low] for t_low in targets_lower if t_low in col_map]
                    if not valid_targets: continue
                    exposure_matrix = all_dummies[valid_targets].values
                    n_constraints = len(valid_targets)
                    
                group_exposures = ca.mtimes(exposure_matrix.T, current_var)
                self.lbg.extend([lb] * n_constraints)
                self.ubg.extend([ub] * n_constraints)
                self.g_exprs.append(group_exposures)

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX()
        x = ca.vertcat(*list(self.vars.values()))
        return {"x": x, "f": self.objective, "g": g}