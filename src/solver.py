import casadi as ca
import numpy as np

def solve_optimization(builder, x0=None, discrete_vector=None) -> dict:
    """
    Assemble et résout le problème d'optimisation compilé dans le builder.
    Supporte nativement la bascule automatique vers BONMIN en cas de variables discrètes.
    """
    nlp = builder.build_nlp()
    
    lbx, ubx = builder.lbx, builder.ubx
    lbg, ubg = builder.lbg, builder.ubg
    
    # Configuration des options par défaut
    opts = {}
    
    # --- DÉTECTION ET ACTIVATION DE BONMIN ---
    if discrete_vector is not None and any(discrete_vector):
        solver_plugin = 'bonmin'
        opts['discrete'] = discrete_vector
        # Options de base pour stabiliser la recherche par arbre (Branch and Bound)
        opts['bonmin.algorithm'] = 'B-BB' 
        opts['bonmin.time_limit'] = 30.0  # Sécurité temps max 30s
    else:
        solver_plugin = 'ipopt'
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
    
    try:
        # Instanciation dynamique du solveur requis
        solver = ca.nlpsol('solver', solver_plugin, nlp, opts)
        
        # Exécution du calcul numérique
        if x0 is not None:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        else:
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            
        # Extraction du statut de convergence
        stats = solver.stats()
        status = stats.get("return_status", "Unknown")
        success = stats.get("success", False)
        
        # Récupération des valeurs optimales trouvées
        x_opt = np.array(sol['x']).flatten()
        f_opt = float(sol['f'])
        
        # CORRECTION : On utilise la propriété native de ton builder (n_rows) au lieu d'un len()
        n_assets = builder.n_rows
        
        # On ne renvoie à l'interface que la première section correspondant aux poids réels 'x'
        x_values_to_return = x_opt[:n_assets].tolist()
        
        return {
            "success": success,
            "status": status,
            "objective": f_opt,
            "x_values": x_values_to_return,
            "error": ""
        }
        
    except Exception as e:
        return {
            "success": False,
            "status": "Crash",
            "objective": 0.0,
            "x_values": [],
            "error": str(e)
        }