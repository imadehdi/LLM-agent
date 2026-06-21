import pandas as pd
import numpy as np

# Fixer le seed pour la reproductibilité des scénarios
np.random.seed(42)

# 1. Onglet : Donnees_Actifs
df_actifs = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "JNJ", "PFE", "JPM", "GS"],
    "expected_return": [0.12, 0.11, 0.06, 0.05, 0.09, 0.10],
    "sector": ["Tech", "Tech", "Health", "Health", "Finance", "Finance"]
})

# 2. Onglet : Matrice_Covariance
cov_data = [
    [0.045, 0.035, 0.005, 0.004, 0.015, 0.020],
    [0.035, 0.050, 0.006, 0.005, 0.012, 0.018],
    [0.005, 0.006, 0.015, 0.010, 0.002, 0.003],
    [0.004, 0.005, 0.010, 0.020, 0.001, 0.002],
    [0.015, 0.012, 0.002, 0.001, 0.030, 0.025],
    [0.020, 0.018, 0.003, 0.002, 0.025, 0.040]
]
df_cov = pd.DataFrame(cov_data, columns=df_actifs["Ticker"])

# 3. Onglet : Matrice_Scenarios (60 jours de rendements historiques)
# La MAD s'applique sur des rendements d'actifs classiques.
rendements_historiques = np.random.normal(loc=0.0005, scale=0.02, size=(60, 6))
df_scenarios = pd.DataFrame(rendements_historiques, columns=df_actifs["Ticker"])

# Écriture du fichier Excel multi-onglets
with pd.ExcelWriter("portefeuille.xlsx", engine="xlsxwriter") as writer:
    df_actifs.to_excel(writer, sheet_name="Donnees_Actifs", index=False)
    df_cov.to_excel(writer, sheet_name="Matrice_Covariance", index=False)
    df_scenarios.to_excel(writer, sheet_name="Matrice_Scenarios", index=False)

print("🎯 Fichier 'portefeuille.xlsx' prêt pour le test MAD & Turnover !")