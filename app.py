import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.agent_formulator import FormulationAgent
from src.agent_executor import OptimizationExecutor

st.set_page_config(
    page_title="AI Optimization Engine",
    page_icon=None,
    layout="wide"
)

load_dotenv()

model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ============================================================
# INITIALISATION DE LA MEMOIRE (CHAT HISTORY)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! Upload your data in the sidebar, and let me know how you want to optimize your portfolio."
        }
    ]

# ============================================================
# SIDEBAR : CONFIGURATION ET DONNEES
# ============================================================
with st.sidebar:
    st.header("Configuration")

    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Select an Excel file (.xlsx)", type=["xlsx"])

    df_data = None
    matrix_inputs = {}

    if uploaded_file is not None:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        if "Donnees_Actifs" in sheet_names:
            df_data = pd.read_excel(uploaded_file, sheet_name="Donnees_Actifs")
        else:
            df_data = pd.read_excel(uploaded_file, sheet_name=0)
            
        n_rows = len(df_data)
        st.success(f"{n_rows} assets loaded.")

        if "Matrice_Covariance" in sheet_names:
            df_cov = pd.read_excel(uploaded_file, sheet_name="Matrice_Covariance")
            # Force la sélection des colonnes numériques uniquement pour ignorer les titres/labels
            numeric_cov = df_cov.select_dtypes(include=[np.number])
            matrix_inputs["Sigma"] = numeric_cov.values.tolist()
            st.caption("Covariance matrix loaded.")
        
        if "Benchmark" in sheet_names:
            df_bench = pd.read_excel(uploaded_file, sheet_name="Benchmark")
            # On ne garde que les nombres pour éviter les crashs de typage
            numeric_bench = df_bench.select_dtypes(include=[np.number])
            # On convertit la colonne en une liste plate (1D)
            matrix_inputs["Benchmark"] = numeric_bench.values.flatten().tolist()
            st.caption("Benchmark weights loaded.")


        if "Matrice_Scenarios" in sheet_names:
            df_scenarios = pd.read_excel(uploaded_file, sheet_name="Matrice_Scenarios")
            # Force la sélection des colonnes numériques uniquement
            numeric_scenarios = df_scenarios.select_dtypes(include=[np.number])
            matrix_inputs["Scenarios"] = numeric_scenarios.values.tolist()
            st.caption(f"Scenarios loaded ({len(numeric_scenarios)} periods).")

        matrix_inputs["x0"] = [1.0 / n_rows] * n_rows
        
        with st.expander("Preview Data"):
            st.dataframe(df_data)

    st.markdown("---")
    st.subheader("2. Semantic Guide")
    default_ontology = {
      "metric_zones": [
        {
          "name": "allocation",
          "description": "Distribution measures, weights or shares in the portfolio.",
          "examples": ["weight", "active_weight", "allocation", "sum", "sum_all"]
        },
        {
          "name": "performance",
          "description": "Measures of expected return and yield.",
          "examples": ["expected_return", "realized_return", "return", "yield"]
        },
        {
          "name": "risk",
          "description": "Global measures of risk, volatility, and drawdowns.",
          "examples": ["volatility", "variance", "risk", "sigma", "covariance", "max_drawdown"]
        },
        {
          "name": "extra_financial",
          "description": "ESG scores and environmental impact metrics like carbon footprint.",
          "examples": ["esg_score", "esg", "carbon_footprint", "carbon_intensity", "emissions"]
        }
      ],
      "objective_functions": [
        {"name": "maximize_expected_return", "description": "Maximize the expected return of the portfolio."},
        {"name": "minimize_variance", "description": "Minimize the portfolio variance or volatility."},
        {"name": "maximize_sharpe_ratio", "description": "Maximize the risk-adjusted return."},
        {"name": "pareto_frontier", "description": "Generate a Pareto frontier or trade-off curve between two conflicting objectives.", "examples": ["pareto frontier", "trade-off curve", "efficient frontier", "compromise curve"]}
      ]
    }
    ontology_text = st.text_area("Ontology (JSON):", value=json.dumps(default_ontology, indent=2), height=150)
    try:
        ontology_dict = json.loads(ontology_text)
    except:
        ontology_dict = default_ontology
        
    if st.button("Clear Chat History"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# ============================================================
# MAIN AREA : INTERFACE DE CHAT
# ============================================================
st.title("Universal AI Optimization Engine")

# CORRECTION : Boucle enumerate pour garantir des identifiants (keys) uniques
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "json" in msg:
            with st.expander("Inspect generated JSON specification"):
                st.json(msg["json"])
        if "fig" in msg:
            st.plotly_chart(msg["fig"], use_container_width=True, key=f"history_fig_{i}")
        if "df" in msg:
            with st.expander("View detailed weights"):
                numeric_cols = msg["df"].select_dtypes(include=['number']).columns
                st.dataframe(msg["df"].style.format("{:.4f}", subset=numeric_cols), use_container_width=True)
        if "diag" in msg:
            with st.expander("Mathematical Diagnostics (Why did it fail?)", expanded=True):
                st.markdown(msg["diag"])

if prompt := st.chat_input("Ex: Minimize the portfolio risk using the Sigma matrix..."):
    
    if df_data is None:
        st.warning("Please upload an Excel file in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        available_columns = df_data.columns.tolist()
        valid_matrices = [k for k in matrix_inputs.keys() if k != "x0"]

        with st.status("Running Optimization Pipeline...", expanded=True) as status:
            
            st.write("Step 1: LLM Semantic Translation...")
            t0 = time.time()
            try:
                formulator = FormulationAgent(model=model_name, host=ollama_host, verbose=False)
                problem = formulator.formulate(
                    user_request=prompt,
                    valid_columns=available_columns,
                    valid_matrices=valid_matrices,
                    ontology_dict=ontology_dict
                )
                config_dict = problem.optimization_config.model_dump()
                config_json = problem.optimization_config.model_dump_json(indent=2)
                t1 = time.time()
                st.write(f"Formulation complete ({t1-t0:.2f}s)")
            except Exception as e:
                status.update(label="Pipeline failed at Step 1", state="error")
                st.error(f"AI interpretation failed: {str(e)}")
                st.stop()

            st.write("Step 2: CasADi Algebraic Compilation & Solving...")
            t2 = time.time()
            try:
                executor = OptimizationExecutor(verbose=False)
                result = executor.run(
                    optimization_config=config_dict,
                    matrix_inputs=matrix_inputs,
                    df_data=df_data
                )
                t3 = time.time()
                st.write(f"Numerical solving complete ({t3-t2:.2f}s)")
            except Exception as e:
                status.update(label="Pipeline failed at Step 2", state="error")
                st.error(f"Solver crashed: {str(e)}")
                st.stop()
                
            total_time = t3 - t0
            status.update(label=f"Pipeline completed in {total_time:.2f}s", state="complete", expanded=False)

        # --- TRAITEMENT DES RESULTATS ---
        if result.get("success"):
            fig = None
            df_export = None
            
            if result.get("type") == "pareto":
                assistant_text = f"Pareto Frontier generated with {len(result['pareto_points'])} optimal portfolios!"
                if result.get("highlight_message"):
                    assistant_text += f"\n\n{result['highlight_message']}"
                
                t1_name = result.get("target_1_name", "Target 1")
                t2_name = result.get("target_2_name", "Target 2")
                
                pareto_data = result["pareto_points"]
                x_vals = [p["target_2_value"] for p in pareto_data]
                y_vals = [p["target_1_value"] for p in pareto_data]
                
                df_pareto = pd.DataFrame({f"{t2_name}": x_vals, f"{t1_name}": y_vals})
                
                first_col = df_data.columns[0]
                asset_names = df_data[first_col].tolist()
                for i, asset in enumerate(asset_names):
                    df_pareto[f"Weight_{asset}"] = [p["weights"][i] * 100 for p in pareto_data]

                fig = px.line(
                    df_pareto, x=f"{t2_name}", y=f"{t1_name}", markers=True,
                    hover_data=[f"Weight_{asset}" for asset in asset_names],
                    title=f"Pareto Frontier ({t2_name} vs {t1_name})"
                )
                fig.update_traces(marker=dict(size=12, color="orange", line=dict(width=2, color="DarkSlateGrey")), line=dict(color="royalblue", width=3))

                best_idx = result.get("best_portfolio_index")
                if best_idx is not None:
                    fig.add_scatter(x=[x_vals[best_idx]], y=[y_vals[best_idx]], mode='markers', marker=dict(color='red', size=20, symbol='star', line=dict(width=2, color='black')), name='AI Highlight', hoverinfo='skip')
                df_export = df_pareto

            else:
                final_obj_val = result.get('objective', 0)
                direction = config_dict.get("objective", {}).get("direction", "min")
                if direction == "max":
                    final_obj_val = -final_obj_val
                    
                assistant_text = f"Optimal solution found! Objective value: **{final_obj_val:.6f}**"
                
                df_results = df_data.copy()
                df_results["Optimal_Result_x"] = result["x_values"]
                df_results["Optimized_Percentage"] = df_results["Optimal_Result_x"] * 100

                first_col = df_data.columns[0]
                fig = px.pie(df_results, values="Optimal_Result_x", names=first_col, hole=0.4, color_discrete_sequence=px.colors.sequential.YlGnBu_r, title="Optimal Asset Allocation")
                df_export = df_results

            st.markdown(assistant_text)
            with st.expander("Inspect generated JSON specification"):
                st.json(config_json)
            if fig:
                # CORRECTION : Utilisation de time.time() pour générer une clé d'affichage unique
                st.plotly_chart(fig, use_container_width=True, key=f"new_fig_{int(time.time())}")
            if df_export is not None:
                with st.expander("View detailed weights"):
                    numeric_cols = df_export.select_dtypes(include=['number']).columns
                    st.dataframe(df_export.style.format("{:.4f}", subset=numeric_cols), use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": assistant_text, "json": config_json, "fig": fig, "df": df_export})

        else:
            err_status = result.get('status', 'Unknown')
            err_msg = f"The solver failed to find a solution. Status: **{err_status}**"
            st.error(err_msg)
            
            diag_text = "When the solver returns `Infeasible_Problem_Detected`, it means your constraints mathematically contradict each other.\n\n"
            diag_text += "**Your Active Constraints:**\n```json\n" + json.dumps(config_dict.get("constraints", []), indent=2) + "\n```\n"
            
            constraints = config_dict.get("constraints", [])
            max_weight_c = next((c for c in constraints if c.get("attribute") == "element" and c.get("max_value") is not None), None)
            sum_c = next((c for c in constraints if c.get("attribute") in ["sum_all", "sum"] and c.get("bound_type") == "eq"), None)
            
            if max_weight_c and sum_c:
                max_w = max_weight_c.get("max_value")
                req_sum = sum_c.get("value")
                if max_w * n_rows < req_sum:
                    diag_text += f"\nMATH CONTRADICTION DETECTED: You require the total sum to be **{req_sum*100}%**, but you capped individual assets at **{max_w*100}%**. With only **{n_rows} assets**, the absolute maximum you can invest is **{max_w * n_rows * 100}%**. The problem is impossible to solve!"
            
            with st.expander("Mathematical Diagnostics (Why did it fail?)", expanded=True):
                st.markdown(diag_text)

            st.session_state.messages.append({"role": "assistant", "content": err_msg, "diag": diag_text})