import os
import json
import numpy as np
import pandas as pd
import casadi as ca
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.agent_formulator import FormulationAgent
from src.agent_executor import OptimizationExecutor

st.set_page_config(
    page_title="AI Optimization Engine",
    page_icon="",
    layout="wide"
)

load_dotenv()

model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

st.title("Universal AI-Powered Optimization Engine")
st.write("Upload an Excel file, describe your constraints in natural language, and the AI will configure and solve your mathematical problem.")

# ============================================================
# ZONE 1 : CHARGEMENT DU FICHIER EXCEL MULTI-ONGLETS
# ============================================================
st.subheader("1. Upload your data")
uploaded_file = st.file_uploader("Select an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Analyse des onglets disponibles dans le fichier Excel
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names
    
    # Initialisation des structures de données
    matrix_inputs = {}
    
    # 1. Chargement des données linéaires (Obligatoire)
    if "Donnees_Actifs" in sheet_names:
        df_data = pd.read_excel(uploaded_file, sheet_name="Donnees_Actifs")
    else:
        # Fallback si l'utilisateur a un Excel simple à un seul onglet
        df_data = pd.read_excel(uploaded_file, sheet_name=0)
        
    n_rows = len(df_data)
    st.dataframe(df_data, use_container_width=True)
    st.info(f"File successfully loaded: **{n_rows} assets** detected.")

    # 2. Chargement optionnel de la covariance (fournie par l'user)
    if "Matrice_Covariance" in sheet_names:
        df_cov = pd.read_excel(uploaded_file, sheet_name="Matrice_Covariance")
        matrix_inputs["Sigma"] = df_cov.values.tolist()
        st.caption("Covariance matrix detected and loaded from Excel.")
        
    # 3. Chargement optionnel des scénarios historiques (fournis par l'user)
    if "Matrice_Scenarios" in sheet_names:
        df_scenarios = pd.read_excel(uploaded_file, sheet_name="Matrice_Scenarios")
        matrix_inputs["Scenarios"] = df_scenarios.values.tolist()
        st.caption(f"Scenario matrix detected ({len(df_scenarios)} periods) and loaded from Excel.")

    # Ajout du vecteur initial par défaut
    matrix_inputs["x0"] = [1.0 / n_rows] * n_rows

    # ============================================================
    # ZONE 2 : REQUÊTE ET ONTOLOGIE SÉMANTIQUE
    # ============================================================
    st.subheader("2. Define the optimization problem")
    
    col_req, col_onto = st.columns([2, 1])
    
    with col_req:
        user_request = st.text_area(
            "Your request in natural language:", 
            placeholder="Example: Minimize the maximum drawdown using the Scenarios matrix...",
            height=150
        )
        
    with col_onto:
        st.markdown("**Semantic guide for the AI (Optional)**")
        
        default_ontology = {
          "metric_zones": [
            {"name": "allocation", "description": "Distribution measures, weights or shares.", "examples": ["weight", "allocation"]}
          ],
          "objective_functions": [
            {"name": "minimize_worst_case", "description": "Minimiser le pire des scénarios ou le maximum drawdown.", "examples": ["maximum drawdown", "mdd", "worst-case", "drawdown max"]}
          ]
        }
        
        ontology_text = st.text_area("Ontology in JSON format:", value=json.dumps(default_ontology, indent=2), height=110)
        try:
            ontology_dict = json.loads(ontology_text)
        except:
            ontology_dict = default_ontology

    # ============================================================
    # ZONE 3 : BOUTON DE LANCEMENT ET CALCULS
    # ============================================================
    if st.button("Run Optimization", type="primary"):
        if not user_request.strip():
            st.warning("Please enter a natural language request before optimizing.")
            st.stop()
            
        available_columns = df_data.columns.tolist()
        valid_matrices = list(matrix_inputs.keys())

        # --- STEP A : APPEL DE L'AGENT 1 (LLM) ---
        with st.spinner("The AI is analyzing your Excel structure and translating your request..."):
            try:
                formulator = FormulationAgent(model=model_name, host=ollama_host, verbose=False)
                problem = formulator.formulate(
                    user_request=user_request,
                    valid_columns=available_columns,
                    valid_matrices=valid_matrices,
                    ontology_dict=ontology_dict
                )
                config_dict = problem.optimization_config.model_dump()
                
                with st.expander("Inspect generated JSON specification"):
                    st.json(problem.optimization_config.model_dump_json(indent=2))
                    
            except Exception as e:
                st.error(f"AI interpretation failed: {str(e)}")
                st.stop()

        # --- STEP B : APPEL DE L'EXÉCUTEUR (CASADI + IPOPT) ---
        with st.spinner("Algebraic compilation and numerical solving in progress..."):
            try:
                executor = OptimizationExecutor(verbose=False)
                result = executor.run(
                    optimization_config=config_dict,
                    matrix_inputs=matrix_inputs,
                    df_data=df_data
                )
            except Exception as e:
                st.error(f"Error during mathematical construction: {str(e)}")
                st.stop()

        # --- STEP C : RESTITUTION GRAPHIQUE ---
        if result["success"]:
            st.success("Optimal solution found successfully!")
            
            m1, m2 = st.columns(2)
            m1.metric("Solver Status", result["status"])
            m2.metric("Final Objective Value", f"{result['objective']:.6f}")

            df_results = df_data.copy()
            df_results["Optimal_Result_x"] = result["x_values"]
            df_results["Optimized_Percentage"] = df_results["Optimal_Result_x"] * 100

            tab_col, chart_col = st.columns([1, 1])

            with tab_col:
                st.subheader("Final Allocation")
                df_display = df_results.copy()
                df_display["Optimal_Result_x"] = df_display["Optimized_Percentage"].map("{:.2f}%".format)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            with chart_col:
                st.subheader("Decision Visualization")
                first_col = df_data.columns[0]
                fig = px.pie(
                    df_results, 
                    values="Optimal_Result_x", 
                    names=first_col,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.YlGnBu_r
                )
                fig.update_layout(margin=dict(t=15, b=15, l=15, r=15))
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error(f"The solver failed to find a solution. Status: {result['status']}. Error: {result['error']}")

else:
    st.info("Waiting for an Excel file. Upload a .xlsx file to start the interface.")