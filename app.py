import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(page_title="Thesis: Weather Forecasting Dashboard", layout="wide")

# --- DATA LOADING ---
def load_all_results():
    all_data = []
    if not os.path.exists('results'):
        os.makedirs('results')
    for filename in os.listdir('results'):
        if filename.endswith('.json'):
            with open(os.path.join('results', filename), 'r') as f:
                all_data.append(json.load(f))
    return all_data


def disable_sidebar_select_typing():
    components.html(
        """
        <script>
        const disableTyping = () => {
            const parentDoc = window.parent.document;
            const selects = parentDoc.querySelectorAll('[data-testid="stSidebar"] [data-baseweb="select"] input');

            selects.forEach((input) => {
                input.setAttribute("readonly", "readonly");
                input.setAttribute("inputmode", "none");

                if (input.dataset.typingDisabled === "true") {
                    return;
                }

                const blockTyping = (event) => {
                    const allowedKeys = ["Tab", "Enter", "Escape", "ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Home", "End"];
                    if (!allowedKeys.includes(event.key)) {
                        event.preventDefault();
                    }
                };

                input.addEventListener("keydown", blockTyping);
                input.addEventListener("paste", (event) => event.preventDefault());
                input.addEventListener("drop", (event) => event.preventDefault());
                input.dataset.typingDisabled = "true";
            });
        };

        disableTyping();
        new MutationObserver(disableTyping).observe(window.parent.document.body, {
            childList: true,
            subtree: true,
        });
        </script>
        """,
        height=0,
        width=0,
    )

results = load_all_results()

# --- SIDEBAR ---
st.sidebar.title("📑 Thesis Results")
if not results:
    st.sidebar.warning("No files found in /results folder.")
else:
    mode = st.sidebar.radio("View Mode", ["Individual Analysis", "Comparison Leaderboard"])
    
    if mode == "Individual Analysis":
        algos = sorted(list(set(r['algorithm'] for r in results)))
        cities = sorted(list(set(r['city'] for r in results)))
        tasks = sorted(list(set(r['task'] for r in results)))

        sel_algo = st.sidebar.selectbox("Algorithm", algos)
        sel_city = st.sidebar.selectbox("Location", cities)
        sel_task = st.sidebar.selectbox("Weather Variable", tasks)
        disable_sidebar_select_typing()

        # Filter
        curr = next((r for r in results if r['algorithm'] == sel_algo 
                     and r['city'] == sel_city and r['task'] == sel_task), None)

        if curr:
            st.title(f"📊 {sel_algo}: {sel_task} in {sel_city}")
            
            # Metric Row
            cols = st.columns(len(curr['metrics']))
            for i, (label, val) in enumerate(curr['metrics'].items()):
                cols[i].metric(label, f"{val}")

            st.divider()

            if curr['type'] == "regression":
                st.subheader("Actual vs Predicted Trend")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=curr['actual'], name="Actual", line=dict(color='#4c72b0')))
                fig.add_trace(go.Scatter(y=curr['predicted'], name="Predicted", line=dict(color='#dd8452', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            
            else: # Classification
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader("Confusion Matrix")
                    fig_cm = px.imshow(curr['matrix'], text_auto=True, color_continuous_scale='Blues',
                                       x=['Light', 'Moderate', 'Heavy'], y=['Light', 'Moderate', 'Heavy'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                with c2:
                    st.subheader("Class Breakdown")
                    st.table(pd.DataFrame(curr['report']))
        else:
            st.error("Select a combination to view results.")

    else: # --- LEADERBOARD MODE ---
        st.title("🏆 Algorithm Performance Leaderboard")
        summary_list = []
        for r in results:
            summary_list.append({
                "City": r['city'],
                "Task": r['task'],
                "Accuracy": r['metrics'].get('Accuracy', 0),
                "R2 Score": r['metrics'].get('R2', 0),
                "MAE": r['metrics'].get('MAE', 0)
            })
        
        df_lead = pd.DataFrame(summary_list)
        st.dataframe(df_lead.sort_values(by="Accuracy", ascending=False), use_container_width=True)
        
        st.subheader("Task Performance Comparison")
        fig_bar = px.bar(df_lead, x="Task", y="Accuracy", color="City", barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True)
