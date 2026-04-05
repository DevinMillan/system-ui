import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thesis: Weather Forecasting Dashboard", layout="wide")

RESULTS_CSV = "results_index.csv"
IMAGE_DIRS = ["results_images", "result_images"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_ALGORITHMS = ["ARIMA", "GRU", "RNN", "LSTM", "LINEAR REGRESSION"]
METRIC_COLUMNS = [
    ("Accuracy", "accuracy"),
    ("MAE", "mae"),
    ("R2", "r2"),
    ("MSE", "mse"),
]


def load_all_results():
    if not os.path.exists(RESULTS_CSV):
        return []

    df = pd.read_csv(RESULTS_CSV, keep_default_na=False)
    results = []
    for _, row in df.iterrows():
        metrics = {}
        for label, column in METRIC_COLUMNS:
            value = row.get(column, "N/A")
            metrics[label] = value if value != "" else "N/A"

        report_raw = row.get("report_json", "[]") or "[]"
        try:
            report = json.loads(report_raw)
        except json.JSONDecodeError:
            report = []

        is_placeholder = str(row.get("is_placeholder", "False")).strip().lower() == "true"
        results.append(
            {
                "algorithm": row["algorithm"],
                "city": row["city"],
                "task": row["task"],
                "type": row["type"],
                "metrics": metrics,
                "report": report,
                "image_name": row.get("image_name", ""),
                "_is_placeholder": is_placeholder,
            }
        )
    return results


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


def build_candidate_names(result):
    image_name = result.get("image_name", "").strip().lower()
    candidates = [image_name] if image_name else []

    if result.get("algorithm") and result.get("city") and result.get("task"):
        fallback_name = "_".join(
            "".join(ch.lower() if ch.isalnum() else "_" for ch in part).strip("_")
            for part in [result["algorithm"], result["city"], result["task"]]
        )
        candidates.append(fallback_name)

    return [name for name in dict.fromkeys(candidates) if name]


def find_result_images(result):
    candidate_names = build_candidate_names(result)
    if not candidate_names:
        return []

    exact_matches = []
    for image_dir in IMAGE_DIRS:
        if not os.path.isdir(image_dir):
            continue
        for name in candidate_names:
            for ext in IMAGE_EXTENSIONS:
                path = os.path.join(image_dir, f"{name}{ext}")
                if os.path.exists(path):
                    exact_matches.append(path)

    if exact_matches:
        return sorted(dict.fromkeys(exact_matches))

    partial_matches = []
    for image_dir in IMAGE_DIRS:
        if not os.path.isdir(image_dir):
            continue
        for file_name in os.listdir(image_dir):
            full_path = os.path.join(image_dir, file_name)
            stem, ext = os.path.splitext(file_name)
            if not os.path.isfile(full_path) or ext.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(candidate in stem.lower() for candidate in candidate_names):
                partial_matches.append(full_path)

    return sorted(dict.fromkeys(partial_matches))


def format_metric(value):
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "N/A"
    return str(value)


results = load_all_results()

st.sidebar.title("Thesis Results")
if not results:
    st.sidebar.warning(f"No data found in `{RESULTS_CSV}`.")
else:
    mode = st.sidebar.radio("View Mode", ["Individual Analysis", "Comparison Leaderboard"])

    if mode == "Individual Analysis":
        algos = [algo for algo in SUPPORTED_ALGORITHMS if any(r["algorithm"] == algo for r in results)]
        cities = sorted({r["city"] for r in results})
        tasks = sorted({r["task"] for r in results})

        sel_algo = st.sidebar.selectbox("Algorithm", algos)
        sel_city = st.sidebar.selectbox("Location", cities)
        sel_task = st.sidebar.selectbox("Weather Variable", tasks)
        disable_sidebar_select_typing()

        curr = next(
            (
                r
                for r in results
                if r["algorithm"] == sel_algo and r["city"] == sel_city and r["task"] == sel_task
            ),
            None,
        )

        if curr:
            st.title(f"{sel_algo}: {sel_task} in {sel_city}")

            metric_items = list(curr["metrics"].items())
            cols = st.columns(len(metric_items))
            for i, (label, val) in enumerate(metric_items):
                cols[i].metric(label, format_metric(val))

            st.divider()
            result_images = find_result_images(curr)

            if result_images:
                st.subheader("Result Visuals")
                for image_path in result_images:
                    image_label = os.path.splitext(os.path.basename(image_path))[0].replace("_", " ").title()
                    st.image(image_path, caption=image_label, use_container_width=True)
            else:
                if curr.get("_is_placeholder"):
                    st.info("This algorithm is currently a placeholder for this city and weather variable. Metrics and visuals are not available yet.")
                else:
                    st.info(
                        "No image files found for this selection in the `results_images` folder. "
                        f"Add an image like `{curr.get('image_name', '')}.png`."
                    )

            if curr["type"] == "classification" and curr.get("report"):
                st.subheader("Class Breakdown")
                st.table(pd.DataFrame(curr["report"]))
        else:
            st.error("Select a combination to view results.")

    else:
        st.title("Algorithm Performance Leaderboard")
        summary_list = []
        for r in results:
            summary_list.append(
                {
                    "Algorithm": r["algorithm"],
                    "City": r["city"],
                    "Task": r["task"],
                    "Accuracy": format_metric(r["metrics"].get("Accuracy", "N/A")),
                    "R2 Score": format_metric(r["metrics"].get("R2", "N/A")),
                    "MAE": format_metric(r["metrics"].get("MAE", "N/A")),
                }
            )

        df_lead = pd.DataFrame(summary_list)
        numeric_accuracy = pd.to_numeric(df_lead["Accuracy"], errors="coerce")
        st.dataframe(
            df_lead.iloc[numeric_accuracy.sort_values(ascending=False, na_position="last").index],
            use_container_width=True,
        )

        st.subheader("Task Performance Comparison")
        df_chart = df_lead.copy()
        df_chart["Accuracy"] = pd.to_numeric(df_chart["Accuracy"], errors="coerce")
        df_chart = df_chart.dropna(subset=["Accuracy"])
        if not df_chart.empty:
            fig_bar = px.bar(df_chart, x="Task", y="Accuracy", color="Algorithm", barmode="group", facet_col="City")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No numeric leaderboard data is available yet.")
