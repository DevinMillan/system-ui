import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thesis: Weather Forecasting Dashboard", layout="wide")

RESULTS_DIR = "results"
IMAGE_DIR = "result_images"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_ALGORITHMS = ["ARIMA", "GRU", "RNN", "LSTM", "LINEAR REGRESSION"]


def load_all_results():
    all_data = []
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename), "r") as f:
                item = json.load(f)
                item["_source_file"] = filename
                all_data.append(item)
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


def slugify(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def build_candidate_names(result):
    base_name = os.path.splitext(result.get("_source_file", ""))[0]
    algorithm = slugify(result["algorithm"])
    city = slugify(result["city"])
    task = slugify(result["task"])
    result_type = slugify(result["type"])

    candidates = [
        base_name,
        f"{algorithm}_{city}_{task}",
        f"{algorithm}_{city}_{task}_{result_type}",
    ]
    return [name for name in candidates if name]


def find_result_images(result):
    if not os.path.isdir(IMAGE_DIR):
        return []

    candidate_names = build_candidate_names(result)
    exact_matches = []

    for name in candidate_names:
        for ext in IMAGE_EXTENSIONS:
            path = os.path.join(IMAGE_DIR, f"{name}{ext}")
            if os.path.exists(path):
                exact_matches.append(path)

    if exact_matches:
        return sorted(dict.fromkeys(exact_matches))

    partial_matches = []
    for file_name in os.listdir(IMAGE_DIR):
        full_path = os.path.join(IMAGE_DIR, file_name)
        stem, ext = os.path.splitext(file_name)
        if not os.path.isfile(full_path) or ext.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(candidate in stem.lower() for candidate in candidate_names):
            partial_matches.append(full_path)

    return sorted(partial_matches)


def with_placeholder_results(results):
    expanded = list(results)
    seen = {(item["algorithm"], item["city"], item["task"]) for item in results}
    combo_types = {}

    for item in results:
        combo_types[(item["city"], item["task"])] = item["type"]

    for (city, task), result_type in combo_types.items():
        for algorithm in SUPPORTED_ALGORITHMS:
            key = (algorithm, city, task)
            if key in seen:
                continue

            metrics = {"Accuracy": "N/A"}
            if result_type == "regression":
                metrics.update({"MAE": "N/A", "R2": "N/A", "MSE": "N/A"})

            expanded.append(
                {
                    "algorithm": algorithm,
                    "city": city,
                    "task": task,
                    "type": result_type,
                    "metrics": metrics,
                    "report": [],
                    "_source_file": "",
                    "_is_placeholder": True,
                }
            )

    return expanded


results = with_placeholder_results(load_all_results())

st.sidebar.title("Thesis Results")
if not results:
    st.sidebar.warning("No files found in /results folder.")
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

            cols = st.columns(len(curr["metrics"]))
            for i, (label, val) in enumerate(curr["metrics"].items()):
                cols[i].metric(label, f"{val}")

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
                    example_name = os.path.splitext(curr.get("_source_file", ""))[0]
                    st.info(
                        f"No image files found for this selection in the `{IMAGE_DIR}` folder. "
                        f"Add an image like `{example_name}.png`."
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
                    "Accuracy": r["metrics"].get("Accuracy", "N/A"),
                    "R2 Score": r["metrics"].get("R2", "N/A"),
                    "MAE": r["metrics"].get("MAE", "N/A"),
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
