import streamlit as st
import pandas as pd
from streamlit_vertical_slider import vertical_slider
from utils import split_utils, files_utils
import random
import json
from concurrent.futures import ThreadPoolExecutor

def app():
    """Optimized Streamlit App for Dataset Splitting and Filtering"""
    data = st.session_state.get("data", pd.DataFrame())
    meta = st.session_state.get("meta", {})

    # Initialize session state
    for key, default in {
        "selections": [],
        "user_choices": [],
        "remove_baseline": True,
        "with_baseline": True,
        "bootstrap": False,
        "bootstrap_occurrences": 3
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    @st.cache_data
    def get_unique_values(column, df):
        """Returns unique non-null values from the specified column as a list."""
        return list(df[column].dropna().unique()) 

    def add_column():
        """Adds a new column selection box dynamically"""
        available_columns = [col for col in data.columns if col not in {s["column"] for s in st.session_state.selections}]
        if available_columns:
            st.session_state.selections.append({"column": "", "values": []})  # Default structure

    def clear_selections():
        """Clears all selections and filters"""
        st.session_state.selections = []  
        st.session_state.user_choices = []
        for key in list(st.session_state.keys()):
            if key.startswith("column_") or key.startswith("values_"):
                del st.session_state[key]  

    dataset_summary, plot = st.columns([4, 2.5]) 
    column_selection, value_selection, dataset_split, other_options = st.columns([2, 2, 1, 2]) 

    selected_columns = set()

    with column_selection:
        st.write("### Segment Selection")
        for idx in range(len(st.session_state.selections)):
            available_columns = [col for col in data.columns if col not in selected_columns]

            selected_column = st.selectbox(
                f"Column {idx + 1}:",
                available_columns,  
                key=f"column_{idx}",
            )
            if selected_column:
                st.session_state.selections[idx]["column"] = selected_column  
                selected_columns.add(selected_column)

        if len(selected_columns) < len(data.columns):
            st.button("➕ Add Another Column", on_click=add_column)

    with value_selection:
        st.write("### Value Selection")
        for idx in range(len(st.session_state.selections)):
            selected_column = st.session_state.selections[idx].get("column", "")

            if selected_column:
                unique_values = get_unique_values(selected_column, data)

                value_map = {
                    f"{val} - {label}" if (label := files_utils.get_label(meta, selected_column, val)) else str(val): val
                    for val in unique_values
                } if meta else {str(val): val for val in unique_values}

                selected_display_values = st.multiselect(
                    f"Values for {selected_column}:",
                    list(value_map.keys()),  
                    key=f"values_{idx}"
                )

                st.session_state.selections[idx]["values"] = [value_map[val] for val in selected_display_values]  

        st.button("❌ Clear All Filters", on_click=clear_selections)

    with dataset_split:
        st.write("### Settings")
        train_size_percentage = 100 - vertical_slider(
            label="Holdout Split",
            default_value=90,
            min_value=0,
            max_value=100,
            step=1,
            key="targeted_split_slider"
        )

    with other_options:
        st.write("### Options")

        # Assign unique keys to prevent duplicate element errors
        st.session_state["with_baseline"] = st.toggle(
            "Baseline", value=st.session_state["with_baseline"], key="with_baseline_toggle"
        )
        
        st.session_state["remove_baseline"] = st.toggle(
            "Remove Baseline from Holdout", 
            value=st.session_state["remove_baseline"], 
            key="remove_baseline_toggle", 
            disabled=not st.session_state["with_baseline"]
        )

        st.session_state["bootstrap"] = st.toggle(
            "Bootstrap", value=st.session_state["bootstrap"], key="bootstrap_toggle"
        )

        if st.session_state["bootstrap"]:
            st.session_state["bootstrap_occurrences"] = st.number_input(
                "Bootstrap Occurrences", 
                min_value=0, max_value=10, 
                value=st.session_state["bootstrap_occurrences"], 
                key="bootstrap_targeted_occurrences"
            )

    current_filters = {s["column"]: s["values"] for s in st.session_state.selections if s["values"]}
    st.session_state.user_choices = [current_filters] if current_filters else []

    filtered_df, remaining_df = split_utils.filter_dataframe(data, st.session_state.user_choices)

    segment_training_size = round(filtered_df.shape[0] * train_size_percentage / 100)
    total_training_size = segment_training_size + remaining_df.shape[0]
    holdout_size = filtered_df.shape[0] - segment_training_size if not st.session_state["remove_baseline"] else filtered_df.shape[0] - 2 * segment_training_size

    with dataset_summary:
        st.write("### Dataset Summary:")
        st.metric("Original Dataset", f"{data.shape[0]}")
        st.metric("Number of Columns", f"{data.shape[1]}")
        st.metric("Segment Size", f"{filtered_df.shape[0]} ({round(filtered_df.shape[0] * 100 / data.shape[0], 2)}%)")
        st.metric("Segment Train Size", f"{segment_training_size} ({round(segment_training_size * 100 / data.shape[0], 2)}%)")
        st.metric("Total Train Size", f"{total_training_size} ({round(total_training_size * 100 / data.shape[0], 2)}%)")
        st.metric("Holdout Size", f"{holdout_size} ({round(holdout_size * 100 / data.shape[0], 2)}%)")

        if st.session_state["with_baseline"]:
            baseline_size = segment_training_size * 2
            st.metric("Baseline Size", f"{baseline_size} ({round(baseline_size * 100 / data.shape[0], 2)}%)")

    with plot:
        split_utils.plot_training_holdout(total_training_size, holdout_size, segment_training_size)

    if st.button("Split Data", key="targeted_split_button", type="primary"):
        files_utils.empty_folder("outputs")

        random_states = [random.randint(1, 100) for _ in range(st.session_state["bootstrap_occurrences"])] if st.session_state["bootstrap"] else None

        split_results = split_utils.targeted_split(
            st.session_state["data"], 
            train_size=train_size_percentage / 100, 
            baseline=st.session_state["with_baseline"], 
            remove_baseline=st.session_state["remove_baseline"], 
            random_states=random_states,
            filters=st.session_state["selections"]
        )

        with ThreadPoolExecutor() as executor:
            for idx, (train_df, holdout_df, baseline_df) in enumerate(split_results):
                suffix = f"_batch_{idx+1}" if st.session_state["bootstrap"] else ""
                executor.submit(files_utils.save_file, train_df, meta, f"outputs/train_{total_training_size}{suffix}.csv")
                executor.submit(files_utils.save_file, holdout_df, meta, f"outputs/holdout_{holdout_size}{suffix}.csv")

                if st.session_state["with_baseline"]:
                    executor.submit(files_utils.save_file, baseline_df, meta, f"outputs/baseline_{baseline_size}{suffix}.csv")

        st.success("Data successfully split!")
