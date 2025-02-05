import streamlit as st
import pandas as pd
from streamlit_vertical_slider import vertical_slider
from utils import split_utils, files_utils

def app():
    data = st.session_state.get("data", pd.DataFrame())
    meta = st.session_state.get("meta", {})

    if "selections" not in st.session_state or not st.session_state.selections:
        st.session_state.selections = [{}]

    if "user_choices" not in st.session_state:
        st.session_state.user_choices = []

    def add_column():
        st.session_state.selections.append({}) 

    def clear_selections():
        st.session_state.selections = [{}]  
        st.session_state.user_choices = []
        for key in list(st.session_state.keys()):
            if key.startswith("column_") or key.startswith("values_"):
                del st.session_state[key]  

    # === 🟢 Dataset Summary & Split (SIDE BY SIDE) ===
    dataset_summary, dataset_summary_aftersplit, plot = st.columns([1, 1, 1]) 

    # 🟡 Reset and Update Filters
    current_filters = {}

    column_selection, value_selection, dataset_split = st.columns([1, 1, 1]) 

    with column_selection:
        st.write("### Column Selection")
        for idx in range(len(st.session_state.selections)):
            selected_column = st.selectbox(
                f"Column {idx + 1}:",
                data.columns,
                key=f"column_{idx}"
            )
        st.button("➕ Add Another Column", on_click=add_column)


    with value_selection:
        st.write("### Value Selection")
        for idx in range(len(st.session_state.selections)):
            selected_column = st.session_state.get(f"column_{idx}", "")

            if selected_column:
                unique_values = [value.item() for value in list(data[selected_column].dropna().unique())]

                value_map = {}
                if meta:
                    value_map = {
                        f"{val} - {label}" if (label := files_utils.get_label(meta, selected_column, val)) else str(val): val
                        for val in unique_values
                    }
                    display_values = list(value_map.keys())  
                else:
                    display_values = unique_values  

                selected_display_values = st.multiselect(
                    f"Values for {selected_column}:",
                    display_values,
                    key=f"values_{idx}"
                )

                selected_values = [value_map[val] for val in selected_display_values] if meta else selected_display_values

                if selected_values:
                    current_filters[selected_column] = selected_values
        st.button("❌ Clear All Filters", on_click=clear_selections)


        with dataset_split:
            st.write("### Split")
            train_size_percentage = 100 - vertical_slider(
                label="Holdout Split",
                default_value=90,
                min_value=0,
                max_value=100,
                step=1,
                key="targeted_split"
            )

    # Apply filters dynamically
    st.session_state.user_choices = [current_filters] if current_filters else []

    # === 🟣 Recalculate filtered dataset dynamically ===
    filtered_df, remaining_df = split_utils.filter_dataframe(data, st.session_state.user_choices)

    segment_training_size = round(filtered_df.shape[0] * train_size_percentage / 100)
    total_training_size = segment_training_size + remaining_df.shape[0]
    holdout_size = filtered_df.shape[0] - segment_training_size

    # ✅ Update Dataset Summary After Filters & Split
    with dataset_summary:
        st.write("### Dataset Summary")
        st.write(f"**Original Dataset Size:** {data.shape[0]}")
        st.write(f"**Segment Size:** {filtered_df.shape[0]} rows ({round(filtered_df.shape[0]*100/data.shape[0], 2)}%)")
    with dataset_summary_aftersplit:  
        st.write("###")  
        st.write(f"**Segment Training Size:** {segment_training_size} rows ({round(segment_training_size/data.shape[0]*100, 2)}%)")
        st.write(f"**Total Training Size:** {total_training_size} rows ({round(total_training_size/data.shape[0]*100, 2)}%)")
        st.write(f"**Holdout Size:** {holdout_size} rows ({round(holdout_size/data.shape[0]*100, 2)}%)")
    with plot:
        split_utils.plot_training_holdout(
            total_training_size=total_training_size, 
            holdout_size=holdout_size, 
            segment_training_size=segment_training_size
        )



