import streamlit as st
import pandas as pd
from streamlit_vertical_slider import vertical_slider
from utils import split_utils, files_utils

def app():
    """Streamlit App for Dataset Splitting and Filtering"""

    data = st.session_state.get("data", pd.DataFrame())
    meta = st.session_state.get("meta", {})

    if "selections" not in st.session_state:
        st.session_state.selections = [{}]  

    if "user_choices" not in st.session_state:
        st.session_state.user_choices = []

    def add_column():
        """Adds a new column selection box dynamically"""
        st.session_state.selections.append({})

    def clear_selections():
        """Resets filters and session state selections"""
        st.session_state.selections = [{}]  
        st.session_state.user_choices = []
        for key in list(st.session_state.keys()):
            if key.startswith("column_") or key.startswith("values_"):
                del st.session_state[key]  

    dataset_summary, dataset_summary_aftersplit, plot = st.columns([1, 1, 1]) 
    column_selection,value_selection, dataset_split = st.columns([1, 1, 1]) 

    current_filters = {}

    with column_selection:
        st.write("### Segment Selection")
        for idx in range(len(st.session_state.selections)):
            selected_column = st.selectbox(
                f"Column {idx + 1}:",
                data.columns,
                key=f"column_{idx}"
            )
        st.button("➕ Add Another Column", on_click=add_column)

    with value_selection:
        st.write("###")
        for idx in range(len(st.session_state.selections)):
            selected_column = st.session_state.get(f"column_{idx}", "")

            if selected_column:
                @st.cache_data
                def get_unique_values(column):
                    """Returns unique non-null values from the specified column as a list."""
                    return list(data[column].dropna().unique()) 

                unique_values = get_unique_values(selected_column)

                value_map = {
                    f"{val} - {label}" if (label := files_utils.get_label(meta, selected_column, val)) else str(val): val
                    for val in unique_values
                } if meta else {str(val): val for val in unique_values}

                selected_display_values = st.multiselect(
                    f"Values for {selected_column}:",
                    list(value_map.keys()),  
                    key=f"values_{idx}"
                )

                selected_values = [value_map[val] for val in selected_display_values]
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

    if current_filters != st.session_state.user_choices:
        st.session_state.user_choices = [current_filters] if current_filters else []

    filtered_df, remaining_df = split_utils.filter_dataframe(data, st.session_state.user_choices)

    segment_training_size = round(filtered_df.shape[0] * train_size_percentage / 100)
    total_training_size = segment_training_size + remaining_df.shape[0]
    holdout_size = filtered_df.shape[0] - segment_training_size

    with dataset_summary:
        st.markdown("### Dataset Summary")
        st.metric(label="Original Dataset Size", value=f"{data.shape[0]} rows")
        st.metric(label="Number of Columns", value=f"{data.shape[1]} rows")
        st.metric(label="Segment Size", value=f"{filtered_df.shape[0]} rows", delta=f"{round(filtered_df.shape[0]*100/data.shape[0], 2)}%")
    
    with dataset_summary_aftersplit:
        st.markdown("### Split Simulation")  
        st.metric(label="Segment Training Size", value=f"{segment_training_size} rows", delta=f"{round(segment_training_size/data.shape[0]*100, 2)}%")
        st.metric(label="Total Training Size", value=f"{total_training_size} rows", delta=f"{round(total_training_size/data.shape[0]*100, 2)}%")
        st.metric(label="Holdout Size", value=f"{holdout_size} rows", delta=f"{round(holdout_size/data.shape[0]*100, 2)}%")

    with plot:
        split_utils.plot_training_holdout(
            total_training_size=total_training_size, 
            holdout_size=holdout_size, 
            segment_training_size=segment_training_size
        )
