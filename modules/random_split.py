import streamlit as st
from utils import split_utils, files_utils
from streamlit_vertical_slider import vertical_slider
import random
import pandas as pd

def app():
    data = st.session_state.get("data", pd.DataFrame())
    meta = st.session_state.get("meta", {})

    if "remove_baseline_from_holdout" not in st.session_state:
        st.session_state["remove_baseline_from_holdout"] = True
    # if "with_baseline" not in st.session_state:
    #     st.session_state["with_baseline"] = True
    if "boostrap" not in st.session_state:
        st.session_state["boostrap"] = False
    if "boostrap_occurences" not in st.session_state:
        st.session_state["boostrap_occurences"] = 3

    total_size = len(data)

    col_shape, col_preview = st.columns([3, 7]) 

    with col_shape:
        st.write("### Shape:")

        col_slider, col_stats = st.columns([3, 5]) 
        
        with col_slider:
            st.write("")
            st.write("")

            train_size_percentage = 100 - vertical_slider(
                label="Holdout Split",
                default_value=90,
                min_value=0,
                max_value=100,
                step=1,
                key="random split"
            )

        train_size = int((train_size_percentage / 100) * total_size)
        holdout_size = total_size - train_size #  if not st.session_state["remove_baseline_from_holdout"] else total_size - (2 * train_size)
        holdout_size_percentage = 100 - train_size_percentage * 2 # if st.session_state["remove_baseline_from_holdout"] else 100 - train_size_percentage

        with col_stats:
            st.metric(label="Total Size", value=f"{total_size} rows")
            st.metric(label=f"Train Size ({train_size_percentage}%)", value=f"{train_size} rows")
            st.metric(label=f"Holdout Size ({holdout_size_percentage}%)", value=f"{holdout_size} rows")
            # if st.session_state["with_baseline"]:
            #     st.metric(label=f"Baseline Size ({train_size_percentage*2}%)", value=f"{train_size*2} rows")

        # new_baseline_state = st.toggle("Baseline", value=st.session_state["with_baseline"])
        # if new_baseline_state != st.session_state["with_baseline"]:
        #     st.session_state["with_baseline"] = new_baseline_state
        #     if not new_baseline_state:
        #         st.session_state["remove_baseline_from_holdout"] = False  # Force remove_baseline to be False
        #     st.rerun()

        # if st.session_state["with_baseline"]:
        #     remove_baseline_from_holdout = st.toggle("Remove Baseline from holdout", value=st.session_state["remove_baseline_from_holdout"])
        #     if remove_baseline_from_holdout != st.session_state["remove_baseline_from_holdout"]:
        #         st.session_state["remove_baseline_from_holdout"] = remove_baseline_from_holdout
        #         st.rerun()

        new_boostrap_state = st.toggle("Boostrap", value=st.session_state["boostrap"])
        if new_boostrap_state != st.session_state["boostrap"]:
            st.session_state["boostrap"] = new_boostrap_state
            st.rerun()

        if st.session_state["boostrap"]:
            st.session_state["boostrap_occurences"] = st.number_input(
                "Boostrap Occurences", min_value=0, max_value=10, value=st.session_state["boostrap_occurences"]
            )

    with col_preview:
        st.write("### Preview:")
        st.write("")
        st.write("")
        st.dataframe(data)

    if st.button("Split Data"):
        files_utils.empty_folder("outputs")

        # Generate multiple random states if bootstrapping is enabled, otherwise set to None
        random_states = [random.randint(1, 100) for _ in range(st.session_state["boostrap_occurences"])] if st.session_state["boostrap"] else None

        # Perform dataset splitting
        split_results = split_utils.random_split(
            st.session_state["data"], 
            train_size=train_size_percentage / 100, 
            baseline=False  # st.session_state["with_baseline"], 
            remove_baseline=False  # st.session_state["remove_baseline_from_holdout"], 
            random_states=random_states
        )

        # Save results for each split (single or multiple depending on bootstrapping)
        for idx, (train_df, holdout_df, baseline_df) in enumerate(split_results):
            suffix = f"_batch_{idx+1}" if st.session_state["boostrap"] else ""
            
            files_utils.save_file(df=train_df, metadata=meta, file_path=f"outputs/train_{train_size}{suffix}" + "." +  st.session_state["file_type"])
            files_utils.save_file(df=holdout_df, metadata=meta, file_path=f"outputs/holdout_{holdout_size}{suffix}" + "." + st.session_state["file_type"])

            # if st.session_state["with_baseline"]:
            #     files_utils.save_file(df=baseline_df, metadata=meta, file_path=f"outputs/baseline_{train_size*2}{suffix}" + "." + st.session_state["file_type"])

        st.success("Data has been successfully split!")

