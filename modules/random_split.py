import streamlit as st
from utils import split_utils, files_utils
from streamlit_vertical_slider import vertical_slider
import random
import os

def app():

    data = st.session_state["data"]
    meta = st.session_state["meta"]
    with_baseline = True

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
            )

        train_size = int((train_size_percentage / 100) * total_size)
        holdout_size = total_size - train_size

        with col_stats:
            st.metric(label="Total Size", value=f"{total_size} rows")
            st.metric(label=f"Train Size ({train_size_percentage}%)", value=f"{train_size} rows")
            st.metric(label=f"Holdout Size ({100 - train_size_percentage}%)", value=f"{holdout_size} rows")
            if with_baseline:
                st.metric(label=f"Baseline Size ({train_size_percentage*2}%)", value=f"{train_size*2} rows")
        
        with_baseline = st.toggle("Baseline", value=True)
        if with_baseline:
            remove_baseline = st.toggle("Remove from holdout", value=True)
        boostrap = st.toggle("Boostrap", value=False)
        if boostrap:
            boostrap_occurences = st.number_input("Boostrap Occurences", min_value=0, max_value=10, value=3)


    with col_preview:
        st.write("### Preview:")
        st.dataframe(data)

    if st.button("Split Data"):
        files_utils.empty_folder("outputs")
        if not boostrap:
            train_df, holdout_df, baseline_df = split_utils.random_split(st.session_state["data"], train_size_percentage, baseline=with_baseline, remove_baseline=remove_baseline)
            files_utils.save_file(df=train_df, metadata=meta, file_path=f"outputs/train_{train_size}" + st.session_state["file_type"])
            files_utils.save_file(df=holdout_df, metadata=meta, file_path=f"outputs/holdout_{holdout_size}" + st.session_state["file_type"])
            if with_baseline:
                    files_utils.save_file(df=baseline_df, metadata=meta, file_path=f"outputs/baseline_{train_size*2}" + st.session_state["file_type"])

            st.success("Data has been successfully split!")
        else:
            random_states = [random.randint(1, 100) for _ in range(boostrap_occurences)]
            for occurence, random_state in enumerate(random_states):
                train_df, holdout_df, baseline_df = split_utils.random_split(st.session_state["data"], train_size_percentage, baseline=with_baseline, remove_baseline=remove_baseline, random_state=random_state)
                files_utils.save_file(df=train_df, metadata=meta, file_path=f"outputs/train_{train_size}_batch_{occurence+1}"  + st.session_state["file_type"])
                files_utils.save_file(df=holdout_df, metadata=meta, file_path=f"outputs/holdout_{holdout_size}_batch_{occurence+1}" + st.session_state["file_type"])
                if with_baseline:
                        files_utils.save_file(df=baseline_df, metadata=meta, file_path=f"outputs/baseline_{train_size*2}_batch_{occurence+1}" + st.session_state["file_type"])

                st.success("Data has been successfully split!")


