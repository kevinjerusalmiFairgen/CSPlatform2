import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def random_split(df: pd.DataFrame, train_size: float = 0.1, baseline: bool = True, remove_baseline: bool = True, random_state=42):
    if not baseline:
        # If no baseline is required, proceed with normal train-test split
        train_df, holdout_df = train_test_split(df, train_size=train_size, random_state=random_state)
        return train_df, holdout_df, None
    
    # Create a baseline of size 2 * train_size
    baseline_size = 2 * train_size
    baseline_df, holdout_df = train_test_split(df, train_size=baseline_size, random_state=random_state)
    
    # Half of the baseline is assigned to train
    train_df, _ = train_test_split(baseline_df, train_size=0.5, random_state=random_state)
    
    if remove_baseline:
        # Remove all baseline data from holdout
        holdout_df = holdout_df.loc[~holdout_df.index.isin(baseline_df.index)]
    else:
        # Only remove train indexes from holdout
        holdout_df = holdout_df.loc[~holdout_df.index.isin(train_df.index)]
    
    return train_df, holdout_df, baseline_df
    

def filter_dataframe(data: pd.DataFrame, filters: list):
    """
    Filters a DataFrame based on a list of column-value mappings from JSON.
    """
    if not filters:
        return data, pd.DataFrame()  
    filtered_df = data.copy()
    
    for filter_dict in filters:
        for column, values in filter_dict.items():

            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    remaining_df = data.drop(filtered_df.index)

    return filtered_df, remaining_df


def plot_training_holdout(total_training_size, holdout_size, segment_training_size):
    print(total_training_size, holdout_size, segment_training_size)
    fig, ax = plt.subplots(figsize=(5, 6))

    rest_training_size = total_training_size - segment_training_size
    sizes = [segment_training_size, rest_training_size, holdout_size]
    labels = ["Training Segment", "Training", "Holdout"]
    colors = ["#f34b4c", "#f8b7ba", "#f0f2f6"]
    # Fix the size of the pie chart explicitly
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, startangle=140, 
        wedgeprops={'edgecolor': 'white'}, autopct='%1.1f%%',
        textprops={'fontsize': 12, 'color': 'black'},  
        radius=0.8,  
        pctdistance=0.70  # Move percentages inside the slices
    )

    for text in texts:
        text.set_color("black")

    ax.add_artist(plt.Circle((0, 0), 0.4, fc='white'))  # Fix the inner circle size

    # Ensure the aspect ratio is equal to prevent distortion
    ax.set_aspect('equal')

    # Move the plot higher to reduce space between title and chart
    ax.set_position([0.1, 0.35, 0.8, 0.2])  # [left, bottom, width, height]

    # Set the title
    #ax.set_title("Training vs Holdout", fontsize=14, fontweight='bold', color="black", pad=-5)

    # Move legend below the chart and fix layout
    fig.legend(
        wedges, labels, loc="lower center", ncol=3, fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, 0.1)  # Moves legend closer to the chart
    )

    fig.tight_layout() 
    st.pyplot(fig, clear_figure=True)






