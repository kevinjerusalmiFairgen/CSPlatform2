import pandas as pd
from sklearn.model_selection import train_test_split

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
    

