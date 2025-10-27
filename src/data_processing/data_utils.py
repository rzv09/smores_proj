"""
Shared utilities for processing TS data
"""
def standardize_piece(df_piece):
    """
    Standardizes a chronological piece of the dataframe
      by subtracting the mean and dividing by std

    Args:
        df_piece (pandas dataframe)
    """

    # column_indices = {name: i for i, name in enumerate(df_piece.columns)}
    mean = df_piece.mean()
    std = df_piece.std()
    return (df_piece - mean) / std