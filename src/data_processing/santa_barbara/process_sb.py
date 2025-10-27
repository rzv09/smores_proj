"""
This file processes the SB dataset 
TODO: add temperature as a covariate feature
"""
from data_utils import standardize_piece
import pandas as pd


def make_timeseries(data_path: str):
    """
    Returns a dataframe object with proper time indexes for July 2017
    To match Florida dataset timeframe

    Args:
        data_path (str): input data file location
    """
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime_UTC'], utc = True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/Los_Angeles')
    start_sb = '2017-07-11'
    end_sb = '2017-07-18'
    mask = (df['site'] == 'SBH') & (df['datetime'] >= start_sb) & (df['datetime'] <= end_sb)
    july_2017_sb = df.loc[mask]
    july_2017_sb.index = july_2017_sb['datetime'].dt.tz_localize(None)
    return july_2017_sb

def get_first_piece(df: pd.DataFrame):
    """Returns first chronological piece of the dataframe"""
    start_time = '2017-07-11 14:00:00'
    end_time = '2017-07-12 08:00:00'
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    return df.loc[mask]

def get_second_piece(df: pd.DataFrame):
    start_time = '2017-07-13 9:00:00'
    end_time = '2017-07-14 06:00:00'
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    return df.loc[mask]

def get_third_piece(df: pd.DataFrame):
    start_time = '2017-07-15 9:00:00'
    end_time = '2017-07-16 06:00:00'
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    return df.loc[mask]
    
def get_fourth_piece(df: pd.DataFrame):
    start_time = '2017-07-16 15:00:00'
    end_time = '2017-07-17 06:00:00'
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    return df.loc[mask]
    


