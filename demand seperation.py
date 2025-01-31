import os
import pandas as pd
import glob
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import datetime

# Constants
Cwater = 4.2  # kJ/(kg*K)
ita = 0.9  # efficiency

# Load MaxPowList once
MaxPowList = pd.read_csv(r'D:\2min-resample\MetaDataSeparation\MetaData Filtered\PowerMaxT.csv', index_col=0)


def mainsw_temp(DateTime):
    day_of_year = DateTime.dayofyear
    ave = 11
    fluc = 5 * np.sin((day_of_year - 141) * 2 * np.pi / 365)
    return ave + fluc


def extract_and_resample_and_calculate(file, resample_freq='30min', max_pow_list=None):
    df = pq.read_table(file).to_pandas()
    df.index = pd.to_datetime(df.index)

    # Apply mains water temperature function
    df['MainsWT'] = df.index.to_series().apply(mainsw_temp)

    # Calculate DHW power
    df['DHWpow'] = df['HwFlow[L/min](float32)'] * (df['HwTOutlet[degC](float32)'] - df['MainsWT']) * df[
        'ChActive'] * Cwater / (ita * 60)

    # Calculate actual power
    max_power = max_pow_list[os.path.basename(file)].iloc[0]
    df['ActPow[%](float32)'] = df['ActPow[%](float32)'] * df['ChActive'] * max_power

    # Resample data
    resampled_gpow = df['ActPow[%](float32)'].resample(resample_freq).mean() / 200
    resampled_DHWpow = df['DHWpow'].resample(resample_freq).mean() / 2

    # Combine resampled data and calculate CH power
    resampled_df = pd.concat([resampled_gpow, resampled_DHWpow], axis=1)
    resampled_df['CHpow'] = resampled_df['ActPow[%](float32)'] - resampled_df['DHWpow']

    return resampled_df['CHpow']


def create_common_time_index(file_list, resample_freq='30min'):
    min_date, max_date = None, None

    for file in tqdm(file_list, desc="Getting Date Stamps"):
        df = pq.read_table(file).to_pandas()
        df.index = pd.to_datetime(df.index)
        current_min_date, current_max_date = df.index.min(), df.index.max()
        if min_date is None or current_min_date < min_date:
            min_date = current_min_date
        if max_date is None or current_max_date > max_date:
            max_date = current_max_date

    common_time_index = pd.date_range(start=min_date, end=max_date, freq=resample_freq)
    index_df = pd.DataFrame(common_time_index, columns=['datetime'])
    index_df.to_csv(r'D:\2min-resample\MetaDataSeparation\MetaData Filtered\time_index.csv', index=False)

    return common_time_index


def main():
    out_dir = r'D:\2min-resample\MetaDataSeparation\MetaData Filtered\load duration model_WODHW'
    directories = [
        r'D:\2min-resample\MetaDataSeparation\MetaData Filtered\With_RetT\processed_files_Version3\\',
        r'D:\\2min-resample\MetaDataSeparation\MetaData Filtered\WO_RetT\processed_files_Version3\\'
    ]

    # Collect file paths from all directories
    file_list = []
    for directory in directories:
        file_list.extend(glob.glob(os.path.join(directory, '*.parquet')))

    # Read common time index from file
    index_df = pd.read_csv(r'D:\2min-resample\MetaDataSeparation\MetaData Filtered\time_index.csv')
    common_time_index = pd.to_datetime(index_df['datetime'])

    # Dictionary to hold all resampled data
    data_dict = {}

    # Process each file
    for file in tqdm(file_list, desc="Processing files"):
        resampled_series = extract_and_resample_and_calculate(file, max_pow_list=MaxPowList)
        resampled_series = resampled_series.reindex(common_time_index)
        data_dict[os.path.basename(file)] = resampled_series

    # Combine all series into a single DataFrame
    combined_df = pd.DataFrame(data_dict, index=common_time_index)

    # Save the combined DataFrame to a CSV file
    output_csv_path = os.path.join(out_dir, 'combined_consumption_WODHW.csv')
    combined_df.to_csv(output_csv_path, index=True)


if __name__ == "__main__":
    main()
