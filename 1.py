import pandas as pd
df4 = pd.read_csv(r"D:\\2min-resample\MetaDataSeparation\MetaData Filtered\combined_consumption_per.csv",index_col=0)
df4 = df4.rename(columns=lambda x: f"{x}.parquet")
df4.to_csv(r"D:\\2min-resample\MetaDataSeparation\MetaData Filtered\combined_consumption_per1.csv")