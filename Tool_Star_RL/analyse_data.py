import json
import pandas as pd
parquet_files = [
    "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet"
]
dataframes = []
for parquet_file in parquet_files:
    # read parquet files and cache
    dataframe = pd.read_parquet(parquet_file)
    dataframes.append(dataframe)
rl_data = pd.concat(dataframes)
