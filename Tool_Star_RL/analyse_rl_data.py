import json
import pandas as pd
import openai
from math_verify import parse, verify
from dateutil import parser
def has_digit(s):
    for ch in s:
        if ch.isdigit():  # 检查字符是否是数字
            return True
    return False
def is_date(s):
    try:
        parser.parse(s, fuzzy=False)   # 严格解析
        return True
    except (ValueError, OverflowError):
        return False
parquet_files = [
    "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet"
]
dataframes = []
for parquet_file in parquet_files:
    # read parquet files and cache
    dataframe = pd.read_parquet(parquet_file)
    dataframes.append(dataframe)
rl_data = pd.concat(dataframes)
math_item=0
qa_item=0
for idx in range(len(rl_data)):
    data_item = rl_data.iloc[idx].to_dict()
    # print(data_item.keys())
    answer = data_item["reward_model"]["ground_truth"]
    question = data_item["question"]
    # if len(answer)==1:
    #     print(answer)
    # if data_item["ability"]=="math":
    if has_digit(answer):
        if is_date(answer):
            qa_item+=1
        else:
            math_item+=1
            if data_item["ability"]!="math":
                print(answer)
                pass
    else:
        qa_item+=1
        if data_item["ability"]!="qa":
            print(answer)
            pass
    # if data_item["ability"]=="qa":
        # qa_item+=1
    # if "A." in question:
    #     print(question)
    #     print(answer)
        
print(f"math_item:{math_item}")
print(f"qa_item:{qa_item}")