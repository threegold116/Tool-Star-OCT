import json
import pandas as pd
import openai
import re
import asyncio
import os
from analyse_llm_serve import llm_evaluate_equivalence_batch

#1. 判断可能无法verify的问题
prompt_template='''
You are a data labeling assistant. Determine whether the given QA sample can be automatically evaluated.
If the sample can be automatically evaluated, output "Yes". If the sample cannot be automatically evaluated, output "No".

Decision Rules
Mark "yes" if any of the following is true:
1.The answer is a precise number, date, or measurable quantity, and the question is objective and verifiable by calculation (math_verify).
2.The answer is a short text (e.g., "Paris" or "Alice Yang").

Mark "no" if any of the following is true:
1.The answer is long and explanatory.
2.The question is multiple-choice.
Respond only with "yes" if it meets the criteria, or "no" if it does not.

Question: What is 7 * 8?
Labeled Answer: 56
Output: yes

Question: Who wrote 'Pride and Prejudice'?
Labeled Answer: Jane Austen
Output: yes

Question: Explain why the sky is blue.
Labeled Answer: The sky appears blue because...
Output: no

Question: {q}
Labeled Answer: {a}
Output:
'''
prompt_template1='''
Judge whether the question is a multi-choice question.

Input:
Question: {q}
Labeled Answer: {a}

Output:
Respond only with "yes" or "no" without any other text.
'''
prompt_template='''
Decision Rules:
Output "yes" if the sample can be automatically graded using exact string match, token-F1, or math verification. Output "no" if it cannot.

Not auto-gradable conditions:
1. The answer is long or explanatory (contains reasoning, justification, or more than 10 words).
2. The answer has multiple equally correct forms (e.g., descriptive text, translations, or different valid numeric formats that are not strictly equivalent).
3. The question is open-ended, discussion-based, or subjective (requires personal opinion or judgment).
4. The question lacks context or time constraints, making the answer non-unique.
5. The answer is overly broad or general.
6. The question is ambiguous or can be interpreted in multiple ways.
7. The question involves future predictions, hypothetical scenarios, or speculative answers.

Output format:
Output only "yes" or "no" in \\boxed{{}}, include explanations.

Question: Explain why the sky is blue.
Labeled Answer: The sky appears blue because...
Output: **Judge**: \\boxed{{no}}. **Explanation**: The answer is long and explanatory.

Question: {q}
Labeled Answer: {a}
Output:
'''


if __name__ == "__main__":
    # 1.设置环境变量，便于llm_server使用
    os.environ["ALIYUN_MODEL_NAME"]="Qwen3-30B-A3B-Instruct-2507"

    # 2.读取parquet文件
    parquet_files = [
        "/home/sxjiang/myproject/agent/Tool-Star-OCT/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet"
    ]
    dataframes = []
    for parquet_file in parquet_files:
        # read parquet files and cache
        dataframe = pd.read_parquet(parquet_file)
        dataframes.append(dataframe)
    rl_data = pd.concat(dataframes)
    math_item=0
    qa_item=0
    items=[]
    question_list = []

    # 2.获取类别,转为字典
    for idx in range(len(rl_data)):
        data_item = rl_data.iloc[idx].to_dict()
        # print(data_item.keys())
        answer = data_item["reward_model"]["ground_truth"]
        question = data_item["question"]
        question_list.append(question)
        # if len(answer)==1:
        #     print(answer)
        if data_item["ability"]=="math":
            math_item+=1
        if data_item["ability"]=="qa":
            qa_item+=1
        items.append(data_item)
    print(f"total items: {len(items)}, total unique questions: {len(set(question_list))}")

    # 3.构造prompt列表，并去重复
    prompt_list = []
    items = []
    for idx in range(len(rl_data)):
        data_item = rl_data.iloc[idx].to_dict()
        # print(data_item.keys())
        answer = data_item["reward_model"]["ground_truth"]
        question = data_item["question"]
        # if len(answer)==1:
        #     print(answer)
        prompt = prompt_template.format(q=question,a=answer)
        if prompt in prompt_list:
            print("Duplicate")
        else:
            prompt_list.append(prompt)
            items.append(data_item)
    
    # 4.推理得到结果
    llm_results = asyncio.run(llm_evaluate_equivalence_batch(
                prompts=prompt_list,
                extract_answer=False
            ))
    wrong_questions = []

    # 5.保存至文件    
    for idx in range(len(llm_results)):
        items[idx]["llm_results"] = llm_results[idx]
        items[idx]["llm_prompt"] = prompt_template.format(q=items[idx]["question"],a=items[idx]["reward_model"]["ground_truth"])
        pattern = r'\\boxed\{(.*?)\}'
        match = re.search(pattern, llm_results[idx])
        if match:
            extracted_answer = match.group(1).strip().lower()
        if extracted_answer.lower()== "no":
            wrong_questions.append(items[idx])

    # 6.保存为parquet文件
    new_items = []
    for idx in range(min(len(items),len(llm_results))):
        pattern = r'\\boxed\{(.*?)\}'
        match = re.search(pattern, llm_results[idx])
        if match:
            extracted_answer = match.group(1).strip().lower()
        if extracted_answer.lower() == "yes":
            new_items.append(items[idx])
    new_df = pd.DataFrame(new_items)
    new_df.to_parquet("rl_data_auto_verify_qwen3_32b.parquet",index=False)

    with open("llm_results_qwen3_32b.json","w") as f:
        json.dump(items,f,indent=4,ensure_ascii=False)

    with open("llm_results_wrong_qwen3_32b.json","w") as f:
        json.dump(wrong_questions,f,indent=4,ensure_ascii=False)
