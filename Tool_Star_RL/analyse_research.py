import jsonlines
import os
import json
import re
rollout_dir = "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_warm_up_0.95"
def get_search_questions(rollout_dir,result_dir,except_experiment=None,resume=False):
    print(rollout_dir)
    pattern = re.compile(r"<search>(.+?)</search>")
    search_questions = set()
    if resume:
        if os.path.exists(os.path.join(result_dir,"search_questions.json")):
            with open(os.path.join(result_dir,"search_questions.json"), "r",encoding="utf-8") as f:
                search_questions = set(json.load(f))
        else:
            search_questions = set()
    for root, dirs, files in os.walk(rollout_dir, topdown=False):
        if except_experiment and except_experiment in root:
            continue
        for file in files:
            if file.endswith("jsonl"):
                # print(file)
                with jsonlines.open(os.path.join(root, file)) as reader:
                    for line in reader:
                        match = pattern.findall(line["sequences_str"])
                        if match:
                            for i in range(2,len(match)):
                                clean_question = match[i].replace("search query:","").strip()
                                if len(clean_question) > 0:
                                    search_questions.add(clean_question)
    return search_questions
if __name__ == "__main__":
    rollout_dir = "/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints"
    result_dir = os.path.dirname(os.path.abspath(__file__))
    search_questions = get_search_questions(rollout_dir,result_dir,resume=True)
    print(len(search_questions))
    with open(os.path.join(result_dir,"search_questions.json"), "w",encoding="utf-8") as f:
        json.dump(list(search_questions), f, ensure_ascii=False, indent=4)
