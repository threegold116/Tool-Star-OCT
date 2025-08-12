import jsonlines
import os
import json
import re
rollout_dir = "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/"
def get_search_questions(rollout_dir,result_dir):
    print(rollout_dir)
    pattern = re.compile(r"<search>(.+?)</search>")
    search_questions = set()
    # if os.path.exists(os.path.join(result_dir,"math_search_questions2.json")):
    #     with open(os.path.join(result_dir,"math_search_questions2.json"), "r",encoding="utf-8") as f:
    #         search_questions = set(json.load(f))
    # else:
    #     search_questions = set()
    for root, dirs, files in os.walk(rollout_dir, topdown=False):
        for file in files:
            if file.endswith("jsonl"):
                print(file)
                with jsonlines.open(os.path.join(root, file)) as reader:
                    for line in reader:
                        if not line["is_python"] or not line["is_search"]:
                            continue
                        # print(line["reason"])
                        print(line["is_python"],line["is_search"])
                        match = pattern.findall(line["sequences_str"])
                        if match:
                            for i in range(2,len(match)):
                                clean_question = match[i].replace("search query:","").strip()
                                if len(clean_question) > 0:
                                    search_questions.add((line["sequences_str"].split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip(),clean_question))
    return search_questions
rollout_dir = "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints/"
result_dir = os.path.dirname(os.path.abspath(__file__))
search_questions = get_search_questions(rollout_dir,result_dir)
print(len(search_questions))
with open(os.path.join(result_dir,"math_search_questions3.json"), "w",encoding="utf-8") as f:
    json.dump(list(search_questions), f, ensure_ascii=False, indent=4)
