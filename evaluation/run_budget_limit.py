from ast import arg, parse
from dis import Instruction
import enum
import json 
from math import fabs
from operator import concat
import random
import math

# from re_search import last_boxed_only_string
# from test import batch_search
# from test import batch_search
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import argparse
import re
import time
import datetime
from transformers import AutoTokenizer
from typing import List, Dict, Optional, final, Union
import requests
import os
from python_executor import PythonExecutor
from tools.web_search_main import deep_search, search_cache, search_cache_file
from tools.debug_code import debug_code_function
from tools.rollback_code import rollback
from tools.refine_code import refine

import re

from utils import *


budgets = [(4, 3),
(7, 5),
(3, 4),
(3, 7),
(5, 4),
(4, 6),
(5, 7),
(7, 6),
(5, 6),
(6, 3)]



class Inference():
    def __init__(self, model_config, params_config, task, dataset_name, output_path, batch_size=4, counts=100, prompt_type='code_search', use_debug=False, use_rollback=False, use_refiner=False, max_response_length=4096, max_calling_times=3, max_tool_budget=30, python_budget=-1, search_budget=-1, search_engine="bing",resume_evaluate=True, max_obs_length=1024,all_wiki=0,budget_idx=-1):
        self.task = task
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.resume_evaluate = resume_evaluate
        self.counts = counts
        self.questions = []
        self.answers = []
        self.budget_idx = budget_idx
        print(f"budget_idx: {self.budget_idx}")
        if self.budget_idx >=0:
            print(f"budget_idx: {self.budget_idx}")
            print(f"new python_budget: {budgets[self.budget_idx][0]}, new search_budget: {budgets[self.budget_idx][1]}")
            self.python_budget, self.search_budget = budgets[self.budget_idx]
        if resume_evaluate and self.resume_data_check():
            print("all evaluated, exit!")
            exit()
        self.model,self.tokenizer = self.load_model(model_config)
        self.params_config = SamplingParams(**params_config)
        self.prompt_type = prompt_type
        self.use_debug = use_debug
        self.use_rollback = use_rollback
        self.use_refiner = use_refiner
        self.prompt_template = ''
        self.python_budget = python_budget
        self.search_budget = search_budget
        self.max_python_times = 3000
        self.max_search_times = 3000
        self.max_calling_times = max_calling_times
        self.max_tool_budget = max_tool_budget
        self.max_debug_times = 0
        self.max_refine_times = 0
        self.max_rollback_times = 0
        self.max_response_length = max_response_length
        self.max_obs_length = max_obs_length
        self.search_engine = search_engine
        self.executor = PythonExecutor(get_answer_from_stdout=True)
        self.all_wiki = all_wiki # -1 represent all of the web search,0 represent half web search half wiki search,1 represent all of the wiki
        if self.prompt_type == 'code_search':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
        elif self.prompt_type == 'code_search_multi_tool':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
You can use both tools in conjunction to enhance your problem-solving capabilities. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
        elif self.prompt_type == 'no-tool-inference':
            self.prompt_template = """
You are a helpful assistant that can solve questions step by step. \
Given a question, you need to think through the reasoning process and provide a clear and concise answer. \
The reasoning process and answer should be enclosed within <think> </think> \
and <answer> </answer> tags respectively. For example, \
<think> This is the reasoning process. </think> \
<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>.
"""
        elif self.prompt_type == 'code_search_autotir':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of tools like Wikipedia search and Python code execution. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, You may invoke the Wikipedia search tool for factual information or use Python code execution for calculation when needed. The reasoning process is enclosed within <think> </think>, and the answer is enclosed within <answer> </answer> tags. If Wikipedia search is used, the search query and result are enclosed in <search> </search> and <result> </result> tags respectively. If Python code execution is needed, the code and results are enclosed within <code> </code> and <result> </result> tags respectively. Example: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process based on search result. </think> <answer> The final answer is \\boxed{answer here} </answer>. Or with Python code execution: <think> This is the reasoning process. </think> <code> python code here </code> <result> code result here </result> <think> This is the reasoning process based on code result. </think> <answer> The final answer is \\boxed{answer here} </answer>. If no tools are needed: <think> This is the reasoning process. </think> <answer> The final answer is \\boxed{answer here} </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.         
"""
        elif self.prompt_type == 'code_search_with_budget':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
You should make each tool call efficient and obtain useful results, considering that each Python interpreter call costs [python_cost] and each search call costs [search_cost]. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \ You have a total budget of [total_budget].
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
            self.prompt_template = self.prompt_template.replace("[python_cost]", str(self.python_budget))
            self.prompt_template = self.prompt_template.replace("[search_cost]", str(self.search_budget))
            self.prompt_template = self.prompt_template.replace("[total_budget]", str(self.max_tool_budget))
        elif self.prompt_type == 'search':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
        elif self.prompt_type == 'math':
            self.prompt_template = """
You are a helpful assistant that can solve the given question step by step with the help of the python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. \
For example, <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
    def load_model(self,config):
        model = LLM(
                    config['model_path'],
                    dtype=config['type'],
                    enforce_eager=True,
                    trust_remote_code=True,
                    max_model_len=config['max_input_len'],
                    gpu_memory_utilization=config['gpu_use'],
                    tensor_parallel_size=config['gpu_num'],
                )
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
        return model, tokenizer
    def resume_data_check(self):
        self.load_datas()
        res = []
        total_examples = min(len(self.questions), self.counts)
        self.questions=[]
        self.answers=[]
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                old_res = json.load(f)
            print(f"resume data check, evaluate from {self.output_path}")
            for line in old_res:
                if line["finished_reason"]=="":
                    return False
                elif line["finished_reason"]=="length_limit":
                    return False
                elif line.get("search_error_empty_times",0)>0:
                    return False
                else:
                    res.append(line)
            print(f"total examples: {total_examples}")
            print(f"resume data check, {len(res)} examples has been evaluated, {total_examples - len(res)} examples left to evaluate")
            if len(res) >= total_examples:
                print(f"all data have been evaluated")
                return  True
        return False
    def run(self):
        self.load_datas()
        res = []
        total_examples = min(len(self.questions), self.counts)
        questions = self.questions[:total_examples]
        answers = self.answers[:total_examples]
        #THREEGOLDCHANGE:resume evaluate
        if os.path.exists(self.output_path) and self.resume_evaluate:
            with open(self.output_path, "r") as f:
                old_res = json.load(f)
            new_questions = []
            new_answers = []
            # print(f"resume evaluate from {self.output_path}")
            for line in old_res:
                if line["finished_reason"]=="":
                    if "question" in line.keys():
                        new_question = line["question"]
                    else:
                        new_question = line["Prompt"].split("<|im_end|>\n<|im_start|>user\n")[-1].replace("<|im_end|>\n<|im_start|>assistant\n","")
                    new_questions.append(new_question)
                    # print(f"resume question:{new_question}")
                    new_answers.append(line["answer"])
                elif line["finished_reason"]=="length_limit" and "budget_limit" not in self.output_path:
                    if "question" in line.keys():
                        new_question = line["question"]
                    else:
                        new_question = line["Prompt"].split("<|im_end|>\n<|im_start|>user\n")[-1].replace("<|im_end|>\n<|im_start|>assistant\n","")
                    new_questions.append(new_question)
                    # print(f"resume question:{new_question}")
                    new_answers.append(line["answer"])
                elif line.get("search_error_empty_times",0)>0:
                    if "question" in line.keys():
                        new_question = line["question"]
                    else:
                        new_question = line["Prompt"].split("<|im_end|>\n<|im_start|>user\n")[-1].replace("<|im_end|>\n<|im_start|>assistant\n","")
                    new_questions.append(new_question)
                    # print(f"resume question:{new_question}")
                    new_answers.append(line["answer"])
                else:
                    res.append(line)
            questions = new_questions
            answers = new_answers
            print(f"resume {len(res)} evaluate from {self.output_path}; {len(questions)} questions left to evaluate")
            if len(res) == total_examples:
                print(f"all data have been evaluated")
                return  
        #THREEGOLDCHANGE

        num_batches = math.ceil(len(questions) / self.batch_size)
        print(f"dataset {self.dataset_name} all counts: {total_examples}, batch size: {self.batch_size}, bath counts: {num_batches}")
        
        
        for batch_idx in tqdm(range(num_batches), desc=f"Processing batches"):
            # deep_search("Who is XiJingPing",search_engine=self.search_engine) 
            # break
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(questions))
            batch_samples = questions[start_idx:end_idx]
            golden_answers = answers[start_idx:end_idx]
            
            prompts = []
            for item in batch_samples:
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": self.prompt_template
                            },
                            {
                                "role": "user",
                                "content": item
                            }
                        ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                    )
                )
            
            outputs = []
            generating = list(range(len(prompts))) 
            completed = [] 
            concat_prompts_outputs = prompts.copy()  
            python_rounds = [0 for _ in range(len(prompts))]
            search_rounds = [0 for _ in range(len(prompts))]
            #THREEGOLDCHANGE:记录调用的总工具次数
            calling_rounds = [0 for _ in range(len(prompts))]
            calling_budegets = [0 for _ in range(len(prompts))]
            finished_reason = ["" for _ in range(len(prompts))]
            search_error_empty_times = [0 for _ in range(len(prompts))]
            python_error_times = [0 for _ in range(len(prompts))]
            #THREEGOLDCHANGE
            rollback_rounds = [0 for _ in range(len(prompts))]
            debug_rounds = [0 for _ in range(len(prompts))]
            refine_rounds = [0 for _ in range(len(prompts))]
            curr_max_tokens = [self.max_response_length] * len(concat_prompts_outputs)
            while generating:
                input_prompts = [concat_prompts_outputs[i] for i in generating]
                active_max_tokens = [curr_max_tokens[i] for i in generating]
                self.params_config.stop = ['</python>', '</search>', '</answer>', '</code>']
                self.params_config.detokenize = True
                self.params_config.max_tokens = max(active_max_tokens)
                #THREEGOLDCHANGE:from detokenizer to post_process_tokens
                t1 = time.time()
                print(self.params_config)
                initial_outputs2 = self.model.generate(
                    input_prompts,
                    self.params_config,
                    use_tqdm=False, 
                )
                t2 = time.time()
                print(f"total generate: {len(input_prompts)}, max prompt length: {max([len(input_prompt) for input_prompt in input_prompts])}, detokenize one turn generate time: {t2 - t1}")
                # print(self.params_config)                
                # self.params_config.stop = None
                # self.params_config.detokenize = False
                # self.params_config.max_tokens = max(active_max_tokens)
                # self.params_config.stop_token_ids = [151644]
                # t1 = time.time()
                # initial_outputs = self.model.generate(
                #     input_prompts,
                #     self.params_config,
                #     use_tqdm=False,
                # )
                # t2 = time.time()
                # print(f"max prompt length: {max([len(input_prompt) for input_prompt in input_prompts])} no detokenize one turn generate time: {t2 - t1}")
                # print(self.params_config)
                # def get_new_output_str(output_token_ids):
                #     output_str = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
                #     if '</search>' in output_str:
                #         output_str = output_str.split('</search>')[0] + '</search>'
                #     elif '</python>' in output_str:
                #         output_str = output_str.split('</python>')[0] + '</python>'
                #     #可能和之前逻辑不一样的地方
                #     elif '</answer>' in output_str:
                #         output_str = output_str.split('</answer>')[0] + '</answer>'
                #         output_str = output_str + self.tokenizer.eos_token
                #     else:
                #         output_str = output_str
                #     return output_str
                # outputs = [get_new_output_str(output.outputs[0].token_ids) for output in initial_outputs]
                #THREEGOLDCHANGE
                outputs = [output.outputs[0].text for output in initial_outputs2]
                ### 这里的outputs可能为空
                
                ### 这里的outputs可能为空
                vllm_finish_reasons = [output.outputs[0].finish_reason for output in initial_outputs2]
                vllm_stop_reasons = [output.outputs[0].stop_reason for output in initial_outputs2]
                python_indices = []
                search_indices = []
                achieve_max_tool_budget = []
                other_indices = []
                text_generating_indices = [] #在超过最大调用次数之后继续输出一次
                
                for i in range(len(outputs)):
                    if "code_search" in self.prompt_type or self.prompt_type == 'no-tool-inference':
                        if outputs[i].strip().endswith('</python>'):
                            #THREEGOLDCHANGE
                            if calling_rounds[generating[i]] >= self.max_calling_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_calling_times"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                            elif self.python_budget+calling_budegets[generating[i]] > self.max_tool_budget: #THREEGOLDCHANGE
                                # 对达到budget上限的推理进行处理
                                text_generating_indices.append((generating[i], outputs[i]))
                                # other_indices.append((generating[i], outputs[i]))
                                # finished_reason[generating[i]] = "achieve_max_tool_budget"
                                # print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                                achieve_max_tool_budget.append((generating[i], outputs[i]))
                            elif python_rounds[generating[i]] >= self.max_python_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_python_times"
                                other_indices.append((generating[i], outputs[i]))
                            else:
                                python_indices.append((generating[i], outputs[i]))
                                python_rounds[generating[i]] += 1
                                calling_rounds[generating[i]] += 1
                                calling_budegets[generating[i]] += 1*self.python_budget
                        elif outputs[i].strip().endswith('</code>'):
                            if calling_rounds[generating[i]] >= self.max_calling_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_calling_times"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                            elif self.python_budget+calling_budegets[generating[i]] > self.max_tool_budget: #THREEGOLDCHANGE
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_tool_budget"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                            elif python_rounds[generating[i]] >= self.max_python_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_python_times"
                                other_indices.append((generating[i], outputs[i]))
                            else:
                                python_indices.append((generating[i], outputs[i]))
                                python_rounds[generating[i]] += 1
                                calling_rounds[generating[i]] += 1
                                calling_budegets[generating[i]] += 1*self.python_budget
                        elif outputs[i].strip().endswith('</search>'):
                            #THREEGOLDCHANGE
                            if calling_rounds[generating[i]] >= self.max_calling_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_calling_times"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                            elif self.search_budget+calling_budegets[generating[i]] > self.max_tool_budget: #THREEGOLDCHANGE
                                # 对达到budget上限的推理进行处理
                                text_generating_indices.append((generating[i], outputs[i]))
                                # other_indices.append((generating[i], outputs[i]))
                                # finished_reason[generating[i]] = "achieve_max_tool_budget"
                                # print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                                achieve_max_tool_budget.append((generating[i], outputs[i]))
                            elif search_rounds[generating[i]] >= self.max_search_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                finished_reason[generating[i]] = "achieve_max_search_times"
                                other_indices.append((generating[i], outputs[i]))
                            else:
                                search_indices.append((generating[i], outputs[i]))
                                search_rounds[generating[i]] += 1
                                calling_rounds[generating[i]] += 1 #THREEGOLDCHANGE
                                calling_budegets[generating[i]] += 1*self.search_budget
                        elif outputs[i].strip().endswith('</answer>'):
                            other_indices.append((generating[i], outputs[i]))
                            # print(outputs[i])
                            finished_reason[generating[i]] = "normal_finish"
                            print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                        else:
                            other_indices.append((generating[i], outputs[i]))
                            if vllm_finish_reasons[i]=="stop" and vllm_stop_reasons[i] is None:
                                finished_reason[generating[i]] = "abnormal_finish_with_eos"
                            elif vllm_finish_reasons[i]=="length":
                                finished_reason[generating[i]] = "length_limit"
                            else:
                                finished_reason[generating[i]] = "abnormal_finish_with_unknown"
                            print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                    elif self.prompt_type == 'search': #TODO:针对其他prompt的修改
                        if outputs[i].strip().endswith('</search>'):
                            if search_rounds[generating[i]] >= self.max_search_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                            else:
                                search_indices.append((generating[i], outputs[i]))
                                search_rounds[generating[i]] += 1
                                calling_rounds[generating[i]] += 1 #THREEGOLDCHANGE
                                calling_budegets[generating[i]] += 1*self.search_budget
                                finished_reason[generating[i]] = "normal_finish"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                        else:
                            other_indices.append((generating[i], outputs[i]))
                            finished_reason[generating[i]] = "normal_finish"
                            print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                    elif self.prompt_type == 'math':
                        if outputs[i].strip().endswith('</python>'):
                            if python_rounds[generating[i]] >= self.max_python_times:
                                text_generating_indices.append((generating[i], outputs[i]))
                                other_indices.append((generating[i], outputs[i]))
                            else:
                                python_indices.append((generating[i], outputs[i]))
                                python_rounds[generating[i]] += 1
                                calling_rounds[generating[i]] += 1 #THREEGOLDCHANGE
                                calling_budegets[generating[i]] += 1*self.python_budget
                                finished_reason[generating[i]] = "normal_finish"
                                print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                        else:
                            other_indices.append((generating[i], outputs[i]))
                            finished_reason[generating[i]] = "normal_finish"
                            print(f"batch {batch_idx} data {i} finished reason: {finished_reason[generating[i]]}")
                
                if python_indices:
                    python_contents = []
                    for i, content in python_indices:
                        python_contents.append(content)
                        concat_prompts_outputs[i] += content
                    python_contents = [extract_python_content(content) for content in python_contents]
                    for i, (idx, content) in enumerate(python_indices):
                        result, report = self.executor.apply(python_contents[i])
                        result_ids = self.tokenizer.encode(result)
                        if len(result_ids) > self.max_obs_length:
                            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(result_ids)} & {self.max_obs_length}")            
                            result_ids = result_ids[:self.max_obs_length]
                        result = self.tokenizer.decode(result_ids)
                        if report == "Done":
                            concat_prompts_outputs[idx] += f'<result>\n{result}\n</result>'
                        else:
                            python_error_times[idx] += 1
                            if not self.use_debug:
                                if not self.use_rollback or rollback_rounds[idx] >= self.max_rollback_times: #Tool-Use Backtracer
                                    concat_prompts_outputs[idx] += f'<result>\n{report}\n</result>'
                                else:
                                    concat_prompts_outputs[idx] = rollback(concat_prompts_outputs[idx])
                                    print(f'=========== code error: {report}, try to rollback =============')
                                    rollback_rounds[idx] += 1
                            else:
                                if debug_rounds[idx] >= self.max_debug_times: #Code Debugger
                                    if not self.use_rollback or rollback_rounds[idx] >= self.max_rollback_times:
                                        concat_prompts_outputs[idx] += f'<result>\n{report}\n</result>'
                                    else:
                                        concat_prompts_outputs[idx] = rollback(concat_prompts_outputs[idx])
                                        print(f'=========== code error: {report}, try to rollback =============')
                                        rollback_rounds[idx] += 1
                                else:
                                    print(f'=========== code error: {report}, try to debug =============')
                                    refine_code = debug_code_function(python_contents[i], report)
                                    debug_rounds[idx] += 1
                                    result, report = self.executor.apply(refine_code)
                                    if report == "Done":
                                        print(f'=========== debug success, the result is: {result}=============')
                                        concat_prompts_outputs[idx] += f'<result>\n{result}\n</result>'
                                    else:
                                        print(f'=========== code error: {report}, debug error =============')
                                        concat_prompts_outputs[idx] += f'<result>\n{report}\n</result>'
                    
                if search_indices:
                    print('#############search begin#############')
                    search_contents = []
                    for i, content in search_indices:
                        search_contents.append(
                            content
                        )
                        concat_prompts_outputs[i] += content
                    search_contents = [extract_search_content(content) for content in search_contents]

                    if (self.task == 'qa' and self.dataset_name != 'webwalker' and self.dataset_name != 'gpqa' and self.dataset_name != 'hle' and self.dataset_name != 'gaia' and self.all_wiki == 0) or self.all_wiki == 1:
                        print("-------------wiki search-------------")
                        search_results = batch_search(search_contents)
                        for i, (idx, content) in enumerate(search_indices):
                            if search_results[i] == 'error':
                                search_error_empty_times[idx] += 1
                                if self.use_rollback and rollback_rounds[idx] < self.max_rollback_times: #Tool-Use Backtracer
                                    pass
                                else:
                                    concat_prompts_outputs[idx] += f'<result>\n\n</result>'
                            else:
                                result_ids = self.tokenizer.encode(search_results[i])
                                if len(result_ids) > self.max_obs_length:
                                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(result_ids)} & {self.max_obs_length}")            
                                    result_ids = result_ids[:self.max_obs_length]
                                search_result = self.tokenizer.decode(result_ids)
                                concat_prompts_outputs[idx] += f'<result>\n{search_result}\n</result>'
                    else:
                        for i, (idx, content) in enumerate(search_indices):
                            try:
                                search_result = deep_search(search_contents[i],search_engine=self.search_engine) 
                                if search_result == '':
                                    search_error_empty_times[idx] += 1
                                result_ids = self.tokenizer.encode(search_result)
                                if len(result_ids) > self.max_obs_length:
                                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(result_ids)} & {self.max_obs_length}")            
                                    result_ids = result_ids[:self.max_obs_length]
                                search_result = self.tokenizer.decode(result_ids)
                                concat_prompts_outputs[idx] += f'<result>\n{search_result}\n</result>'
                            except Exception as e:
                                if self.use_rollback and rollback_rounds[idx] < self.max_rollback_times: #Tool-Use Backtracer
                                    pass
                                else:
                                    print(f"search error: {e}")
                                    concat_prompts_outputs[idx] += f'<result>\n\n</result>'
                    print('search end')
                    
                if achieve_max_tool_budget:
                    print("model have achieved max tool budget")
                    for idx, content in achieve_max_tool_budget:
                        concat_prompts_outputs[idx] += content
                        # concat_prompts_outputs[idx] += f'<result>\nReached the maximum budget, please answer directly based on your previous information\n</result>'
                        concat_prompts_outputs[idx] += f'<result>\nReached the maximum budget, please answer directly based on your previous information\n</result> <think></think><answer>'

                for i in generating:
                    current_sequence = concat_prompts_outputs[i]
                    sequence_tokens = self.tokenizer.encode(current_sequence)
                    #THREEGOLDCHANGE:参考train的rollout计算剩下的生成长度
                    input_tokens = self.tokenizer.encode(prompts[i])
                    curr_max_tokens[i] = self.max_response_length - (len(sequence_tokens) - len(input_tokens))
                    #THREEGOLDCHANGE
                    if not self.use_refiner or refine_rounds[i] >= self.max_refine_times:
                        continue
                    if len(sequence_tokens) >= 8192:
                        print(f"current length of tokens is more than 8192, begin to refine")
                        concat_prompts_outputs[i] = refine(prompts[i], current_sequence) #Reasoning Chain Refiner
                        print(f"===================== refine result =====================")
                        print(concat_prompts_outputs[i])
                        print(f"=====================================================")
                        refine_rounds[i] += 1

                # if text_generating_indices:#逻辑怎么和训练的时候不一样
                #     generate_results = []
                #     for i, content in text_generating_indices:
                #         generate_results.append(
                #             concat_prompts_outputs[i] + content
                #         )
                #         concat_prompts_outputs[i] += content
                #     self.params_config.stop = None
                #     output_texts = self.model.generate(
                #         generate_results,
                #         self.params_config,
                #         use_tqdm=False,
                #     )
                #     for i in range(len(output_texts)):
                #         text = output_texts[i].outputs[0].text
                #         concat_prompts_outputs[text_generating_indices[i][0]] += text
                #         completed.append(text_generating_indices[i][0])
                
                if other_indices:
                    for i, content in other_indices:
                        concat_prompts_outputs[i] += content
                        completed.append(i)

                
                generating = [i for i in generating if i not in completed]

            extracted_answers = []
            for i in range(len(concat_prompts_outputs)):
                text = concat_prompts_outputs[i][len(prompts[i]):]
                # Extract answer using the last occurrence of <answer>...</answer>
                # This ensures we get the latest answer in case there are multiple sections
                last_answer_end = text.rfind('</answer>')
                if last_answer_end != -1:
                    # Find the corresponding opening tag before this closing tag
                    temp_text = text[:last_answer_end]
                    last_answer_start = temp_text.rfind('<answer>')
                    if last_answer_start != -1:
                        temp_answer = text[last_answer_start + len('<answer>'):last_answer_end]
                    else:
                        temp_answer = None
                else:
                    temp_answer = None
                if temp_answer:
                    boxed_answer = temp_answer.strip()
                    boxed_answer = last_boxed_only_string(boxed_answer)
                    if boxed_answer and boxed_answer.startswith("\\boxed{") and boxed_answer.endswith("}"):
                        boxed_content = boxed_answer[7:-1]  # Extract content between \\boxed{ and }
                        boxed_answer = boxed_content
                    if not boxed_answer:
                        final_answer = temp_answer
                    else:
                        final_answer = boxed_answer
                else:
                    boxed_answer = text.strip()
                    final_answer = last_boxed_only_string(boxed_answer)
                    if final_answer and final_answer.startswith("\\boxed{") and final_answer.endswith("}"):
                        final_answer = final_answer[7:-1]  # Extract content between \\boxed{ and }
                extracted_answers.append(final_answer)
            
            for i in range(len(batch_samples)):
                print(f"batch {batch_idx}, data {i}: refine result: {extracted_answers[i]}")
                res.append(
                    {
                        "Prompt": prompts[i], 
                        "Full_output": concat_prompts_outputs[i][len(prompts[i]):], 
                        "Output": extracted_answers[i], 
                        "question": batch_samples[i],
                        "answer": golden_answers[i],
                        "finished_reason": finished_reason[i],
                        "calling_rounds": calling_rounds[i],
                        "calling_budegets": calling_budegets[i],
                        "python_budget": self.python_budget,
                        "python_rounds": python_rounds[i],
                        "search_budget": self.search_budget,
                        "search_rounds": search_rounds[i],
                        "search_error_empty_times": search_error_empty_times[i],
                        "python_error_times": python_error_times[i],
                    }
                )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
            f.close()
        print(f"results have been saved to {self.output_path}")
        global search_cache
        global search_cache_file
        cache_file = search_cache_file
        print(f"save search cache to {cache_file}")
        with open(cache_file, "w", encoding='utf-8') as f:
            json.dump(search_cache, f, indent=4, ensure_ascii=False)
        timestamp_ms = int(time.time() * 1000)
        print(timestamp_ms)  # 输出示例：1722147645123
        time_cache_file = os.path.join(os.path.dirname(os.path.abspath(cache_file)), f"search_cache_{timestamp_ms}.json")
        with open(time_cache_file, "w", encoding='utf-8') as f:
            json.dump(search_cache, f, indent=4, ensure_ascii=False)

    def load_datas(self):
        data_path = f'/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/data/{self.dataset_name}/test.jsonl'
        print(json.dumps(
            {
                'dataset': self.dataset_name, 'output': self.output_path,
            }, ensure_ascii=False, indent=4
        ))
        if 'aime24' in data_path or 'amc23' in data_path or \
            'gsm8k' in data_path or 'tabmwp' in data_path or 'gaokao2023en' in data_path or 'college_math' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.questions.append(data['question'])
                    answer = data['answer']
                    if 'gsm8k' in data_path:
                        answer = extract_solution(answer)
                    self.answers.append(answer)
        elif 'svamp' in data_path or 'asdiv' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    body = data['body'] if 'body' in data else data['Body']
                    question = data['question'] if 'question' in data else data['Question']
                    answer = data['answer'] if 'answer' in data else data['Answer']
                    if 'asdiv' in data_path:
                        answer = answer.split(" (")[0]
                    self.questions.append(body + " " + question)
                    self.answers.append(answer)
        elif 'mawps' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['input']
                    answer = data['target']
                    self.questions.append(question)  
                    self.answers.append(answer)
        elif 'carp_en' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['content']
                    answer = data['answer']
                    self.questions.append(question)
                    self.answers.append(answer)
        elif 'minerva_math' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['problem']
                    answer = data['solution']
                    try:
                        answer = remove_boxed(last_boxed_only_string(answer))
                    except:
                        pass
                    self.questions.append(question)
                    self.answers.append(answer)
        elif 'olympiadbench' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['question']
                    answer = data['final_answer'][0]
                    self.questions.append(question)
                    self.answers.append(answer)
        elif '/math/test' in data_path or 'aime25' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['problem']
                    answer = data['answer']
                    self.questions.append(question)
                    self.answers.append(answer)
        elif 'gaia' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['Question']
                    answer = data['answer']
                    self.questions.append(question)
                    self.answers.append(answer)
        elif 'gpqa' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['question']
                    answer = data['answer']
                    self.questions.append(question)
                    self.answers.append(answer)
        elif 'OlymMATH' in data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['problem']
                    answer = data['answer']
                    self.questions.append(question)
                    self.answers.append(answer)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.questions.append(data['question'])
                    answer = data['answer']
                    self.answers.append(answer)




if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Torl test")
    argument_parser.add_argument(
        "--model_path",
        type=str,
        default="/home/sxjiang/model/Tool-Star-Qwen-3B",
        help="Model path to use for testing",
    )
    argument_parser.add_argument(
        "--gpu_use",
        type=float,
        default=0.7,
        help="GPU to use for testing",
    )
    argument_parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
    )
    argument_parser.add_argument(
        "--max_response_length",
        type=int,
        default=4096,
    )
    argument_parser.add_argument(
        "--max_input_len",
        type=int,
        default=8192,
    )
    argument_parser.add_argument(
        "--task",
        type=str,
        default='qa',
    )
    argument_parser.add_argument(
        "--dataset_name",
        type=str,
        default='hotpotqa',
    )
    argument_parser.add_argument(
        "--output_path",
        type=str,
        default="/home/sxjiang/myproject/agent/Tool-Star-OCT/evaluation/result/debug_result_2.json",
        help="Path to the data file",
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    argument_parser.add_argument(
        "--prompt_type",
        type=str,
        default='code_search',
    )
    argument_parser.add_argument(
        "--counts",
        type=int,
        default=100,
    )
    argument_parser.add_argument(
        "--use_debug",
        action='store_true',
    )
    argument_parser.add_argument(
        "--use_rollback",
        action='store_true',
    )
    argument_parser.add_argument(
        "--use_refiner",
        action='store_true',
    )
    argument_parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    #THREEGOLDCHANGE:增加初始化部分
    argument_parser.add_argument(
        "--max_obs_length",
        type=int,
        default=512
    )
    argument_parser.add_argument(
        "--all_wiki",
        type=int,
        default=0
    )
    argument_parser.add_argument(
        "--resume_evaluate",
        action='store_true',
    )
    argument_parser.add_argument(
        "--max_calling_times",
        type=int,
        default=4
    )
    argument_parser.add_argument(
        "--max_tool_budget",
        type=int,
        default=30
    )
    argument_parser.add_argument(
        "--python_budget",
        type=int,
        default=-1
    )
    argument_parser.add_argument(
        "--search_budget",
        type=int,
        default=-1
    )
    argument_parser.add_argument(
        "--search_engine",
        type=str,
        default="langsearch",
    )
    argument_parser.add_argument(
        "--budget_idx",
        type=int,
        default=-1
    )
    #THREEGOLDCHANGE
    
    args = argument_parser.parse_args()

    model_config = {
        'model_path': args.model_path,
        'type': torch.bfloat16,
        # 'type': torch.float16,
        'max_input_len': args.max_input_len,
        'gpu_use': args.gpu_use,
        'gpu_num': torch.cuda.device_count(),
        'lora_path': None,
    }
    params_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_response_length,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
        'repetition_penalty': 1.1,
        'n': 1,
        # 'stop': ['```python'],
        # 'seed': 7777,
        'include_stop_str_in_output': True,
    }
    print(f"params:{args}")
    inference = Inference(
        model_config=model_config,
        params_config=params_config,
        task=args.task,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        counts=args.counts,
        prompt_type=args.prompt_type,
        use_debug=args.use_debug,
        use_rollback=args.use_rollback,
        use_refiner=args.use_refiner,
        max_response_length=args.max_response_length,
        max_calling_times=args.max_calling_times,
        max_tool_budget=args.max_tool_budget,
        python_budget=args.python_budget,
        search_budget=args.search_budget,
        search_engine=args.search_engine,
        resume_evaluate=args.resume_evaluate,
        max_obs_length=args.max_obs_length,
        all_wiki=args.all_wiki,
        budget_idx=args.budget_idx
    )
    inference.run()