import numpy as np
import torch

def get_group_advantage(rewards):
    group_advantage = []
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    for i in range(len(rewards)):
        group_advantage.append((rewards[i] - mean_reward) / std_reward)
    return np.array(group_advantage)




def get_oct_penalty(budgets,rewards,oct_smooth=3*2):
    acc_budgets = []
    for i,reward in enumerate(rewards):
        if rewards[i] >0:
            acc_budgets.append(budgets[i])
    def map_to_2n(calling_cost,optim_cost):# 3.map calling_cost to 2*optim_cost
        if calling_cost == 0 and optim_cost == 0:
            return 0
        elif optim_cost== 0:
            return calling_cost
        else:
            return 2*optim_cost*calling_cost/(optim_cost+calling_cost)
    optim_budget = np.min(acc_budgets)
    print(f"optim_budget: {optim_budget}")
    oct_scores = np.zeros(len(rewards))
    for i,reward in enumerate(rewards):
        calling_cost = budgets[i] #m
        if reward<=0:
            oct_scores[i] = 1.0
            continue
        map_costs = map_to_2n(calling_cost=calling_cost,optim_cost=optim_budget)
        if map_costs==0 and optim_budget==0:
            oct_scores[i] = 1.0
        elif optim_budget==0:
            print("SMOOTH")
            oct_scores[i] = np.cos(np.pi*calling_cost/(2*calling_cost+oct_smooth))
        else:
            oct_scores[i] = np.sin(np.pi*map_costs/(2*optim_budget))
    return oct_scores

def grpo_oct_sim(rewards,budgets,oct_smooth=3*2,mode="multiply"):
    print("#"*10)
    print(get_group_advantage(rewards))
    print("------add oct penalty------")
    print(get_oct_penalty(budgets,rewards))
    if mode=="multiply":
        print("---multiply mode---")
        print(get_oct_penalty(budgets,rewards)*get_group_advantage(rewards))
    elif mode=="add":
        print("---add mode---")
        print(get_oct_penalty(budgets,rewards)+get_group_advantage(rewards))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

# 工具A：4 工具B：1
# 调用A:1 + B:1时才能答对（在f1 score下）
# 调用次数限制3
# 单调用A或者B都没办法答对
# 模拟：答得半对但是工具buget小，以及答得全对但是工具buget高+OCT的penalty乘法因子形式==》答得对肯定好一些
budgets = np.array([1,4,3,5,2,2,5,6])
rewards = np.array([0.68,-1,0,1,-1,0,1,1])
oct_smooth = 3*1
grpo_oct_sim(rewards,budgets)
# 工具A：4 工具B：1
# 调用A:1 + B:1时才能答对（在f1 score下）
# 调用次数限制3
# 单调用A或者B都没办法答对
# 模拟：答得半对但是工具buget小，以及答得全对但是工具buget高+OCT的penalty加法因子形式==》答得对肯定好一些
budgets = np.array([1,4,3,5,2,2,5,6])
rewards = np.array([0.68,-1,0,1,-1,0,1,1])
oct_smooth = 3*1
grpo_oct_sim(rewards,budgets,mode="add")