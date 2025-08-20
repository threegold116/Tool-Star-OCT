import numpy as np
import torch
import matplotlib.pyplot as plt
import os
def get_group_advantage(rewards):
    group_advantage = []
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    for i in range(len(rewards)):
        group_advantage.append((rewards[i] - mean_reward) / std_reward)
    return np.array(group_advantage)


def get_oct_all_penalty(budgets,rewards,oct_smooth=3*2):
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
        if reward<0:
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

def get_oct_penalty_all_min(budgets,rewards,oct_smooth=3*2):
    def map_to_2n(calling_cost,optim_cost):# 3.map calling_cost to 2*optim_cost
        if calling_cost == 0 and optim_cost == 0:
            return 0
        elif optim_cost== 0:
            return calling_cost
        else:
            return 2*optim_cost*calling_cost/(optim_cost+calling_cost)
    optim_budget = np.min(budgets)
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
            oct_scores[i] = np.cos(np.pi*calling_cost/(2*calling_cost+oct_smooth))
        else:
            oct_scores[i] = np.sin(np.pi*map_costs/(2*optim_budget))
    return oct_scores

def grpo_oct_sim(rewards,budgets,oct_smooth=3*2,mode="multiply"):
    print("#"*10)
    print("rewards",rewards)
    print("budgets",budgets)
    print("no_oct_advantage",get_group_advantage(rewards))
    print("------add oct penalty------")
    print("oct_radio",get_oct_penalty(budgets,rewards))
    print("all_min_oct_radio",get_oct_penalty_all_min(budgets,rewards))
    print("------add all oct penalty------")
    all_oct_scores = get_oct_all_penalty(budgets,rewards)
    if mode=="multiply":
        print("---multiply mode---")
        print("oct_multiply_reward",get_oct_penalty(budgets,rewards)*rewards)
        print("oct_multiply_advantage",get_group_advantage(get_oct_penalty(budgets,rewards)*rewards))
        print("all_min_oct_multiply_reward",get_oct_penalty_all_min(budgets,rewards)*rewards)
        print("all_min_oct_multiply_advantage",get_group_advantage(get_oct_penalty_all_min(budgets,rewards)*rewards))
        print("------add all oct penalty------")
        new_rewards = np.zeros(len(rewards))
        for idx in range(len(all_oct_scores)):

            if rewards[idx] > 0:
                new_rewards[idx] = rewards[idx] * all_oct_scores[idx]
            if rewards[idx] == 0:
                new_rewards[idx] = all_oct_scores[idx] - 1.0
            if rewards[idx] <0:
                new_rewards[idx] = rewards[idx]
        print("oct_multiply_all_reward",new_rewards)
        print("oct_multiply_all_advantage",get_group_advantage(new_rewards))
        
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
# grpo_oct_sim(rewards,budgets)
# 工具A：4 工具B：1
# 调用A:1 + B:1时才能答对（在f1 score下）
# 调用次数限制3
# 单调用A或者B都没办法答对
# 模拟：答得半对但是工具buget小，以及答得全对但是工具buget高+OCT的penalty加法因子形式==》答得对肯定好一些
# budgets = np.array([1,4,3,5,2,2,5,6])
# rewards = np.array([0.68,-1,0,1,-1,0,1,1])
# oct_smooth = 3*1
# grpo_oct_sim(rewards,budgets,mode="add")
budgets = np.array([3,4,3,5,1,1,7,6])
rewards = np.array([1,-1,0,1,-1,0,1,1])
oct_smooth = 3*1
grpo_oct_sim(rewards,budgets)

budgets = np.array([3,4,3,3,1,1,3,3])
rewards = np.array([1,-1,0,1,-1,0,1,1])
oct_smooth = 3*1
grpo_oct_sim(rewards,budgets)

def draw_with_max(x,y,result_dir,name,optim_cost,smooth):
    plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
    plt.plot(x, y, marker='o', label='Line')  # 画折线图并加点
    max_x = x[y.index(max(y))]
    max_y = max(y)
    print(max_x,max_y)
    # # 添加一条竖线
    plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')
    # # 添加文字标注
    # plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)
    plt.title(f"optim_cost: {optim_cost}, smooth: {smooth}")
    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()
def draw_multi_lines(x,y_list,labels,result_dir,name):
    plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
    colors = plt.cm.tab20(np.arange(len(y_list)) / len(y_list))
    for y,label,color in zip(y_list,labels,colors):
        plt.plot(x, y, marker='o', label=label, color=color)  # 画折线图并加点
    # labelLines(plt.gca().get_lines(), zorder=2.5) 4
    plt.legend()


    # # 添加文字标注
    # plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()    
def map_to_2n(calling_cost,optim_cost):# 3.map calling_cost to 2*optim_cost
    if calling_cost == 0 and optim_cost == 0:
        return 0
    elif optim_cost== 0:
        return calling_cost
    else:
        return 2*optim_cost*calling_cost/(optim_cost+calling_cost)
def sim_oct(cost,optim_cost,smooth):
    def map_to_2n(calling_cost,optim_cost):# 3.map calling_cost to 2*optim_cost
        if calling_cost == 0 and optim_cost == 0:
            return 0
        elif optim_cost== 0:
            return calling_cost
        else:
            return 2*optim_cost*calling_cost/(optim_cost+calling_cost)
    optim_budget = optim_cost
    oct_penalty = 0
    calling_cost = cost #m
    map_costs = map_to_2n(calling_cost=calling_cost,optim_cost=optim_budget)
    if map_costs==0 and optim_budget==0:
        oct_penalty = 1.0
    elif optim_budget==0:
        oct_penalty = np.cos(np.pi*calling_cost/(2*calling_cost+smooth))
    else:
        oct_penalty = np.sin(np.pi*map_costs/(2*optim_budget))
    return oct_penalty
result_dir="/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/analyse/sim_oct"
os.makedirs(result_dir,exist_ok=True)

costs = np.arange(32)
optim_cost = 1
smooth = 2
draw_with_max(costs,[sim_oct(cost,optim_cost,smooth) for cost in costs],result_dir,f"sim_oct_optim_{optim_cost}_smooth_{smooth}",optim_cost,smooth)


costs = np.arange(32)
optim_cost = 4
smooth = 2
draw_with_max(costs,[sim_oct(cost,optim_cost,smooth) for cost in costs],result_dir,f"sim_oct_optim_{optim_cost}_smooth_{smooth}",optim_cost,smooth)

costs = np.arange(32)
optim_cost = 0
smooth = 2
draw_with_max(costs,[sim_oct(cost,optim_cost,smooth) for cost in costs],result_dir,f"sim_oct_optim_{optim_cost}_smooth_{smooth}",optim_cost,smooth)


costs = np.arange(32)
optim_cost = 2
smooth = 2
y = [sim_oct(cost,optim_cost,smooth) for cost in costs]
smooth = 3
y1 = [sim_oct(cost,optim_cost,smooth) for cost in costs]
smooth = 4
y2 = [sim_oct(cost,optim_cost,smooth) for cost in costs]
draw_multi_lines(costs,[y,y1,y2],["smooth_2","smooth_3","smooth_4"],result_dir,f"sim_oct_optim_{optim_cost}_smooth_multi")


costs = np.arange(32)
optim_cost = 2
smooth = 2
draw_with_max(costs,[sim_oct(cost,optim_cost,smooth) for cost in costs],result_dir,f"sim_oct_optim_{optim_cost}_smooth_{smooth}",optim_cost,smooth)


costs = np.arange(32)
optim_cost = 2
smooth = 2
y = [sim_oct(cost,optim_cost,smooth) for cost in costs]
y1 = [map_to_2n(cost,optim_cost) for cost in costs]
draw_multi_lines(costs,[y,y1],["oct_score","map2n"],result_dir,f"sim_oct_map2n_optim_{optim_cost}_smooth_multi")

costs = np.arange(32)
optim_cost = 2
smooth = 2
y = [np.sin(cost*np.pi/(cost+optim_cost)) for cost in costs]
y1 = [np.sin(cost) for cost in costs]
y2 = [cost*np.pi/(cost+optim_cost) for cost in costs]
draw_with_max(costs,y,result_dir,f"sin_oct_optim_{optim_cost}_smooth_{smooth}",optim_cost,smooth)
draw_multi_lines(costs,[y,y1,y2],["sin_map","sin","map2n"],result_dir,f"sin_oct_map2n_optim_{optim_cost}_smooth_multi")
