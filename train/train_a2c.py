import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.load_data import load_fjsp_instance
from Fjsp_env import FJSPEnvironment
from A2C import A2CAgent
import matplotlib.pyplot as plt

# 加载FJSP实例
file_path = "../data/Mk01.fjs"
n_jobs, n_machines, operations = load_fjsp_instance(file_path)
env = FJSPEnvironment(n_jobs, n_machines, operations)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 这里是A2C智能体的初始化参数
agent = A2CAgent(
    state_size, 
    action_size, 
    lr=9e-5,        
    gamma=0.99      
)

n_episodes = 1000   # 训练轮数，可增大如2000、5000
test_freq = 20
best_makespan = float('inf')
best_schedule = None
makespans = []
episode_rewards = []
losses = []  # Initialize losses as an empty list
avg_makespans = []  # Initialize avg_makespans as an empty list
step_counts = []
action_counts = np.zeros(action_size, dtype=int)

for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    trajectory = []
    step_count = 0

    while not done:
        # 获取有效动作
        valid_actions = []
        for op_idx in env.unscheduled_ops:
            job_idx, op_pos = env._get_job_op_from_idx(op_idx)
            op_options = env.operations[job_idx][op_pos]
            valid_machines = [opt[0] for opt in op_options]
            for machine_idx in valid_machines:
                action = op_idx * env.n_machines + machine_idx
                valid_actions.append(action)

        action, _ = agent.select_action(state, valid_actions)
        action_counts[action] += 1
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, next_state, done, valid_actions))
        state = next_state
        total_reward += reward
        step_count += 1

    loss = agent.update(trajectory)
    losses.append(loss)
    makespan = max(env.machine_available_time)
    makespans.append(makespan)
    episode_rewards.append(total_reward)
    step_counts.append(step_count)

    if makespan < best_makespan:
        best_makespan = makespan
        best_schedule = env.schedule.copy()

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Makespan: {makespan}, Best: {best_makespan}, Reward: {total_reward}")

# 输出最佳调度结果
print("\nBest Scheduling Result:")
print(f"Minimum Makespan: {best_makespan}")
print("\nDetailed Schedule:")
for job_idx, op_pos, machine_idx, start_time, completion_time in best_schedule:
    print(f"Job {job_idx+1}, Operation {op_pos+1}, Machine {machine_idx+1}, Start: {start_time}, End: {completion_time}")

# 设置matplotlib使用不需要中文支持的字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 多子图绘制
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# 1. 完工时间变化及均值±std
window = 50
means = np.convolve(makespans, np.ones(window)/window, mode='valid')
stds = [np.std(makespans[max(0, i-window+1):i+1]) for i in range(len(makespans))]
axs[0, 0].plot(makespans, alpha=0.3, label='Raw Makespan')
axs[0, 0].plot(np.arange(window-1, len(makespans)), means, 'b-', label='Mean')
axs[0, 0].fill_between(np.arange(window-1, len(makespans)), means-stds[window-1:], means+stds[window-1:], color='b', alpha=0.1)
axs[0, 0].set_title('Makespan Changes During Training')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. 损失变化
if losses:
    axs[0, 1].plot(losses, 'g-')
    axs[0, 1].set_title('Training Loss Over Episodes')
    axs[0, 1].grid(True)

# 3. 总奖励变化及均值±std
reward_means = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
reward_stds = [np.std(episode_rewards[max(0, i-window+1):i+1]) for i in range(len(episode_rewards))]
axs[1, 0].plot(episode_rewards, alpha=0.3, label='Raw Total Reward')
axs[1, 0].plot(np.arange(window-1, len(episode_rewards)), reward_means, 'm-', label='Mean')
axs[1, 0].fill_between(np.arange(window-1, len(episode_rewards)), reward_means-reward_stds[window-1:], reward_means+reward_stds[window-1:], color='m', alpha=0.1)
axs[1, 0].set_title('Total Reward Per Episode')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. 动作分布
axs[1, 1].bar(np.arange(action_size), action_counts, color='orange', alpha=0.7)
axs[1, 1].set_title('Action Selection Distribution')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('a2c_multi_panel.png')
plt.show()

# 绘制甘特图
plt.figure(figsize=(12, 8))
colors = plt.cm.jet(np.linspace(0, 1, n_jobs))
machine_labels = [f"Machine {i+1}" for i in range(n_machines)]
for job_idx, op_pos, machine_idx, start_time, completion_time in best_schedule:
    duration = completion_time - start_time
    plt.barh(machine_idx, duration, left=start_time, height=0.5,
             color=colors[job_idx], edgecolor='black')
    plt.text(start_time + duration/2, machine_idx,
             f"J{job_idx+1}-O{op_pos+1}", ha='center', va='center')
plt.grid(axis='x')
plt.xlabel('Time')
plt.yticks(range(n_machines), machine_labels)
plt.title('Best Schedule Gantt Chart')
plt.savefig('best_schedule_gantt.png')
plt.show()