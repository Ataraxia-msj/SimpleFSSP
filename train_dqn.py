import numpy as np
import matplotlib.pyplot as plt
import torch
from load_data import load_fjsp_instance
from Fjsp_env import FJSPEnvironment
from Dqn import DQNAgent

# 加载FJSP实例
file_path = "data/Mk01.fjs"
n_jobs, n_machines, operations = load_fjsp_instance(file_path)
print(f"Loaded {n_jobs} jobs, {n_machines} machines FJSP instance")

# 创建环境
env = FJSPEnvironment(n_jobs, n_machines, operations)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 创建智能体
agent = DQNAgent(state_size, action_size)

# 训练参数调整
n_episodes = 3000  # 增加训练轮数
batch_size = 64  # 增大批次大小提高稳定性
test_freq = 20  # 减少评估频率
best_makespan = float('inf')
best_schedule = None
makespans = []
avg_makespans = []  # 用于平滑的平均完工时间
losses = []  # 记录损失
episode_rewards = [] # 新增：记录每轮的总奖励
epsilon_values = [] # 新增：记录每轮的探索率

# 收敛指标 - 更合理的设置
convergence_window = 50  # 检查连续50次评估的表现
early_stopping_patience = 200  # 减少提前停止的耐心值
target_performance = None  # 将在前100轮训练后动态设置
min_epsilon = 0.1  # 最小探索率

last_improvement = 0
window_makespans = []

# 在训练循环开始前添加探索率衰减逻辑
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995  # 每轮衰减率

# 训练循环
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    episode_losses = []
    
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
        
        # 选择动作
        action = agent.act(state, valid_actions, epsilon)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 增强奖励信号
        if done and max(env.machine_available_time) <= 60:  # 如果完成且makespan很好
            reward += 500  # 额外奖励
        
        # 修改奖励函数 - 更加渐进的奖励
        if done:
            makespan = max(env.machine_available_time)
            # 渐进式奖励，根据完工时间给予不同程度的奖励
            if makespan <= 50:
                reward += 1000  # 非常出色的调度
            elif makespan <= 60:
                reward += 500  # 很好的调度
            elif makespan <= 80:
                reward += 200  # 良好的调度
            elif makespan <= 90:
                reward += 10  # 可接受的调度
        
        # 存储经验
        agent.remember(state, action, reward, next_state, done)
        
        # 训练网络
        if len(agent.memory.buffer) > batch_size:
            loss = agent.replay(batch_size)
            episode_losses.append(loss)
        
        state = next_state
        total_reward += reward
    
    # 记录本轮损失
    avg_episode_loss = np.mean(episode_losses) if episode_losses else 0 # Handle case with no losses
    losses.append(avg_episode_loss)
    episode_rewards.append(total_reward) # 新增：记录当前轮的总奖励
    
    # 计算调度的完工时间
    makespan = max(env.machine_available_time)
    makespans.append(makespan)
    
    # 更新最佳调度
    if makespan < best_makespan:
        best_makespan = makespan
        best_schedule = env.schedule.copy()
        last_improvement = episode
    
    # 维护窗口平均值用于评估收敛性
    window_makespans.append(makespan)
    if len(window_makespans) > convergence_window:
        window_makespans.pop(0)
    
    # 计算平滑的平均完工时间
    if episode % test_freq == 0:
        # 评估模式：禁用探索，多次测试取平均
        eval_makespans = []
        for _ in range(5):  # 每次评估运行5次
            eval_state = env.reset()
            eval_done = False
            while not eval_done:
                # 纯贪婪策略
                eval_valid_actions = []
                for op_idx in env.unscheduled_ops:
                    job_idx, op_pos = env._get_job_op_from_idx(op_idx)
                    op_options = env.operations[job_idx][op_pos]
                    valid_machines = [opt[0] for opt in op_options]
                    
                    for machine_idx in valid_machines:
                        action = op_idx * env.n_machines + machine_idx
                        eval_valid_actions.append(action)
                
                # 修正评估时的动作选择逻辑 - 使用纯贪婪策略
                with torch.no_grad():
                    q_values = agent.q_network(torch.FloatTensor(eval_state.reshape(1, -1)).to(agent.device)).cpu().numpy()[0]
                    # 仅考虑有效动作
                    valid_q_values = {a: q_values[a] for a in eval_valid_actions}
                    eval_action = max(valid_q_values, key=valid_q_values.get)
                
                eval_next_state, _, eval_done, _ = env.step(eval_action)
                eval_state = eval_next_state
            
            eval_makespan = max(env.machine_available_time)
            eval_makespans.append(eval_makespan)
        
        avg_makespan = np.mean(eval_makespans)
        avg_makespans.append(avg_makespan)
        
        # 动态设置目标完工时间
        if episode == 100 and target_performance is None:
            best_so_far = min(avg_makespans)
            target_performance = max(best_so_far * 0.9, 8)  # 设置为当前最佳的90%或8，取较大值
            print(f"Target performance dynamically set to: {target_performance}")
        
        # 打印进度
        print(f"Episode {episode}/{n_episodes}, Makespan: {makespan}, Eval Avg: {avg_makespan:.1f}, Best: {best_makespan}")
        
        # 检查是否满足收敛条件
        if target_performance is not None and avg_makespan <= target_performance:
            print(f"Target performance reached at episode {episode}!")
            if episode > convergence_window and all(m <= target_performance * 1.1 for m in avg_makespans[-10:]):
                print("Model converged to target performance. Stopping early.")
                break
        
        # 提前停止条件
        if episode - last_improvement > early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} episodes. Stopping early.")
            break
    
    # 学习率调度
    if episode % 300 == 0 and episode > 0:
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] *= 0.95
        print(f"Learning rate adjusted to {agent.optimizer.param_groups[0]['lr']:.6f}")
    
    # 在每轮结束时更新探索率
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon_values.append(epsilon) # 新增：记录当前轮的探索率
    if episode % test_freq == 0:
        print(f"Current exploration rate (epsilon): {epsilon:.4f}")

# 输出最佳调度结果
print("\nBest Scheduling Result:")
print(f"Minimum Makespan: {best_makespan}")
print("\nDetailed Schedule:")
for job_idx, op_pos, machine_idx, start_time, completion_time in best_schedule:
    print(f"Job {job_idx+1}, Operation {op_pos+1}, Machine {machine_idx+1}, Start: {start_time}, End: {completion_time}")

# 设置matplotlib使用不需要中文支持的字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 绘制学习曲线（多子图版本）
fig, axs = plt.subplots(2, 2, figsize=(18, 12)) # 修改：增加子图数量和画布大小

# 完工时间变化图
axs[0, 0].plot(makespans, 'b-', alpha=0.3, label='Raw Makespan')
if avg_makespans:
    eval_indices = np.arange(0, len(makespans), test_freq)[:len(avg_makespans)]
    axs[0, 0].plot(eval_indices, avg_makespans, 'r-', label='Evaluation Avg Makespan')
axs[0, 0].set_title('Makespan Changes During Training')
axs[0, 0].set_xlabel('Episodes')
axs[0, 0].set_ylabel('Makespan')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 损失变化图
if losses:
    axs[0, 1].plot(losses, 'g-')
    axs[0, 1].set_title('Training Loss Over Episodes')
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)

# 每轮总奖励图
if episode_rewards:
    axs[1, 0].plot(episode_rewards, 'm-')
    axs[1, 0].set_title('Total Reward Per Episode')
    axs[1, 0].set_xlabel('Episodes')
    axs[1, 0].set_ylabel('Total Reward')
    axs[1, 0].grid(True)

# Epsilon 衰减图
if epsilon_values:
    axs[1, 1].plot(epsilon_values, 'c-')
    axs[1, 1].set_title('Epsilon Decay Over Episodes')
    axs[1, 1].set_xlabel('Episodes')
    axs[1, 1].set_ylabel('Epsilon Value')
    axs[1, 1].grid(True)


plt.tight_layout()
plt.savefig('learning_curves_detailed.png') # 修改：保存文件名
plt.show()

# 绘制甘特图
plt.figure(figsize=(12, 8))
colors = plt.cm.jet(np.linspace(0, 1, n_jobs))

# 创建机器标签，从1开始
machine_labels = [f"Machine {i+1}" for i in range(n_machines)]

for job_idx, op_pos, machine_idx, start_time, completion_time in best_schedule:
    duration = completion_time - start_time
    plt.barh(machine_idx, duration, left=start_time, height=0.5,
             color=colors[job_idx], edgecolor='black')
    plt.text(start_time + duration/2, machine_idx,
             f"J{job_idx+1}-O{op_pos+1}", ha='center', va='center')

plt.grid(axis='x')
plt.xlabel('Time')
plt.yticks(range(n_machines), machine_labels)  # 设置y轴标签为机器名称
plt.title('Best Schedule Gantt Chart')
plt.savefig('best_schedule_gantt.png')
plt.show()