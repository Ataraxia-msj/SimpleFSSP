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
n_episodes = 2000  # 增加训练轮数
batch_size = 32  # 减小批次大小，更频繁更新
test_freq = 10  # 每隔多少轮评估一次
best_makespan = float('inf')
best_schedule = None
makespans = []
avg_makespans = []  # 用于平滑的平均完工时间
losses = []  # 记录损失

# 收敛指标
convergence_window = 50  # 检查连续50次评估的表现
target_performance = 8  # 目标完工时间
early_stopping_patience = 200  # 如果连续200轮没有改进，提前停止

last_improvement = 0
window_makespans = []

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
        action = agent.act(state, valid_actions)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 增强奖励信号
        if done and max(env.machine_available_time) <= 7:  # 如果完成且makespan很好
            reward += 50  # 额外奖励
        
        # 存储经验
        agent.remember(state, action, reward, next_state, done)
        
        # 训练网络
        if len(agent.memory.buffer) > batch_size:
            loss = agent.replay(batch_size)
            episode_losses.append(loss)
        
        state = next_state
        total_reward += reward
    
    # 记录本轮损失
    if episode_losses:
        losses.append(np.mean(episode_losses))
    
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
                
                with torch.no_grad():
                    eval_action = agent.act(eval_state, eval_valid_actions)
                    eval_action = np.argmax(agent.q_network(torch.FloatTensor(eval_state.reshape(1, -1)).to(agent.device)).cpu().numpy()[0])
                    # 确保选择的动作有效
                    if eval_action not in eval_valid_actions:
                        eval_action = np.random.choice(eval_valid_actions)
                
                eval_next_state, _, eval_done, _ = env.step(eval_action)
                eval_state = eval_next_state
            
            eval_makespan = max(env.machine_available_time)
            eval_makespans.append(eval_makespan)
        
        avg_makespan = np.mean(eval_makespans)
        avg_makespans.append(avg_makespan)
        
        # 打印进度
        print(f"Episode {episode}/{n_episodes}, Makespan: {makespan}, Eval Avg: {avg_makespan:.1f}, Best: {best_makespan}")
        
        # 检查是否满足收敛条件
        if avg_makespan <= target_performance:
            print(f"Target performance reached at episode {episode}!")
            if episode > convergence_window and all(m <= target_performance * 1.1 for m in avg_makespans[-10:]):
                print("Model converged to target performance. Stopping early.")
                break
        
        # 提前停止条件
        if episode - last_improvement > early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} episodes. Stopping early.")
            break
    
    # 学习率调度
    if episode % 200 == 0 and episode > 0:
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] *= 0.9
        print(f"Learning rate adjusted to {agent.optimizer.param_groups[0]['lr']:.6f}")

# 输出最佳调度结果
print("\nBest Scheduling Result:")
print(f"Minimum Makespan: {best_makespan}")
print("\nDetailed Schedule:")
for job_idx, op_pos, machine_idx, start_time, completion_time in best_schedule:
    print(f"Job {job_idx+1}, Operation {op_pos+1}, Machine {machine_idx+1}, Start: {start_time}, End: {completion_time}")

# 设置matplotlib使用不需要中文支持的字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 绘制学习曲线（多子图版本）
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# 完工时间变化图
axs[0].plot(makespans, 'b-', alpha=0.3, label='Raw')
if avg_makespans:
    eval_indices = np.arange(0, len(makespans), test_freq)[:len(avg_makespans)]
    axs[0].plot(eval_indices, avg_makespans, 'r-', label='Evaluation Avg')
axs[0].set_title('Makespan Changes During Training')
axs[0].set_xlabel('Episodes')
axs[0].set_ylabel('Makespan')
axs[0].legend()
axs[0].grid(True)

# 损失变化图
if losses:
    axs[1].plot(losses)
    axs[1].set_title('Training Loss')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

plt.tight_layout()
plt.savefig('learning_curves.png')
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