import numpy as np
import gym
from gym import spaces

class FJSPEnvironment(gym.Env):
    """柔性作业车间调度问题的强化学习环境"""
    
    def __init__(self, n_jobs, n_machines, operations):
        super().__init__()
        
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.operations = operations
        
        # 计算总工序数
        self.total_ops = sum(len(job) for job in operations)
        
        # 动作空间: 选择(工序, 机器)对，动作空间大小等于总工序数乘以机器数（这里动作空间肯定是大了，因为每个工序不是所有的机器都可以加工的）
        self.action_space = spaces.Discrete(self.total_ops * self.n_machines)
        
        # 观测空间
        # 每个工序的状态: [已调度(0/1), 剩余工序数, 前一道工序完成时间]
        # 每台机器的状态: [可用时间]
        obs_dim = self.total_ops * 3 + self.n_machines
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 初始化环境状态
        self.reset()
        
    def reset(self):
        """重置环境状态"""
        # 作业状态: 当前工序索引。创建一个列表，长度为作业数，每个元素初始化为0，表示每个作业的第一道工序未完成
        self.job_status = [0] * self.n_jobs
        
        # 机器状态: 每台机器的可用时间
        self.machine_available_time = [0] * self.n_machines
        
        # 工序完成时间
        self.operation_completion_time = {}
        
        # 当前调度
        self.schedule = []
        
        # 待调度工序
        self.unscheduled_ops = self._get_available_operations()
        
        # 当前时间
        self.current_time = 0
        
        # 是否结束
        self.done = False
        
        return self._get_observation()
        
    def step(self, action):
        """执行一步调度"""
        # 解码动作
        op_idx, machine_idx = self._decode_action(action)
        
        if op_idx not in self.unscheduled_ops:
            # 无效动作，给予惩罚
            reward = -100
            return self._get_observation(), reward, self.done, {}
        
        # 执行调度
        # 通过工序索引获取作业索引和工序位置
        job_idx, op_pos = self._get_job_op_from_idx(op_idx)
        
        # 获取工序选项
        # 这里假设operations是一个列表，包含每个作业的工序选项，每个工序选项是一个元组(机器索引, 加工时间)
        # 例如：operations = [[(0, 2), (1, 3)], [(0, 4), (1, 1)]]表示作业1有两道工序，作业2也有两道工序
        op_options = self.operations[job_idx][op_pos] 
        
        # 检查所选机器是否可以处理该工序
        valid_machines = [opt[0] for opt in op_options]
        if machine_idx not in valid_machines:
            # 无效机器选择，给予惩罚
            reward = -100
            return self._get_observation(), reward, self.done, {}
        
        # 获取加工时间
        proc_time = None
        for m, p_time in op_options:
            if m == machine_idx:
                proc_time = p_time
                break
        
        # 计算工序开始时间
        if op_pos == 0:  # 作业的第一道工序
            start_time = self.machine_available_time[machine_idx]
        else:
            prev_op_key = (job_idx, op_pos - 1)
            prev_completion_time = self.operation_completion_time.get(prev_op_key, 0)
            start_time = max(prev_completion_time, self.machine_available_time[machine_idx])
        
        # 更新机器可用时间
        completion_time = start_time + proc_time
        self.machine_available_time[machine_idx] = completion_time
        
        # 记录工序完成时间
        op_key = (job_idx, op_pos)
        self.operation_completion_time[op_key] = completion_time
        
        # 更新作业状态
        self.job_status[job_idx] += 1
        
        # 记录调度
        self.schedule.append((job_idx, op_pos, machine_idx, start_time, completion_time))
        
        # 更新未调度工序
        self.unscheduled_ops.remove(op_idx)
        self.unscheduled_ops.extend(self._get_newly_available_ops(job_idx))
        
        # 更新当前时间
        self.current_time = max(self.current_time, completion_time)
        
        # 检查是否所有工序都已调度
        self.done = len(self.unscheduled_ops) == 0
        
        # 计算奖励
        reward = self._calculate_reward(completion_time)
        
        return self._get_observation(), reward, self.done, {}
    
    def _decode_action(self, action):
        """解码动作为(工序索引, 机器索引)"""
        op_idx = action // self.n_machines      # action整除机器数得到工序索引 
        machine_idx = action % self.n_machines  # action除以机器数取余得到机器索引
        return op_idx, machine_idx
    
    def _get_job_op_from_idx(self, op_idx):
        """从工序索引获取(作业索引, 作业内工序位置)"""
        count = 0
        for j, job in enumerate(self.operations):
            for op in range(len(job)):
                if count == op_idx:
                    return j, op
                count += 1
        return -1, -1
    
    def _get_available_operations(self):
        """获取当前可调度的工序"""
        available_ops = []
        op_idx = 0
        
        for job_idx, job_ops in enumerate(self.operations):
            if self.job_status[job_idx] < len(job_ops):
                # 如果是第一道工序或前一道工序已完成
                if self.job_status[job_idx] == 0 or \
                   (job_idx, self.job_status[job_idx] - 1) in self.operation_completion_time:
                    available_ops.append(op_idx)
            
            op_idx += len(job_ops)
            
        return available_ops
    
    def _get_newly_available_ops(self, job_idx):
        """获取新可用的工序"""
        newly_available = []
        
        # 首先检查完成作业的下一道工序是否可用
        if self.job_status[job_idx] < len(self.operations[job_idx]):
            op_idx = 0
            for j, job_ops in enumerate(self.operations):
                if j == job_idx and self.job_status[j] < len(job_ops):
                    newly_available.append(op_idx + self.job_status[j])
                op_idx += len(job_ops)
        
        return newly_available
    
    def _get_observation(self):
        """获取环境观测"""
        obs = []
        
        # 工序状态
        op_idx = 0
        for job_idx, job_ops in enumerate(self.operations):
            for op_pos in range(len(job_ops)):
                # 是否已调度
                scheduled = 1 if self.job_status[job_idx] > op_pos else 0
                # 剩余工序数
                remaining_ops = len(job_ops) - op_pos - scheduled
                # 前一道工序完成时间
                prev_completion_time = 0
                if op_pos > 0:
                    prev_op_key = (job_idx, op_pos - 1)
                    prev_completion_time = self.operation_completion_time.get(prev_op_key, 0)
                
                obs.extend([scheduled, remaining_ops, prev_completion_time])
                op_idx += 1
        
        # 机器状态
        obs.extend(self.machine_available_time)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, completion_time):
        """计算奖励函数"""
        # 计算当前makespan
        current_makespan = max(self.machine_available_time)
        
        if self.done:
            # 完成所有工序，给予强化奖励
            return 600 - current_makespan * 5  # 完成奖励，makespan越小奖励越大
        
        # 计算完成的工序百分比
        completed_ops = sum(self.job_status)
        completion_percentage = completed_ops / self.total_ops
        
        # 计算资源利用率
        idle_time = sum(max(self.machine_available_time) - t for t in self.machine_available_time)
        utilization = 1.0 - (idle_time / (self.n_machines * current_makespan) if current_makespan > 0 else 0)
        
        # 组合奖励
        progress_reward = completion_percentage * 20.0  # 进度奖励
        time_reward = -20.0 * current_makespan  # 时间惩罚
        balance_reward = utilization * 80.0  # 利用率奖励
        
        return progress_reward + time_reward + balance_reward
    
    def render(self, mode='human'):
        """渲染当前调度"""
        if mode == 'human':
            print("Current schedule:")
            for job_idx, op_pos, machine_idx, start_time, completion_time in self.schedule:
                print(f"Job {job_idx+1}, Op {op_pos+1}, Machine {machine_idx+1}, Start: {start_time}, Complete: {completion_time}")
            print(f"Current makespan: {max(self.machine_available_time)}")
        
        return None