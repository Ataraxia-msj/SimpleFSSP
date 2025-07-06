import numpy as np

def load_fjsp_instance(file_path):
    """
    加载FJSP实例数据
    
    Args:
        file_path: FJSP文件路径
        
    Returns:
        n_jobs: 作业数
        n_machines: 机器数
        operations: 工序数据，包含每个工序可以在哪些机器上加工及加工时间
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 去除空行和注释行
        content_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//'):
                content_lines.append(line)
        
        if not content_lines:
            raise ValueError(f"文件 {file_path} 没有有效内容")
        
        # 解析第一行获取基本信息
        parts = content_lines[0].strip().split()
        if len(parts) < 3:
            raise ValueError(f"文件格式错误: 第一行应包含至少3个数字")
            
        n_jobs = int(parts[0])
        n_machines = int(parts[1])
        
        # 检查文件行数是否足够
        if len(content_lines) < n_jobs + 1:
            raise ValueError(f"文件格式错误: 行数不足，应至少有 {n_jobs + 1} 行")
        
        operations = []
        for i in range(1, n_jobs + 1):
            if i >= len(content_lines):
                raise ValueError(f"文件格式错误: 缺少作业 {i} 的数据")
                
            job_ops = []
            line_parts = content_lines[i].strip().split()
            
            if not line_parts:
                raise ValueError(f"文件格式错误: 作业 {i} 的行为空")
                
            idx = 0
            if idx >= len(line_parts):
                raise ValueError(f"文件格式错误: 作业 {i} 缺少工序数量信息")
                
            n_operations = int(line_parts[idx])
            idx += 1
            
            for j in range(n_operations):
                if idx >= len(line_parts):
                    raise ValueError(f"文件格式错误: 作业 {i} 的工序 {j+1} 缺少可选机器数量")
                    
                n_options = int(line_parts[idx])
                idx += 1
                op_options = []
                
                for k in range(n_options):
                    if idx + 1 >= len(line_parts):
                        raise ValueError(f"文件格式错误: 作业 {i} 的工序 {j+1} 的第 {k+1} 个机器选项数据不完整")
                    
                    try:
                        machine = int(line_parts[idx]) - 1  # 机器编号从0开始
                        idx += 1
                        proc_time = int(line_parts[idx])
                        idx += 1
                    except ValueError:
                        raise ValueError(f"文件格式错误: 作业 {i} 的工序 {j+1} 的第 {k+1} 个机器选项数据无效")
                        
                    op_options.append((machine, proc_time))
                
                job_ops.append(op_options)
            
            operations.append(job_ops)
        
        return n_jobs, n_machines, operations
        
    except Exception as e:
        print(f"加载FJSP数据时出错: {str(e)}")
        print(f"请检查文件 {file_path} 是否存在且格式正确")
        raise