from gurobipy import *
import itertools

# ------------------------------
# Input: Job → List of Operations → List of (Machine, ProcessingTime)
# ------------------------------
jobs = [
    [  # Job 0
        [(0, 5), (2, 4)],         # Operation 0: machine 0 or 2
        [(4, 3), (2, 5), (1, 1)]  # Operation 1: machine 4, 2, or 1
    ],
    [  # Job 1
        [(1, 6)],                 # Operation 0: machine 1
        [(2, 1)]                  # Operation 1: machine 2
    ],
    [  # Job 2
        [(1, 6)],                 # Operation 0: machine 1
        [(2, 4), (5, 2)]          # Operation 1: machine 2 or 5
    ]
]

num_jobs = len(jobs)
num_machines = 6  # Machines: 0 to 5
M = 1e5

model = Model("Flexible_FSSP")

# ------------------------------
# Decision Variables
# ------------------------------
x = {}   # x[j,o,m] = 1 if operation o of job j assigned to machine m
s = {}   # s[j,o] = start time of operation o of job j

for j, job in enumerate(jobs):
    for o, ops in enumerate(job):
        s[j, o] = model.addVar(lb=0, name=f"s_{j}_{o}")
        for m, t in ops:
            x[j, o, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")

Cmax = model.addVar(lb=0, name="Cmax")

# ------------------------------
# Constraints
# ------------------------------

# Each operation assigned to exactly one machine
for j, job in enumerate(jobs):
    for o, ops in enumerate(job):
        model.addConstr(quicksum(x[j, o, m] for m, _ in ops) == 1)

# Technological order (within-job precedence constraints)
for j, job in enumerate(jobs):
    for o in range(len(job) - 1):
        curr_ops = job[o]
        next_ops = job[o + 1]
        model.addConstr(
            s[j, o] + quicksum(t * x[j, o, m] for m, t in curr_ops) <= s[j, o + 1]
        )

# Disjunctive constraints: no overlap on the same machine
for m in range(num_machines):
    # Find all operations that can be processed on machine m
    ops_on_m = [(j, o, t) for j, job in enumerate(jobs)
                          for o, ops in enumerate(job)
                          for m2, t in ops if m2 == m]

    for (j1, o1, t1), (j2, o2, t2) in itertools.combinations(ops_on_m, 2):
        if (j1, o1) == (j2, o2): continue

        # Add a disjunctive constraint using binary variable y
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{o1}_{j2}_{o2}_{m}")

        model.addConstr(
            s[j1, o1] + t1 <= s[j2, o2] + M * (1 - y)
        )
        model.addConstr(
            s[j2, o2] + t2 <= s[j1, o1] + M * y
        )

        # Only valid if both ops are assigned to machine m
        model.addConstr(
            x[j1, o1, m] + x[j2, o2, m] >= 2 * y
        )

# Cmax constraint: ensure Cmax ≥ finish time of each job
for j, job in enumerate(jobs):
    o = len(job) - 1
    model.addConstr(
        Cmax >= s[j, o] + quicksum(t * x[j, o, m] for m, t in job[o])
    )

# ------------------------------
# Objective: minimize Cmax
# ------------------------------
model.setObjective(Cmax, GRB.MINIMIZE)

# ------------------------------
# Solve model
# ------------------------------
model.optimize()

# ------------------------------
# Output results
# ------------------------------
if model.status == GRB.OPTIMAL:
    print(f"\n✅ Optimal Cmax: {Cmax.X:.2f}\n")
    for j, job in enumerate(jobs):
        for o, ops in enumerate(job):
            start = s[j, o].X
            for m, _ in ops:
                if x[j, o, m].X > 0.5:
                    print(f"Job {j} Operation {o} assigned to Machine {m+1} starts at {start:.2f}")
else:
    print("❌ No optimal solution found.")
