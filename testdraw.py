from dss import CircuitConfig
from drawcircuit import draw_circuit
import random

def main():
    N_QUBITS = 8
    DEPTH = 5
    my_config = CircuitConfig(num_qubits=N_QUBITS, circuit_depth=DEPTH)

    # 随机填充单比特门
    for r in range(my_config.circuit_depth + 1):
        for q in range(my_config.num_qubits):
            if random.random() < 0.9:
                my_config.set_gate1q(r, q, random.randint(1, 4))
            # my_config.set_gate1q(r, q, 1)
    
    # 随机填充双比特门
    for r in range(my_config.circuit_depth):
        for q in range(my_config.num_qubits):
            if my_config._is_valid_gate2q_position(r, q):
                my_config.set_gate2q(r, q, 2)

    # 手动设置以确保所有类型都被绘制
    # 常规门
    if my_config._is_valid_gate2q_position(0, 0):
        my_config.set_gate2q(0, 0, 0)  # 类型0
    if my_config._is_valid_gate2q_position(0, 2):
        my_config.set_gate2q(0, 2, 2)  # CNOT
    
    # 常规 SWAP
    if my_config._is_valid_gate2q_position(1, 1):
        my_config.set_gate2q(1, 1, 3)  # SWAP
    if my_config._is_valid_gate2q_position(2, 4):
        my_config.set_gate2q(2, 4, 3)  # SWAP
    
    # PBC CNOT (如果启用了周期边界条件)
    if my_config.pbc and my_config._is_valid_gate2q_position(1, N_QUBITS - 1):
        my_config.set_gate2q(1, N_QUBITS - 1, 2)
    
    # PBC SWAP
    if my_config.pbc and my_config._is_valid_gate2q_position(3, N_QUBITS - 1):
        my_config.set_gate2q(3, N_QUBITS - 1, 3)
    
    # PBC 锁定门
    if my_config.pbc and my_config._is_valid_gate2q_position(2, N_QUBITS - 1):
        my_config.set_gate2q(2, N_QUBITS - 1, 0)

    # 绘制电路
    print(f"正在绘制 {N_QUBITS} qubits, depth {DEPTH} 的电路 (包含测量层)...")
    print(f"周期边界条件: {my_config.pbc}")
    print(f"双比特门总数: {my_config.total_gate2q}")
    draw_circuit(my_config)


if __name__ == "__main__":
    main()