from dss import *

def main():
    """
    DSS优化算法主程序
    
    演示如何使用DSS算法优化量子测量电路配置,
    以最小化测量Pauli算符集合的总成本。
    """
    # ========================================================================
    # 第一步: 定义问题参数
    # ========================================================================
    NUM_QUBITS = 4          # 量子比特数
    CIRCUIT_DEPTH = 1       # 电路深度(双比特门层数)
    NUM_CIRCUITS = 10       # 测量电路数量
    EPSILON = 4           # 成本函数超参数
    
    # print("\n" + "="*70)
    # print("  DSS 量子测量优化程序".center(70))
    # print("="*70)
    # print(f"\n  📋 问题参数:")
    # print(f"     量子比特数     : {NUM_QUBITS}")
    # print(f"     电路深度       : {CIRCUIT_DEPTH}")
    # print(f"     测量电路数     : {NUM_CIRCUITS}")
    # print(f"     超参数 ε       : {EPSILON}")
    
    # ========================================================================
    # 第二步: 创建Pauli算符集合
    # ========================================================================
    # print(f"\n{'-'*70}")
    # print("  构建 Pauli 算符集合")
    # print(f"{'-'*70}")
    
    # 初始化Pauli集合
    paulis = PauliOperatorCollection(num_qubits=NUM_QUBITS)
    
    # 添加需要测量的Pauli算符
    # 注: 在实际应用中,这些算符通常来自量子化学哈密顿量或其他物理问题
    pauli_strings = [
        ('XXYY', 1.0),
        ('YYZZ', 1.0),
        ('ZZXX', 1.0),
    ]
    
    # print(f"  ⏳ 添加 Pauli 算符...")
    for pauli_str, weight in pauli_strings:
        paulis.add_from_string(pauli_str, weight=weight)
        # print(f"     ✓ {pauli_str} (权重: {weight})")
    
    # 显示Pauli集合摘要
    # print(f"\n  📊 Pauli 算符集合摘要:")
    # print(f"     算符数量: {len(paulis)}")
    # print(f"     量子比特: {paulis.num_qubits}")
    # print(f"\n  详细列表:")
    # for i, op in enumerate(paulis):
    #     print(f"     [{i}] {op.to_string()} (w={op.weight})")
    
    # ========================================================================
    # 第三步: 初始化并运行DSS优化
    # ========================================================================
    # print(f"\n{'='*70}")
    
    # 创建DSS优化器
    dss_optimizer = DSS(
        pauli_collection=paulis,
        circuit_depth=CIRCUIT_DEPTH,
        num_measurements=NUM_CIRCUITS,
        epsilon=EPSILON
    )
    
    # 运行优化算法
    optimized_circuits = dss_optimizer.run()
    
    # ========================================================================
    # 第四步: 保存优化结果
    # ========================================================================
    # print(f"\n{'='*70}")
    # print("  保存优化结果".center(70))
    # print(f"{'='*70}")
    
    output_folder = "results"
    dss_optimizer.save_results(folder_name=output_folder)
    
    # ========================================================================
    # 第五步: 结果分析和展示
    # ========================================================================
    # print(f"\n{'='*70}")
    # print("  优化结果分析".center(70))
    # print(f"{'='*70}")
    
    final_cost = dss_optimizer.cost_calculator.cost_function_value
    
    print(f"\n  🎯 总体优化结果:")
    print(f"     优化电路数     : {len(optimized_circuits)}")
    print(f"     最终成本函数值 : {final_cost:.6e}")
    
    # -------------------- 分析第一个电路 --------------------
    print(f"\n{'-'*70}")
    print("  电路 1 详细配置".center(70))
    print(f"{'-'*70}")
    
    first_circuit = optimized_circuits[0]
    
    # --- 单比特门统计 ---
    print(f"\n  🔹 单比特门 (G1):")
    g1_gates = []
    g1_stats = {}  # 统计各类门的数量
    
    for (r, q), gate_type in first_circuit.iter_gate1q():
        if gate_type not in [0, 1]:  # 排除随机门和单位门
            gate_name = MatConfig1.INT_TO_GATE_1.get(gate_type, 'Unknown')
            g1_gates.append((r, q, gate_name))
            g1_stats[gate_name] = g1_stats.get(gate_name, 0) + 1
    
    if g1_gates:
        print(f"     确定性门数量: {len(g1_gates)}")
        print(f"     门类型分布:")
        for gate_name, count in sorted(g1_stats.items()):
            print(f"       • {gate_name}: {count} 个")
        
        print(f"\n     详细位置:")
        for r, q, gate_name in g1_gates:
            print(f"       Layer {r}, Qubit {q} → {gate_name}")
    else:
        print(f"     (无确定性门)")
    
    # --- 双比特门统计 ---
    print(f"\n  🔸 双比特门 (G2):")
    g2_gates = []
    g2_stats = {}  # 统计各类门的数量
    
    for (r, q), gate_type in first_circuit.iter_gate2q():
        if gate_type not in [0, 1]:  # 排除随机门和单位门
            gate_name = MatConfig2.INT_TO_GATE_2.get(gate_type, 'Unknown')
            q2 = (q + 1) % first_circuit.num_qubits
            g2_gates.append((r, q, q2, gate_name))
            g2_stats[gate_name] = g2_stats.get(gate_name, 0) + 1
    
    if g2_gates:
        print(f"     确定性门数量: {len(g2_gates)}")
        print(f"     门类型分布:")
        for gate_name, count in sorted(g2_stats.items()):
            print(f"       • {gate_name}: {count} 个")
        
        print(f"\n     详细位置:")
        for r, q, q2, gate_name in g2_gates:
            print(f"       Layer {r}, Qubits ({q},{q2}) → {gate_name}")
    else:
        print(f"     (无确定性门)")
    
    # -------------------- 所有电路的总体统计 --------------------
    print(f"\n{'-'*70}")
    print("  所有电路统计摘要".center(70))
    print(f"{'-'*70}")
    
    total_g1 = 0
    total_g2 = 0
    all_g1_stats = {}
    all_g2_stats = {}
    
    for k, circuit in enumerate(optimized_circuits):
        # 统计单比特门
        for (r, q), gate_type in circuit.iter_gate1q():
            if gate_type not in [0, 1]:
                total_g1 += 1
                gate_name = MatConfig1.INT_TO_GATE_1.get(gate_type, 'Unknown')
                all_g1_stats[gate_name] = all_g1_stats.get(gate_name, 0) + 1
        
        # 统计双比特门
        for (r, q), gate_type in circuit.iter_gate2q():
            if gate_type not in [0, 1]:
                total_g2 += 1
                gate_name = MatConfig2.INT_TO_GATE_2.get(gate_type, 'Unknown')
                all_g2_stats[gate_name] = all_g2_stats.get(gate_name, 0) + 1
    
    print(f"\n  📈 跨所有 {NUM_CIRCUITS} 个电路:")
    print(f"     总确定性单比特门: {total_g1}")
    print(f"     总确定性双比特门: {total_g2}")
    print(f"     总确定性门数    : {total_g1 + total_g2}")
    
    if all_g1_stats:
        print(f"\n     单比特门分布:")
        for gate_name, count in sorted(all_g1_stats.items()):
            percentage = 100 * count / total_g1 if total_g1 > 0 else 0
            print(f"       • {gate_name}: {count} ({percentage:.1f}%)")
    
    if all_g2_stats:
        print(f"\n     双比特门分布:")
        for gate_name, count in sorted(all_g2_stats.items()):
            percentage = 100 * count / total_g2 if total_g2 > 0 else 0
            print(f"       • {gate_name}: {count} ({percentage:.1f}%)")
    
    # ========================================================================
    # 程序结束
    # ========================================================================
    print(f"\n{'='*70}")
    print("  程序执行完成".center(70))
    print(f"{'='*70}")
    print(f"\n  ✅ 所有结果已保存至: {output_folder}/")
    print(f"  📊 查看优化报告: {output_folder}/optimization_report.md")
    print(f"  🖼️  查看电路图像: {output_folder}/svg_circuits/\n")

if __name__ == "__main__":
    main()