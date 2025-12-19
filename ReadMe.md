# Derandomized Shallow Shadows (DSS) 量子电路优化器

这是一个基于 Python 实现的 **Derandomized Shallow Shadows (DSS)** 算法项目。该项目旨在通过经典贪心算法和张量网络技术，优化浅层量子测量电路的配置，以最小化测量特定 Pauli 算符集合所需的样本复杂度（即最小化预测方差）。

该算法结合了 **quimb** 库进行高效的张量网络收缩，并使用 **matplotlib** 对生成的量子电路进行可视化。

## 项目功能

  * **贪心优化策略**：
      * 采用逐层、逐门的贪心策略优化测量电路。
      * **阶段 1**：优化双比特门（使用 2 维简化基，从右向左优化）。
      * **阶段 2**：优化单比特门（使用 4 维完整 Pauli 基，从左向右优化）。
  * **高效张量网络计算**：
      * 将量子电路演化和 Pauli 权重计算映射为张量网络。
      * 利用 `quimb` 库实现高效的张量收缩，避免了全希尔伯特空间模拟的指数级开销。
      * 支持 Brickwall 电路结构。
  * **灵活的门配置**：
      * 支持多种单比特门：H, S, HS, SH, HSH 等。
      * 支持多种双比特门：CNOT, SWAP, Identity。
      * 支持周期性边界条件 (PBC)。
  * **电路可视化**：
      * 自动生成 SVG 格式的量子电路图。
      * 清晰展示单比特门、双比特门连线及测量操作。
  * **报告生成**：
      * 自动生成 Markdown 格式的优化报告，包含成本函数变化和详细的门配置。

## 环境依赖

请确保你的 Python 环境安装了以下依赖库：

```bash
pip install numpy matplotlib quimb
```

## 文件结构

  * **`dss.py`**: 核心逻辑文件。包含了张量网络构建、成本函数计算、DSS 贪心优化算法以及主程序入口。
  * **`drawcircuit.py`**: 绘图工具文件。负责将优化后的电路配置绘制成可视化图像。

## 快速开始

你可以直接运行 `dss.py` 来执行一个示例优化任务：

```bash
python dss.py
```

### 自定义使用示例

如果你想在自己的脚本中使用该库，可以参考以下代码：

```python
from dss import DSS, PauliOperatorCollection

# 1. 定义问题参数
NUM_QUBITS = 4          # 量子比特数
CIRCUIT_DEPTH = 1       # 电路深度
NUM_CIRCUITS = 10       # 测量电路数量
EPSILON = 0.1           # 成本函数超参数

# 2. 构建需要测量的 Pauli 算符集合
paulis = PauliOperatorCollection(num_qubits=NUM_QUBITS)
paulis.add_from_string('XXYY', weight=1.0)
paulis.add_from_string('YYZZ', weight=1.0)
paulis.add_from_string('ZZXX', weight=1.0)

# 3. 初始化优化器
dss_optimizer = DSS(
    pauli_collection=paulis,
    circuit_depth=CIRCUIT_DEPTH,
    num_measurements=NUM_CIRCUITS,
    epsilon=EPSILON
)

# 4. 运行优化
print("开始优化...")
optimized_circuits = dss_optimizer.run()

# 5. 保存结果 (图像和报告)
output_folder = "my_optimization_results"
dss_optimizer.save_results(folder_name=output_folder)

print(f"优化完成，结果已保存至 {output_folder}")
```

## 算法细节

### 1\. 成本函数 (Cost Function)

算法旨在最小化以下形式的成本函数（Shadow Norm 的置信度界）：

$
\text{Cost} = \sum_{P} w_P \prod_{k=1}^{N} \exp\left(-\frac{\epsilon^2}{2} p_k(P)\right)
$
其中 $p_k(P)$ 是第 $k$ 个测量电路能够成功“击中”（即在计算基下对角化）Pauli 算符 $P$ 的概率。

### 2\. 计算模式 (Modes)

为了平衡精度和速度，算法使用了两种张量网络表示模式：

* **Mode 1 (4维完整表示)**：用于优化单比特门。精确模拟 Pauli 算符在 Clifford 门下的演化。
* **Mode 2 (2维简化表示)**：用于优化双比特门。忽略具体的 Pauli 类型（X/Y/Z），仅关注算符是否为 Identity，利用“Clifford 平均”特性加速计算。

### 3\. 优化流程

算法初始化 $N$ 个随机 Clifford 电路，然后对每个电路进行两轮扫描：

1.  **双比特门优化**：使用 Mode 2，遍历所有双比特门位置（CNOT, SWAP, I），选择使成本函数下降最多的门。
2.  **单比特门优化**：使用 Mode 1，固定双比特门后，遍历所有单比特门位置，选择最优的 Clifford 旋转门。

## 输出结果

运行结束后，程序会在指定文件夹生成：

* **`svg_circuits/`**: 包含所有优化后电路的 SVG 矢量图。
* **`optimization_report.md`**: 包含优化参数、最终成本值以及每个电路详细门配置的报告。

## 注意事项

* 量子比特数较大时（例如 \> 20），张量网络收缩可能会消耗较多内存和时间。
* 电路深度 `circuit_depth` 指的是双比特门的层数。
* 当 `num_qubits` 为偶数时，代码默认启用周期性边界条件（闭环结构）。
