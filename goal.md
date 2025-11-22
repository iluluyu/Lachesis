# Derandomized Quantum Circuit (去随机化量子线路)

我们在 Shallow Classical Shadow 的基础上，对相关的随机门换成简单的门，以提高我们对特定算符的测量置信度。

Shallow Classical Shadow 由有限制深度（例如 $d \sim \mathcal{O}(\log n)$）的 Brickwall 形式的线路构成。为了让其在我们的实验中更加顺利的进行，我们在每一层 2-qubit 门的前后加了一层 1-qubit 的门，以便于更快的优化。

我们实际是关心一组算符 $\{P_k\}_{k=1}^M$，其中每个算符 $P_k$ 都是 Pauli 算符，$P_k \in \{I, X, Y, Z\}^{\otimes n}$。去随机的步骤就是优化线路的参数使得

一个特定的**成本函数 (COST Function)** 得到最小化。这里的**COST Function**为测量任务的置信度。

## 优化问题目标 (Objective Function)

优化的目标是最小化以下成本函数 $COST$。这个函数是基于我们希望估计的**所有** Pauli 算符 $P$ 的“置信度”构建的。

其形式在论文的公式 (1) 和 (A5) 中给出：

$$
COST_{\epsilon}(\{\mathcal{U}_{i}\}_{i=1}^{N})=\sum_{P}w_{P}\prod_{i=1}^{N}\exp\left[-\frac{\epsilon^{2}}{2}p_{i}(P)\right]
$$

这里的核心是最小化这个 $COST$ 值，它对应着**最大化**我们“学习”到（即测量到）我们关心的那组 Pauli 算符的概率。

### 关键参数说明：
* **$\{\mathcal{U}_{i}\}_{i=1}^{N}$**: 代表 $N$ 个测量线路的系综 (ensembles)。在优化开始时，它们是随机的；优化结束后，它们变成确定的电路。
* **$w_{P}$**: 每个 Pauli 算符 $P$ 的**权重**。这代表了它在我们总的估计任务中的重要性。例如，在量子化学问题中，这个权重 $w_P$ 可以设为哈密顿量中该项的系数 $w_P = |c_P|$。
* **$p_{i}(P)$**: 这是最关键的部分，代表第 $i$ 个测量电路（或系综） $\mathcal{U}_i$ **“命中”或“对角化”** 算符 $P$ 的概率。在论文中，这被称为 **"Pauli weight"** (Pauli 权重)。
    * 它的定义为：$p_{i}(P)=\frac{1}{2^{n}}\mathbb{E}_{U\sim\mathcal{U}_{i}}\sum_{b\in\{0,1\}^{n}}\langle b|UPU^{\dagger}|b\rangle^{2}$。

## 优化方法 (Optimization Method)

去随机化的过程在论文的 **Algorithm 1 (DSS 算法)** 中有详细定义。这是一个**贪心算法 (greedy algorithm)**。

优化的具体步骤如下：

1.  **初始化 (Initialization):**
    算法从 $N$ 个完全随机的浅层线路系综 $\{\mathcal{U}_i\}_{i=1}^N$ 开始。每个系综 $\mathcal{U}_i$ 都具有 Brickwall 结构，由 $d$ 层随机双量子比特门和 $d+1$ 层随机单量子比特门交错构成。在这一阶段，所有的门都处于“随机采样”状态（在表 I 中用 '0' 表示）。

2.  **逐门去随机 (Gate-by-gate Derandomization):**
    算法会遍历这 $N$ 个电路中的**每一个门**，逐个将其从“随机”替换为“确定”。

3.  **确定性门选项 (Deterministic Gate Options):**
    对于每一个正在被优化的门，算法会尝试用一小组固定的、确定的门来替换它。这些选项在论文的 Table I 中有定义：
    * **双量子比特门 (2-qubit gates):** 选项为 **{Identity, CNOT, SWAP}**。
    * **单量子比特门 (1-qubit gates):** 选项为 $Cl(2)$ 克利福德群中的 **6 种独立旋转**（例如 $H, S, HS$ 等，它们实现 $X \leftrightarrow Z$, $Y \leftrightarrow X$, $X \to Z \to Y$ 等置换）。

4.  **贪心选择 (Greedy Selection):**
    算法会分别计算将当前门替换为上述每一个确定性选项（例如，CNOT）后的 $COST$ 函数总值。然后，它会“贪心”地选择那个能使 $COST$ **最小化**的门，并将其固定下来 。

5.  **最终输出 (Final Output):**
    算法重复步骤 2-4，直到所有 $N$ 个电路中的所有门都从随机采样变为确定的门。最终的输出是 $N$ 个完全确定的、最优化的浅层测量线路 $\{U_i^{DSS}\}_{i=1}^N$。

### 优化的效率：张量网络 (Tensor Network)

上述贪心算法的每一步都依赖于计算 $COST$ 函数，而计算 $COST$ 函数又依赖于计算 $p_i(P)$。

* **挑战:** 经典计算机上模拟量子线路来计算 $p_i(P)$（即 $U P U^\dagger$）通常需要指数级的资源，这会使算法不可行 。
* **解决方案:** 本文的一个关键贡献是利用**张量网络 (Tensor Network) 技术** 。
* **原理:** 该技术将 Pauli 权重 $p_i(P)$ 的计算（这是一个二次量）映射到一个经典的马尔可夫过程 (Markovian classical processes) 。通过这种方式，算法可以在**多项式时间**内高效地计算出 $COST$ 函数的值（前提是线路深度 $d$ 是有界的，例如 $d = \mathcal{O}(\text{polylog}(n))$。

### Derandomized Shallow Shadows (DSS) 算法详解

DSS 算法的核心思想是，与其使用完全随机的浅层线路（Shallow Shadows）来测量所有的 Pauli 算符，不如**针对性地**设计一组浅层确定性线路，使其对我们**感兴趣的**特定 Pauli 算符 $\{P\}$ 具有最高的测量效率。

这个“设计”的过程，就是一个“去随机化”的优化过程。

### 1\. 算法伪代码 (基于 Algorithm 1)

这里是 DSS 算法的详细中文伪代码，它描述了如何从 $N$ 个随机线路系综 (ensembles) 出发，最终得到 $N$ 个确定的测量线路。

```pseudocode
# --- Derandomized Shallow Shadows (DSS) 算法 ---

# --- 输入 (Input) ---
# N: 测量预算 (总的测量电路数量)
# n: 量子比特数
# {P}: 我们希望估计的 Pauli 算符集合
# d: 测量电路允许的最大深度 (双量子比特门的层数)

# --- 输出 (Output) ---
# {U_i(t_final, s_final)}_i=1^N: N 个确定的、最优化的测量线路

# --- 算法主体 (Algorithm) ---

函数 DSS_Algorithm(N, n, {P}, d):
    
    # 1. 初始化线路结构
    #    创建 N 个测量系综 (ensembles) {U_i}。
    #    每个 U_i 由 t^(i) 和 s^(i) 两个向量定义：
    #    t^(i): 存储所有双量子比特门 (d * floor(n/2) 个)
    #    s^(i): 存储所有单量子比特门 ((d+1) * n 个)
    
    #    将所有门初始化为 "随机" 状态 (在 Table I 中用 '0' 表示)
    对于 i 从 1 到 N:
        t^(i) = 全零向量  # 初始 t[g] = 0 (代表 ~Cl(2^2))
        s^(i) = 全零向量  # 初始 s[g] = 0 (代表 ~Cl(2))
    
    初始化pauliweight矩阵
        pauliweight= np.zeros(pauli_number, ensemble_number)

        对每一个pauli算符， 计算初始线路下的pauli-weight $p_i$
            pauliweight(pauli_index, : ) = p_i
    
    计算初始的cost_function

    # 2. 逐个电路进行去随机化
    对于 j 从 1 到 N:  # 遍历 N 个测量电路中的第 j 个
        
        # 2a. 优化(去随机)所有双量子比特门 顺序：从右往左（从测量到输入）
        对于 g_2qbit 在 第 j 个电路的所有双量子比特门位置:
            
            # 遍历所有确定的双量子比特门选项
            对于 V in {1 (Identity), 2 (SWAP), 3 (CNOT)}:
                
                # 假设将 t^(j)[g_2qbit] 设为 V，并计算总成本
                # 注意：此时 {P} 中的所有 p_j(P) 都需要重新计算
                f(V) = 计算 COST({U_i}_i=1^N | t^(j)[g_2qbit] = V)
                
                如果 f(V) < cost_function:
                    best_cost = f(V)
                    best_gate_option = V
            
            # 贪心选择：永久固定这个门
            t^(j)[g_2qbit] = best_gate_option

        # 2b. 优化(去随机)所有单量子比特门 顺序：从左往右（从输入到测量）
        对于 g_1qbit 在 第 j 个电路的所有单量子比特门位置:
            
            # 遍历所有确定的单量子比特门选项 (6种旋转)
            对于 W in {1, 2, 3, 4, 5, 6}:
                
                # 假设将 s^(j)[g_1qbit] 设为 W，并计算总成本
                f(W) = 计算 COST({U_i}_i=1^N | s^(j)[g_1qbit] = W)
                
                如果 f(W) < best_cost:
                    best_cost = f(W)
                    best_gate_option = W
            
            # 贪心选择：永久固定这个门
            s^(j)[g_1qbit] = best_gate_option

    # 3. 返回最终的确定性电路
    返回 {U_i(t^(i), s^(i))}_i=1^N  # 此时所有 t 和 s 向量都已填满

# --- 成本函数 (COST Function) ---
函数 计算 COST({U_i}_i=1^N):
    total_cost = 0
    
    对于 {P} 中的每一个 Pauli 算符 P:
        product_confidence = 1.0
        
        对于 i 从 1 到 N:  # 遍历 N 个测量电路
            # p_i(P) 是关键：第 i 个电路测量到 P 的概率
            # 这一步使用张量网络高效计算
            p_i(P) = 高效计算 Pauli 权重(U_i, P)
            
            # epsilon (ε) 是一个超参数
            product_confidence *= exp(- (ε^2 / 2) * p_i(P))
        
        # w_P 是算符 P 的重要性权重
        total_cost += w_P * (2 * product_confidence)
        
    返回 total_cost
```

### 2\. 算法关键细节说明

#### A. 成本函数 (COST Function)

这是整个优化过程的核心。

  * **目标：** $COST_{\epsilon}(\{\mathcal{U}_{i}\}_{i=1}^{N})=\sum_{P}w_{P}\prod_{i=1}^{N}\exp\left[-\frac{\epsilon^{2}}{2}p_{i}(P)\right]$。
  * **物理意义：** 这个函数代表了“估计误差的置信度上界”。最小化这个值，等价于**最大化**我们对特定算符集合 $\{P\}$ 的测量概率 $p_i(P)$。
  * **$p_i(P)$ (Pauli Weight):** 这是最关键的计算单元。
      * **随机时：** $p_i(P)$ 是一个 0 到 1 之间的概率，表示随机系综 $\mathcal{U}_i$ 测量到 $P$ 的平均概率。
      * **确定时：** 当线路 $\mathcal{U}_i$ 被完全去随机化后，它变成一个确定的 Clifford 线路 $U_i$。此时 $p_i(P)$ 变为 0 或 1。它等于 1 当且仅当 $U_i$ 能将 $P$ 变换为纯 $Z$ 算符（即 $U_i P U_i^\dagger$ 是一个 $I$ 和 $Z$ 的张量积），否则为 0。
  * **$w_P$ (权重):** 允许我们“偏心”。在量子化学模拟中，我们更关心哈密顿量中系数 $c_P$ 大的项，因此可以设置 $w_P = |c_P|$，使算法集中资源优化这些重要的项。在一般的`Pauli算符`测量任务中我们统一设计为1.

#### B. 贪心优化 (Greedy Optimization)

算法不是一次性优化所有门（这在计算上是不可能的），而是采用贪心策略。

  * **顺序：** 它先遍历第 1 个电路，固定它的所有门；再遍历第 2 个电路，固定它的所有门……直到第 $N$ 个。
  * **逐门固定：** 在处理第 $j$ 个电路时，它先固定所有双量子比特门，再固定所有单量子比特门。
  * **局部最优：** 在每一步，算法只做出**当前**看起来最好的选择（即能最大程度降低 $COST$ 的门）。
  * **优点：** 这是一个高效的启发式方法。
  * **缺点：** 贪心算法不保证找到全局最优解，但论文中的数值模拟表明，它已经非常有效，并且优于其他（随机的）浅层影子方法。

#### C. 高效的成本计算：张量网络 (Tensor Network)

贪心算法的每一步（尝试 CNOT、SWAP 等）都需要重新计算一次 $COST$，而这需要计算 $p_i(P)$。

  * **挑战：** $p_i(P)$ 的定义 $\mathbb{E}_{U\sim\mathcal{U}_{i}}\sum_{b}\langle b|UPU^{\dagger}|b\rangle^{2}$ 涉及对整个希尔伯特空间的期望和求和，这在经典上是指数级困难的。
  * **DSS 的解决方案：** 论文作者利用了一个巧妙的映射。$p_i(P)$ 的计算可以被看作是在一个“翻倍”的希尔伯特空间中（$U \to U^{\otimes 2}$）计算 Pauli 算符 $P^{\otimes 2}$ 在随机线路演化下的概率分布。
  * **映射到张量网络：** 这个演化过程可以被精确地表示为一个张量网络。
      * 每个**Pauli 算符**（$I, X, Y, Z$）被表示为一个 4 维向量 $\vec{v}$。
      * 每个**量子门**（无论是随机的还是确定的）被表示为一个 4x4 或 16x16 的**随机矩阵 (stochastic matrix)**（一个张量）。例如，随机单比特门 $\tilde{U}_{Cl(2)}$ 是一个将 $X, Y, Z$ 映射到 $\{X, Y, Z\}$ 均匀混合的矩阵。
      * **$p_i(P)$ 的计算** 变成了在张量网络中进行一系列矩阵乘法（张量收缩）。
  * **效率：** 由于线路是 Brickwall 结构的浅层线路，这个张量网络的计算复杂度只与 $n$ 呈**多项式**关系（与 $4^d$ 或 $2^d$ 成正比，但由于 $d$ 很小，例如 $d=\mathcal{O}(\log n)$，总复杂度是可控的）。

张量网络 (Tensor Network, TN) 是 DSS 算法在经典计算机上**高效计算成本函数**的核心技术。

在 DSS 算法中有两层含义：
1.  **外层优化 (Greedy Algorithm)：** 这是指 Algorithm 1 本身，通过贪心策略，逐个尝试将随机门替换为确定性门（如 CNOT, SWAP），并选择使**总成本函数 $COST$** 最小的那个门。
2.  **内层计算 (TN Contraction)：** 这是指在执行外层优化的**每一步**时，我们都需要计算 $COST$ 函数的值。这个函数依赖于 $p_i(P)$，即第 $i$ 个电路测量到 $P$ 的概率。**张量网络技术就是用来高效完成这个“内层计算”的**。

### 1. 如何表示 Pauli 算符 (及所有元素)

在张量网络中，我们不是在模拟波函数，而是在模拟算符在“Pauli 空间”中的演化。DSS 算法的作者们采用了一种巧妙的“双副本 (doubled copy)”表示法，这使得所有的门和算符都可以被表示为实数张量，并且避免了跟踪复杂的相位 (phase)。

在这个表示法下，计算被映射到一个经典的马尔可夫过程。

#### A. Pauli 算符的表示
一个 $n$ 量子比特的 Pauli 算符 $P$ 被表示为一个由 $n$ 个向量组成的张量积 $\vec{v}_{P} = \bigotimes_{s=1}^{n} \vec{v}^{(s)}$。

关键在于，每个**单比特 Pauli 算符** $P_s$（的双副本 $P_s^{\otimes 2}$）被表示为一个 **4 维向量** $\vec{v}^{(s)}$：

* $I^{\otimes 2}$ (Identity) $\rightarrow \vec{v}^{(I)} = (1, 0, 0, 0)^T$
* $X^{\otimes 2}$ (Pauli-X) $\rightarrow \vec{v}^{(X)} = (0, 1, 0, 0)^T$
* $Y^{\otimes 2}$ (Pauli-Y) $\rightarrow \vec{v}^{(Y)} = (0, 0, 1, 0)^T$
* $Z^{\otimes 2}$ (Pauli-Z) $\rightarrow \vec{v}^{(Z)} = (0, 0, 0, 1)^T$

这个 4 维空间就是我们张量网络的“物理索引 (physical leg)”。

#### B. 量子门的表示
量子门（无论是确定的还是随机的）被表示为**张量** (即矩阵或高阶矩阵)，它们作用于上述的 4 维向量空间。我们一般用下面的图表给出编码：
| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Two-qubit gate options** | ~Cl($2^2$) | Identity | CNOT | SWAP | | | |
| **Single-qubit gate options** | ~Cl(2) | Identity | X ↔ Z | Y ↔ X | Z ↔ Y | X → Z → Y | X → Y → Z |
| | | | $H$ | $S$ | $HSH$ | $SH$ | $HS$ |

* **确定的单比特门 (Fixed 1-qubit gate):**
    表示为一个 $4 \times 4$ 的**置换矩阵**。例如，Hadamard (H) 门的作用是 $HZH^\dagger = X$ 和 $HXH^\dagger = Z$，它保持 $Y$ 不变。因此，它的张量 $\tilde{H}$ 会交换 $\vec{v}^{(X)}$ 和 $\vec{v}^{(Z)}$：
    $$
    \tilde{H} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0
    \end{pmatrix}
    $$
    进而我们可以给出所有的矩阵表示
    $$
    \tilde{S} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1
    \end{pmatrix}
    $$
    $$
    \tilde{HSH} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 0
    \end{pmatrix}
    $$
    $$
    \tilde{SH} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 1 & 0 & 0
    \end{pmatrix}
    $$
    $$
    \tilde{HS} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0
    \end{pmatrix}
    $$

* **确定的双比特门 (Fixed 2-qubit gate):**
    表示为一个 $16 \times 16$ 的置换矩阵（作用于两个 4 维向量的张量积，即 $4^2=16$ 维空间）。例如，CNOT 门会根据 $CNOT (P_1 \otimes P_2) CNOT^\dagger$ 的规则置换 16 个 Pauli 算符。我们关心的只有SWAP和CNOT门的张量表示，这两个都是稀疏矩阵,其中SWAP可以给出对应的张量表示
    $$\tilde{SWAP}= \sum_{i,j,k,l}\delta_{kj}\delta_{li}|k, l\rangle \langle i, j|$$
    而CNOT的张量表示就会更加复杂,我们可以给出一个映射表格。并且可以根据
    $$\mathrm{CNOT}\ P_1 \otimes P_2\ \mathrm{CNOT} = \mathrm{CNOT} \ P_1 \otimes I \ \mathrm{CNOT}\ \mathrm{CNOT}  \ I \otimes P_2 \ \mathrm{CNOT} $$
    来给出更多的运算.我们先给出一个基本表格

    |$II$ | $IX$ | $IY$ | $IZ$ | $XI$ | $YI$ | $ZI$ |
    |:---: | :---: | :---: | :---: | :---: | :---: | :---: |
    |$II$ | $IX$ | $ZY$ | $ZZ$ | $XX$ | $YX$ | $ZI$ |
    
    进而我们可以给出所有的稀疏矩阵的元素

    |$II$ | $IX$ | $IY$ | $IZ$ | 
    |:---: | :---: | :---: | :---: |
    |$II$ | $IX$ | $ZY$ | $ZZ$ |

    |$XI$ | $XX$ | $XY$ | $XZ$ | 
    |:---: | :---: | :---: | :---: |
    |$XX$ | $XI$ | $YZ$ | $YY$ |

    |$YI$ | $YX$ | $YY$ | $YZ$ | 
    |:---: | :---: | :---: | :---: |
    |$YX$ | $YI$ | $XZ$ | $XY$ |

    |$ZI$ | $ZX$ | $ZY$ | $ZZ$ | 
    |:---: | :---: | :---: | :---: |
    |$ZI$ | $ZX$ | $IY$ | $IZ$ |


* **随机门 (Random gate):**
    表示为对应确定性门张量的**平均**。例如，一个随机的单比特 Clifford 门 ($\sim Cl(2)$)，它会将 $X, Y, Z$ 等概率地映射到 $X, Y, Z$。它的 $4 \times 4$ 张量 $\tilde{U}_{Cl(2)}$ 为：
    $$
    \tilde{U}_{Cl(2)} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1/3 & 1/3 & 1/3 \\
    0 & 1/3 & 1/3 & 1/3 \\
    0 & 1/3 & 1/3 & 1/3
    \end{pmatrix}
    $$

#### C. 测量的表示
我们的目标是计算 $p_i(P)$，即最终算符是纯 $Z$ 串（由 $I$ 和 $Z$ 构成）的概率。这在张量网络中通过在最后对每个比特收缩一个“测量向量” $\vec{v}_{mmt}$ 来实现。

这个向量只“选择” $I$ 和 $Z$ 分量：
* $\vec{v}_{mmt} = (1, 0, 0, 1)^T$

### 2. 张量网络计算 (优化) 的具体过程

张量网络计算的“优化”指的是**高效的收缩顺序**。

#### A. 构建网络
首先，我们将 $p_i(P)$ 的计算过程构建为一个完整的张量网络。这个网络的结构**完全等同于**量子线路的结构。

* **输入 (Input):** 初始 Pauli 算符 $P$ 对应的 $n$ 个 4 维向量 $\vec{v}_P$。
* **中间 (Middle):** 由 $d$ 层门张量（$4 \times 4$ 或 $16 \times 16$ 的矩阵）构成，按照线路的 Brickwall 结构连接。
* **输出 (Output):** $n$ 个 4 维的“测量向量” $\vec{v}_{mmt}$。

$p_i(P)$ 的值就是**这个张量网络所有索引全部收缩后得到的标量值**。

#### B. 高效收缩 (计算优化)
* **挑战：** 如果我们天真地先把所有门张量相乘，我们会得到一个巨大的 $4^n \times 4^n$ 矩阵，这是指数级灾难。
* **解决方案 (高效收缩)：** 张量网络的高效之处在于**收缩顺序**。我们不横向（按层）收缩，而是**纵向（按比特）收缩**。

    1.  **分片 (Slicing):** 如论文中的 Supplementary Figure 3 所示，算法将线路“切”成 $O(n)$ 个“楼梯”形状的小块。
    2.  **内部收缩：** 首先在每个“小块”内部进行收缩。
    3.  **矩阵乘积：** 收缩完一个“小块”后，会得到一个矩阵 $A_j$，这个矩阵的维度**只与深度 $d$ 相关**（例如 $4^{d-1} \times 4^{d-1}$），而**与总比特数 $n$ 无关**。
    4.  **最终计算：** 整个 $p_i(P)$ 的计算就简化为 $O(n)$ 个这种小矩阵的乘积（一个矩阵乘积态 Matrix Product State 的收缩）。

* **复杂度：** 最终，计算 $p_i(P)$ 的总时间复杂度是 $O(|\{P\}| \times N \times \text{poly}(n) \times \exp(d))$。
    * 因为 $d$ 是“浅层”的（例如 $d = \mathcal{O}(\log n)$），$\exp(d)$ 是一个很小的数，所以总算法复杂度是 $\text{poly}(n)$，这是高效的。

#### C. 进一步的计算优化 (Signature Basis)
论文还提到了一个“附加技巧”。在优化双比特门（步骤 2a）时，所有的单比特门都是随机的 ($\sim Cl(2)$)。

在这种情况下，我们甚至不需要区分 $X, Y, Z$，因为它们总是在被平均。算法可以切换到一个更紧凑的 **"Signature Basis" (2 维向量)**：

* $I^{\otimes 2} \rightarrow (1, 0)^T$
* $\{X^{\otimes 2}, Y^{\otimes 2}, Z^{\otimes 2}\} \rightarrow (0, 1)^T$

这使得门张量的维度从 $4^{d-1}$ 降低到 $2^{d-1}$，进一步加快了计算速度。因为此时并不需要关心单qubit门，我们只需要关心几个双qubit门的样子，如下所示
$$ \text{Identity} \rightarrow \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
$$\text{SWAP} \rightarrow \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
$$\text{CNOT} \rightarrow \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1}{3} & 0 & \frac{2}{9} \\ 0 & 0 & \frac{1}{3} & \frac{2}{9} \\ 0 & \frac{2}{3} & \frac{2}{3} & \frac{5}{9} \end{pmatrix}$$
$$\text{CI}(2^2) \rightarrow \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1}{5} & \frac{1}{5} & \frac{1}{5} \\ 0 & \frac{1}{5} & \frac{1}{5} & \frac{1}{5} \\ 0 & \frac{3}{5} & \frac{3}{5} & \frac{3}{5} \end{pmatrix}$$


### 总结

DSS 算法是一个经典的优化过程，它在经典计算机上运行，目的是“编译”出一组最优的量子测量线路。它使用贪心策略，在“门选项”空间中搜索，其搜索的“指南针”就是 使用置信度搭建的$COST$ 函数。而使这一切在计算上可行的“引擎”，就是高效的张量网络收缩技术。