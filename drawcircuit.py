import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import random

# --- 1. 更新的 CircuitConfig 类 ---
# ============================================================================
class CircuitConfig:
    """
    量子电路配置类
    
    管理量子电路的门配置，包括单比特门(G1)和双比特门(G2)。
    电路采用砖墙结构，双比特门分为偶数层和奇数层交替排列。
    
    电路结构:
        - 偶数层(0,2,4...): 作用在(0,1), (2,3), (4,5)...量子比特对上
        - 奇数层(1,3,5...): 作用在(1,2), (3,4), (5,6)...量子比特对上
        - 周期边界: 当量子比特数为偶数时，最后一个比特与第一个比特配对
    
    属性:
        num_qubits: 量子比特数
        circuit_depth: 电路深度（双比特门层数）
        pbc: 是否启用周期边界条件
        gate1q: 单比特门配置 shape (depth+1, num_qubits)
        gate2q_even/odd: 偶数/奇数层双比特门配置
        gate2q_global: 全局门编号的一维视图
    """
    
    def __init__(self, num_qubits: int = 4, circuit_depth: int = 0):
        """
        初始化量子电路配置
        
        参数:
            num_qubits: 量子比特数，默认4
            circuit_depth: 双比特门层数，默认0
        """
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        
        # 周期边界条件：仅当量子比特数为偶数时启用
        self.pbc = (num_qubits % 2 == 0)
        
        # -------------------- 单比特门配置 --------------------
        # shape: (depth+1, num_qubits)
        # 每层双比特门后都有一层单比特门，因此是depth+1层
        self.gate1q = np.zeros(
            (self.circuit_depth + 1, self.num_qubits), 
            dtype=np.int8
        )
        
        # -------------------- 双比特门配置 --------------------
        # 偶数层: 层索引0,2,4,...
        num_even_layers = (self.circuit_depth + 1) // 2
        num_even_gates = self.num_qubits // 2
        self.gate2q_even = np.zeros(
            (num_even_layers, num_even_gates), 
            dtype=np.int8
        )
        
        # 奇数层: 层索引1,3,5,...
        num_odd_layers = self.circuit_depth // 2
        num_odd_gates = self.num_qubits // 2 if self.pbc else (self.num_qubits - 1) // 2
        self.gate2q_odd = np.zeros(
            (num_odd_layers, num_odd_gates), 
            dtype=np.int8
        )
        
        # -------------------- 统计信息 --------------------
        self.num_gates_per_even_layer = num_even_gates
        self.num_gates_per_odd_layer = num_odd_gates
        self.total_gate2q = num_even_layers * num_even_gates + num_odd_layers * num_odd_gates
        
        # 创建全局门编号的一维视图
        self._create_gate2q_view()
    
    # ========================================================================
    # 内部辅助方法
    # ========================================================================
    
    def _create_gate2q_view(self):
        """创建双比特门的一维全局视图，用于按编号访问"""
        if self.total_gate2q == 0:
            self.gate2q_global = np.array([], dtype=np.int8)
            return
        
        self.gate2q_global = np.zeros(self.total_gate2q, dtype=np.int8)
        self._sync_gates(from_global=False)
    
    def _sync_gates(self, from_global: bool):
        """
        同步全局视图和分层数组
        
        参数:
            from_global: True=从global同步到even/odd, False=从even/odd同步到global
        """
        idx = 0
        
        # 按层对遍历：(偶数层, 奇数层), (偶数层, 奇数层), ...
        for layer_pair in range((self.circuit_depth + 1) // 2):
            # 处理偶数层
            if layer_pair < self.gate2q_even.shape[0]:
                for gate_pos in range(self.num_gates_per_even_layer):
                    if from_global:
                        self.gate2q_even[layer_pair, gate_pos] = self.gate2q_global[idx]
                    else:
                        self.gate2q_global[idx] = self.gate2q_even[layer_pair, gate_pos]
                    idx += 1
            
            # 处理奇数层
            if layer_pair < self.gate2q_odd.shape[0]:
                for gate_pos in range(self.num_gates_per_odd_layer):
                    if from_global:
                        self.gate2q_odd[layer_pair, gate_pos] = self.gate2q_global[idx]
                    else:
                        self.gate2q_global[idx] = self.gate2q_odd[layer_pair, gate_pos]
                    idx += 1
    
    def _is_valid_gate2q_position(self, layer: int, qubit: int) -> bool:
        """
        检查双比特门位置是否有效
        
        规则:
            1. layer必须在[0, circuit_depth)范围内
            2. 偶数层只能放在偶数量子比特上，奇数层只能放在奇数量子比特上
            3. 无周期边界时，最后一个量子比特不能作为起始位置
        """
        # 检查范围
        if not (0 <= layer < self.circuit_depth and 0 <= qubit < self.num_qubits):
            return False
        
        # 检查奇偶性匹配
        if layer % 2 != qubit % 2:
            return False
        
        # 检查边界条件
        if qubit == self.num_qubits - 1 and not self.pbc:
            return False
        
        return True
    
    # ========================================================================
    # 全局编号访问接口
    # ========================================================================
    
    def gate_index_to_position(self, gate_index: int) -> tuple:
        """
        将全局门编号转换为(layer, qubit)位置
        
        参数:
            gate_index: 全局门编号 [0, total_gate2q)
            
        返回:
            (layer, qubit): 层索引和量子比特索引
        """
        if not (0 <= gate_index < self.total_gate2q):
            raise ValueError(f"门编号必须在 0 到 {self.total_gate2q-1} 之间")
        
        # 计算所属的层对
        gates_per_layer_pair = self.num_gates_per_even_layer + self.num_gates_per_odd_layer
        layer_pair = gate_index // gates_per_layer_pair
        index_in_pair = gate_index % gates_per_layer_pair
        
        # 判断是偶数层还是奇数层
        if index_in_pair < self.num_gates_per_even_layer:
            # 偶数层
            layer = layer_pair * 2
            gate_pos = index_in_pair
            qubit = gate_pos * 2
        else:
            # 奇数层
            layer = layer_pair * 2 + 1
            gate_pos = index_in_pair - self.num_gates_per_even_layer
            qubit = gate_pos * 2 + 1
        
        return (layer, qubit)
    
    def position_to_gate_index(self, layer: int, qubit: int) -> int:
        """
        将(layer, qubit)位置转换为全局门编号
        
        参数:
            layer: 层索引
            qubit: 量子比特索引
            
        返回:
            gate_index: 全局门编号
        """
        if not self._is_valid_gate2q_position(layer, qubit):
            raise ValueError(f"位置 ({layer}, {qubit}) 不是有效的双比特门位置")
        
        # 计算基准偏移
        layer_pair = layer // 2
        gates_per_layer_pair = self.num_gates_per_even_layer + self.num_gates_per_odd_layer
        base_offset = layer_pair * gates_per_layer_pair
        
        # 根据奇偶层计算具体位置
        if layer % 2 == 0:
            # 偶数层
            gate_pos = qubit // 2
            return base_offset + gate_pos
        else:
            # 奇数层
            gate_pos = qubit // 2
            return base_offset + self.num_gates_per_even_layer + gate_pos
    
    def set_gate2q_by_index(self, gate_index: int, gate_type: int) -> None:
        """
        通过全局门编号设置双比特门
        
        参数:
            gate_index: 全局门编号
            gate_type: 门类型编码
        """
        if not (0 <= gate_index < self.total_gate2q):
            raise ValueError(f"门编号必须在 0 到 {self.total_gate2q-1} 之间")
        
        self.gate2q_global[gate_index] = gate_type
        self._sync_gates(from_global=True)
    
    def get_gate2q_by_index(self, gate_index: int) -> int:
        """
        通过全局门编号获取双比特门类型
        
        参数:
            gate_index: 全局门编号
            
        返回:
            gate_type: 门类型编码
        """
        if not (0 <= gate_index < self.total_gate2q):
            raise ValueError(f"门编号必须在 0 到 {self.total_gate2q-1} 之间")
        
        return int(self.gate2q_global[gate_index])
    
    # ========================================================================
    # 位置访问接口
    # ========================================================================
    
    def set_gate2q(self, layer: int, qubit: int, gate_type: int) -> None:
        """
        通过层和量子比特位置设置双比特门
        
        参数:
            layer: 层索引
            qubit: 量子比特索引
            gate_type: 门类型编码
        """
        if not self._is_valid_gate2q_position(layer, qubit):
            raise ValueError(f"位置 ({layer}, {qubit}) 不是有效的双比特门位置")
        
        gate_index = qubit // 2
        
        if layer % 2 == 0:
            # 偶数层
            self.gate2q_even[layer // 2, gate_index] = gate_type
        else:
            # 奇数层
            self.gate2q_odd[layer // 2, gate_index] = gate_type
        
        self._sync_gates(from_global=False)
    
    def get_gate2q(self, layer: int, qubit: int) -> int:
        """
        通过层和量子比特位置获取双比特门类型
        
        参数:
            layer: 层索引
            qubit: 量子比特索引
            
        返回:
            gate_type: 门类型编码（无效位置返回0）
        """
        if not self._is_valid_gate2q_position(layer, qubit):
            return 0
        
        gate_index = qubit // 2
        
        if layer % 2 == 0:
            # 偶数层
            return int(self.gate2q_even[layer // 2, gate_index])
        else:
            # 奇数层
            return int(self.gate2q_odd[layer // 2, gate_index])
    
    def set_gate1q(self, layer: int, qubit: int, gate_type: int) -> None:
        """
        设置单比特门
        
        参数:
            layer: 层索引 [0, depth]
            qubit: 量子比特索引
            gate_type: 门类型编码
        """
        self.gate1q[layer, qubit] = gate_type
    
    def get_gate1q(self, layer: int, qubit: int) -> int:
        """
        获取单比特门类型
        
        参数:
            layer: 层索引
            qubit: 量子比特索引
            
        返回:
            gate_type: 门类型编码
        """
        return int(self.gate1q[layer, qubit])
    
    # ========================================================================
    # 迭代器接口
    # ========================================================================
    
    def iter_gate2q(self):
        """
        迭代所有双比特门
        
        产生:
            ((layer, qubit), gate_type)
        """
        for r in range(self.circuit_depth):
            for q in range(self.num_qubits):
                if self._is_valid_gate2q_position(r, q):
                    gate_type = self.get_gate2q(r, q)
                    yield (r, q), gate_type
    
    def iter_gate1q(self):
        """
        迭代所有单比特门
        
        产生:
            ((layer, qubit), gate_type)
        """
        for r in range(self.circuit_depth + 1):
            for q in range(self.num_qubits):
                gate_type = self.get_gate1q(r, q)
                yield (r, q), gate_type
# ============================================================================


# ============================================================================
# 量子电路可视化绘图函数
# ============================================================================

def draw_control_dot(ax, x, y):
    """
    绘制控制比特标记（实心黑点）
    
    用于CNOT等控制门的控制比特位置。
    
    参数:
        ax: matplotlib坐标轴对象
        x: 横坐标位置
        y: 纵坐标位置（量子比特索引）
    """
    dot = patches.Circle(
        (x, y), 
        radius=0.125, 
        facecolor='black', 
        zorder=3
    )
    ax.add_patch(dot)


def draw_target_xor(ax, x, y):
    """
    绘制目标比特标记（⊕符号）
    
    用于CNOT等控制门的目标比特位置。
    绘制一个带十字的空心圆圈。
    
    参数:
        ax: matplotlib坐标轴对象
        x: 横坐标位置
        y: 纵坐标位置（量子比特索引）
    """
    radius = 0.125
    
    # 绘制空心圆圈
    circle = patches.Circle(
        (x, y), 
        radius=radius, 
        facecolor='white', 
        edgecolor='black', 
        zorder=3, 
        fill=True
    )
    ax.add_patch(circle)
    
    # 绘制垂直线（|）
    ax.plot(
        [x, x], 
        [y - radius, y + radius], 
        color='black', 
        zorder=4, 
        lw=1.5
    )
    
    # 绘制水平线（—）
    ax.plot(
        [x - radius, x + radius], 
        [y, y], 
        color='black', 
        zorder=4, 
        lw=1.5
    )


def draw_circuit(config: CircuitConfig, svg_full_path: str = ''):
    """
    绘制量子电路图
    
    将CircuitConfig对象可视化为量子电路图，包含：
        - 量子比特线（水平线）
        - 单比特门（矩形框，显示门名称）
        - 双比特门（CNOT、SWAP等）
        - 测量门（M标记）
    
    电路布局:
        - 偶数列(0,2,4...): 单比特门
        - 奇数列(1,3,5...): 双比特门
        - 最后一列: 测量门
    
    参数:
        config: 电路配置对象
        svg_full_path: 保存路径。如果提供则保存文件；否则显示图像
                      支持格式: svg, png, pdf, jpg
    
    示例:
        >>> circuit = CircuitConfig(num_qubits=4, circuit_depth=3)
        >>> draw_circuit(circuit, 'output.svg')  # 保存为SVG
        >>> draw_circuit(circuit)                 # 显示图像
    """
    
    # ========================================================================
    # 阶段A: 绘图参数配置
    # ========================================================================
    
    # 门的颜色配置
    GATE_1Q_COLORS = {
        0: 'lightcoral',   # Cl(2): Clifford平均
        1: 'skyblue',      # I_1: 单位门
        2: 'lightgreen',   # H: Hadamard门
        3: 'violet',       # S: 相位门
        4: 'gold',         # HSH: 组合门
        5: 'orange',       # SH: 组合门
        6: 'pink',         # HS: 组合门
    }
    GATE_2Q_COLORS = {
        0: 'lightcoral',   # Cl(2^2): Clifford平均
        1: 'skyblue',      # I_2: 单位门
        2: 'lightgreen',   # SWAP: 交换门
        3: 'violet',       # CNOT: 受控非门
    }

        # 单比特门名称映射
    GATE_1Q_NAMES = {
        0: 'Cl(2)',
        1: 'I₁',
        2: 'H',
        3: 'S',
        4: 'HSH',
        5: 'SH',
        6: 'HS',
    }

    # 双比特门名称映射
    GATE_2Q_NAMES = {
        0: 'Cl(2²)',
        1: 'I₂',
        2: 'SWAP',
        3: 'CNOT',
    }
    
    # 门的尺寸参数
    GATE_WIDTH = 0.8       # 门框宽度
    GATE_HEIGHT = 0.8      # 门框高度
    CORNER_RADIUS = 0.2    # 圆角半径
    
    # 提取电路参数
    num_qubits = config.num_qubits
    depth = config.circuit_depth
    
    # ========================================================================
    # 阶段B: 初始化画布
    # ========================================================================
    
    # 计算画布尺寸
    max_gate_x = 2 * depth + 1                    # 最后一个双比特门的x坐标
    x_measure_pos = max_gate_x + 0.3              # 测量门的x坐标
    canvas_max_x = x_measure_pos + GATE_WIDTH/2 + 0.3  # 画布右边界
    
    # 自动调整图像大小
    fig_width = max(10, (canvas_max_x + 1.5) * 0.7)
    fig_height = max(4, (num_qubits + 1) * 0.7)
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-2.5, canvas_max_x)  # 原来是-1.5,改为-2.5
    ax.set_ylim(-0.5, num_qubits - 0.5)
    ax.invert_yaxis()  # y轴向下，q0在顶部
    ax.axis('off')     # 隐藏坐标轴
    
    # ========================================================================
    # 阶段C: 预处理SWAP门并绘制量子比特线
    # ========================================================================
    
    # 收集所有SWAP门的位置（用于分段绘制量子比特线）
    swap_locations = {}
    for (r, q1), gate_type in config.iter_gate2q():
        if gate_type == 2:  # SWAP门
            q2 = (q1 + 1) % config.num_qubits
            x_pos = r * 2 + 1
            
            # 记录SWAP门在两个量子比特上的位置
            if q1 not in swap_locations:
                swap_locations[q1] = []
            if q2 not in swap_locations:
                swap_locations[q2] = []
            
            swap_locations[q1].append(x_pos)
            swap_locations[q2].append(x_pos)
    
    # 对每个量子比特的SWAP位置排序
    for q in swap_locations:
        swap_locations[q].sort()
    
    # 绘制所有量子比特线
    x_margin = GATE_WIDTH / 2  # 门框边缘留白
    
    for q in range(num_qubits):
        # 绘制量子比特标签
        ax.text(
            -1.5, q,  # 原来是-0.7,改为-1.5
            f'q{q}:', 
            ha='right', 
            va='center', 
            fontsize=10
        )
        
        # 分段绘制量子比特线（在SWAP门处断开）
        swaps_on_this_line = swap_locations.get(q, [])
        current_x = -1.0  # 原来是-0.5,改为-1.0
        
        # 绘制每一段线（在SWAP门之间）
        for x_pos in swaps_on_this_line:
            ax.plot(
                [current_x, x_pos - x_margin], 
                [q, q], 
                color='black', 
                zorder=1, 
                lw=2, 
                solid_capstyle='round'
            )
            current_x = x_pos + x_margin  # 更新起点
        
        # 绘制最后一段线（到测量门）
        ax.plot(
            [current_x, x_measure_pos - GATE_WIDTH/2],
            [q, q], 
            color='black', 
            zorder=1, 
            lw=2, 
            solid_capstyle='round'
        )
    
    # ========================================================================
    # 阶段D: 绘制单比特门
    # ========================================================================
    
    for (r, q), gate_type in config.iter_gate1q():
        # 跳过单位门（不显示）
        if gate_type == 1:
            continue
        
        # 计算门的位置
        x_pos = r * 2  # 偶数列
        
        # 获取门名称
        gate_name = GATE_1Q_NAMES.get(gate_type, str(gate_type))
        
        # 选择门的颜色
        color = GATE_1Q_COLORS.get(gate_type, 'gray')
        
        # 绘制圆角矩形框
        bbox = patches.FancyBboxPatch(
            (x_pos - GATE_WIDTH/2, q - GATE_HEIGHT/2),  # 左下角坐标
            GATE_WIDTH, 
            GATE_HEIGHT,
            boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
            facecolor=color, 
            edgecolor='black', 
            zorder=2
        )
        ax.add_patch(bbox)
        
        # 在门框中心添加门名称标签
        ax.text(
            x_pos, q, 
            gate_name, 
            ha='center', 
            va='center', 
            zorder=3, 
            fontsize=8,
            fontweight='bold'
        )
    
    # ========================================================================
    # 阶段E: 绘制双比特门
    # ========================================================================
    
    for (r, q1), gate_type in config.iter_gate2q():
        # 跳过单位门（不显示）
        if gate_type == 1:
            continue
        
        # 计算门的位置和作用的两个量子比特
        x_pos = r * 2 + 1  # 奇数列
        q2 = (q1 + 1) % config.num_qubits  # 周期边界条件
        
        # 获取门名称
        gate_name = GATE_2Q_NAMES.get(gate_type, str(gate_type))
        
        # -------------------- 周期边界门（跨越边界）--------------------
        if q2 < q1:
            # --- Clifford平均门 (gate_type=0) ---
            if gate_type == 0:
                color = GATE_2Q_COLORS.get(gate_type, 'gray')
                
                # 绘制上方门框（q2位置）
                bbox_top = patches.FancyBboxPatch(
                    (x_pos - GATE_WIDTH/2, q2 - GATE_HEIGHT/2), 
                    GATE_WIDTH, 
                    GATE_HEIGHT, 
                    boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}", 
                    facecolor=color, 
                    edgecolor='black', 
                    zorder=2
                )
                ax.add_patch(bbox_top)
                
                # 绘制下方门框（q1位置）
                bbox_bot = patches.FancyBboxPatch(
                    (x_pos - GATE_WIDTH/2, q1 - GATE_HEIGHT/2), 
                    GATE_WIDTH, 
                    GATE_HEIGHT, 
                    boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}", 
                    facecolor=color, 
                    edgecolor='black', 
                    zorder=2
                )
                ax.add_patch(bbox_bot)
                
                # 添加门名称标签
                ax.text(x_pos, q1, gate_name, ha='center', va='center', 
                       zorder=3, fontsize=7, fontweight='bold')
                ax.text(x_pos, q2, gate_name, ha='center', va='center', 
                       zorder=3, fontsize=7, fontweight='bold')
                
                # 绘制连接虚线（穿越边界）
                ax.plot([x_pos, x_pos], [q1, num_qubits - 0.5], 
                       color=color, linestyle='--', zorder=1.5)
                ax.plot([x_pos, x_pos], [q2, -0.5], 
                       color=color, linestyle='--', zorder=1.5)
            
            # --- CNOT门 (gate_type=3) ---
            elif gate_type == 3:
                # 绘制控制位（下方）
                draw_control_dot(ax, x_pos, q1)
                
                # 绘制目标位（上方）
                draw_target_xor(ax, x_pos, q2)
                
                # 绘制连接线（穿越边界）
                ax.plot([x_pos, x_pos], [q1, num_qubits - 0.5], 
                       color='black', linestyle='--', zorder=1.5, lw=2)
                ax.plot([x_pos, x_pos], [q2, -0.5], 
                       color='black', linestyle='--', zorder=1.5, lw=2)
            
            # --- SWAP门 (gate_type=2) ---
            elif gate_type == 2:
                # SWAP门的X形交叉线由4条贝塞尔曲线组成
                swap_x_left = x_pos - x_margin
                swap_x_right = x_pos + x_margin
                curve_handle = 0.25  # 曲线控制点偏移
                
                Path = mpath.Path
                y_top_edge = -0.5
                y_bot_edge = num_qubits - 0.5
                
                # 曲线1: 从q2左侧到上边界
                verts1 = [
                    (swap_x_left, q2), 
                    (swap_x_left + curve_handle, q2), 
                    (x_pos, y_top_edge + curve_handle), 
                    (x_pos, y_top_edge)
                ]
                codes1 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path1 = Path(verts1, codes1)
                patch1 = patches.PathPatch(
                    path1, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch1)
                
                # 曲线2: 从q2右侧到上边界
                verts2 = [
                    (swap_x_right, q2), 
                    (swap_x_right - curve_handle, q2), 
                    (x_pos, y_top_edge + curve_handle), 
                    (x_pos, y_top_edge)
                ]
                codes2 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path2 = Path(verts2, codes2)
                patch2 = patches.PathPatch(
                    path2, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch2)
                
                # 曲线3: 从q1左侧到下边界
                verts3 = [
                    (swap_x_left, q1), 
                    (swap_x_left + curve_handle, q1), 
                    (x_pos, y_bot_edge - curve_handle), 
                    (x_pos, y_bot_edge)
                ]
                codes3 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path3 = Path(verts3, codes3)
                patch3 = patches.PathPatch(
                    path3, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch3)
                
                # 曲线4: 从q1右侧到下边界
                verts4 = [
                    (swap_x_right, q1), 
                    (swap_x_right - curve_handle, q1), 
                    (x_pos, y_bot_edge - curve_handle), 
                    (x_pos, y_bot_edge)
                ]
                codes4 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path4 = Path(verts4, codes4)
                patch4 = patches.PathPatch(
                    path4, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch4)
        
        # -------------------- 常规相邻门 --------------------
        else:
            # --- Clifford平均门 (gate_type=0) ---
            if gate_type == 0:
                color = GATE_2Q_COLORS.get(gate_type, 'gray')
                
                # 绘制跨越两个量子比特的大矩形框
                bbox = patches.FancyBboxPatch(
                    (x_pos - GATE_WIDTH/2, q1 - GATE_HEIGHT/2), 
                    GATE_WIDTH, 
                    1.0 + GATE_HEIGHT,  # 高度覆盖两个量子比特
                    boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}", 
                    facecolor=color, 
                    edgecolor='black', 
                    zorder=2
                )
                ax.add_patch(bbox)
                
                # 在中心添加门名称标签
                ax.text(
                    x_pos, 
                    (q1 + q2) / 2, 
                    gate_name, 
                    ha='center', 
                    va='center', 
                    zorder=3, 
                    fontsize=7,
                    fontweight='bold'
                )
            
            # --- CNOT门 (gate_type=3) ---
            elif gate_type == 3:
                # 绘制垂直连接线
                ax.plot(
                    [x_pos, x_pos], 
                    [q1, q2], 
                    color='black', 
                    zorder=1.5, 
                    lw=2
                )
                
                # 绘制控制位（上方）
                draw_control_dot(ax, x_pos, q1)
                
                # 绘制目标位（下方）
                draw_target_xor(ax, x_pos, q2)
            
            # --- SWAP门 (gate_type=2) ---
            elif gate_type == 2:
                # SWAP门的X形交叉线由2条贝塞尔曲线组成
                x_start = x_pos - x_margin
                x_end = x_pos + x_margin
                curve_handle = 0.25
                
                Path = mpath.Path
                
                # 曲线1: 从q1到q2的对角线
                verts1 = [
                    (x_start, q1), 
                    (x_start + curve_handle, q1), 
                    (x_end - curve_handle, q2), 
                    (x_end, q2)
                ]
                codes1 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path1 = Path(verts1, codes1)
                patch1 = patches.PathPatch(
                    path1, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch1)
                
                # 曲线2: 从q2到q1的对角线
                verts2 = [
                    (x_start, q2), 
                    (x_start + curve_handle, q2), 
                    (x_end - curve_handle, q1), 
                    (x_end, q1)
                ]
                codes2 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path2 = Path(verts2, codes2)
                patch2 = patches.PathPatch(
                    path2, 
                    facecolor='none', 
                    edgecolor='black', 
                    lw=2, 
                    zorder=1.5, 
                    capstyle='round'
                )
                ax.add_patch(patch2)
    
    # ========================================================================
    # 阶段F: 绘制测量门
    # ========================================================================
    
    measure_color = '#E0E0E0'  # 浅灰色背景
    
    for q in range(num_qubits):
        # 绘制测量门框
        bbox_m = patches.FancyBboxPatch(
            (x_measure_pos - GATE_WIDTH/2, q - GATE_HEIGHT/2),
            GATE_WIDTH, 
            GATE_HEIGHT,
            boxstyle=f"round,pad=0,rounding_size={CORNER_RADIUS}",
            facecolor=measure_color, 
            edgecolor='black', 
            zorder=2
        )
        ax.add_patch(bbox_m)
        
        # 添加测量标记'M'
        ax.text(
            x_measure_pos, 
            q, 
            'M', 
            ha='center', 
            va='center', 
            zorder=3, 
            fontsize=10, 
            fontweight='bold'
        )
    
    # ========================================================================
    # 阶段G: 保存或显示图像
    # ========================================================================
    
    if svg_full_path:
        # 提供了保存路径，保存文件
        try:
            # 从路径推断文件格式
            file_format = 'svg'  # 默认格式
            if '.' in svg_full_path:
                ext = svg_full_path.split('.')[-1].lower()
                if ext in ['svg', 'png', 'pdf', 'jpg']:
                    file_format = ext
            
            # 保存图像
            plt.savefig(
                svg_full_path, 
                format=file_format, 
                bbox_inches='tight'
            )
        except Exception as e:
            print(f"  ⚠️  无法保存电路图到 {svg_full_path}: {e}")
        
        # 关闭图形以释放内存
        plt.close(fig)
    else:
        # 未提供路径，显示图像
        plt.show()


# --- 4. 示例用法 ---
if __name__ == "__main__":
    
    N_QUBITS = 8
    DEPTH = 5
    my_config = CircuitConfig(num_qubits=N_QUBITS, circuit_depth=DEPTH)

    # 随机填充单比特门
    for r in range(my_config.circuit_depth + 1):
        for q in range(my_config.num_qubits):
            # if random.random() < 0.9:
            #     my_config.set_gate1q(r, q, random.randint(1, 4))
            my_config.set_gate1q(r, q, 1)
    
    # 随机填充双比特门
    for r in range(my_config.circuit_depth):
        for q in range(my_config.num_qubits):
            if my_config._is_valid_gate2q_position(r, q):
                my_config.set_gate2q(r, q, 2)

    # # 手动设置以确保所有类型都被绘制
    # # 常规门
    # if my_config._is_valid_gate2q_position(0, 0):
    #     my_config.set_gate2q(0, 0, 0)  # 类型0
    # if my_config._is_valid_gate2q_position(0, 2):
    #     my_config.set_gate2q(0, 2, 2)  # CNOT
    
    # # 常规 SWAP
    # if my_config._is_valid_gate2q_position(1, 1):
    #     my_config.set_gate2q(1, 1, 3)  # SWAP
    # if my_config._is_valid_gate2q_position(2, 4):
    #     my_config.set_gate2q(2, 4, 3)  # SWAP
    
    # # PBC CNOT (如果启用了周期边界条件)
    # if my_config.pbc and my_config._is_valid_gate2q_position(1, N_QUBITS - 1):
    #     my_config.set_gate2q(1, N_QUBITS - 1, 2)
    
    # # PBC SWAP
    # if my_config.pbc and my_config._is_valid_gate2q_position(3, N_QUBITS - 1):
    #     my_config.set_gate2q(3, N_QUBITS - 1, 3)
    
    # # PBC 锁定门
    # if my_config.pbc and my_config._is_valid_gate2q_position(2, N_QUBITS - 1):
    #     my_config.set_gate2q(2, N_QUBITS - 1, 0)

    # 绘制电路
    print(f"正在绘制 {N_QUBITS} qubits, depth {DEPTH} 的电路 (包含测量层)...")
    print(f"周期边界条件: {my_config.pbc}")
    print(f"双比特门总数: {my_config.total_gate2q}")
    draw_circuit(my_config)