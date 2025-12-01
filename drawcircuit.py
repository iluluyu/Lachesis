import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import random
from dss import CircuitConfig

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
    
    # 如果没有提供路径 (即 svg_full_path 为 None 或空字符串)，则默认设为当前目录下的 test.svg
    if not svg_full_path:
        svg_full_path = "test.svg"

    # 统一执行保存逻辑
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
        # 可选：打印提示方便确认
        # print(f"  ✓ 电路图已保存至: {svg_full_path}")
        
    except Exception as e:
        print(f"  ⚠️  无法保存电路图到 {svg_full_path}: {e}")

    # 关闭图形以释放内存
    plt.close(fig)