"""
DSS (Derandomized Shadow Sampling) 量子电路优化算法

该模块实现了用于量子测量优化的DSS算法,通过优化量子电路的门配置来最小化测量成本。
主要包含以下几个核心组件:
1. 量子门矩阵配置 (MatConfig1, MatConfig2)
2. Pauli算符表示和操作 (PauliOperator, PauliOperatorCollection)
3. 量子电路配置 (CircuitConfig)
4. 成本函数计算 (CostCalculator)
5. DSS优化算法 (DSS)

"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from typing import Union, List, Literal
import os



# ============================================================================
# 第一部分: 量子门矩阵配置类
# ============================================================================

class MatConfigBase:
    """
    基础配置类,包含共享的映射关系
    
    该类定义了Pauli算符和量子门在整数和字符串之间的映射关系,
    供所有配置类继承使用。
    """
    
    # Pauli算符映射: 字符串 <-> 整数
    # I=0, X=1, Y=2, Z=3
    PAULI_TO_INT = {    
        'Pauli_I': 0,
        'Pauli_X': 1,
        'Pauli_Y': 2,
        'Pauli_Z': 3
    }
    INT_TO_PAULI = {v: k for k, v in PAULI_TO_INT.items()}
    
    # 单比特门映射: 字符串 <-> 整数
    # Cl2=0(Clifford平均), I1=1(单位), H=2(Hadamard), S=3(相位), 等
    GATE_TO_INT_1 = {
        'Cl2': 0,
        'I1': 1,
        'H': 2,
        'S': 3,
        'HSH': 4,
        'SH': 5,
        'HS': 6,
    }
    INT_TO_GATE_1 = {v: k for k, v in GATE_TO_INT_1.items()}
    
    # 双比特门映射: 字符串 <-> 整数
    # Cl4=0(Clifford平均), I2=1(单位), SWAP=2, CNOT=3
    GATE_TO_INT_2 = {
        'Cl4': 0,
        'I2': 1,
        'SWAP': 2,
        'CNOT': 3,
    }
    INT_TO_GATE_2 = {v: k for k, v in GATE_TO_INT_2.items()}


class MatConfig1(MatConfigBase):
    """
    单比特和双比特量子门配置 (完整4维表示)
    
    该类使用4维向量表示Pauli基 [I, X, Y, Z],适用于精确的量子门演化计算。
    所有矩阵都以numpy数组形式存储,使用float64精度。
    
    维度说明:
    - 单比特门: (4, 4) 矩阵,作用在4维Pauli空间
    - 双比特门: (4, 4, 4, 4) 张量,前两个维度是输出,后两个维度是输入
    - index: (control_out, target_out, control_in, target_in)
    """
    
    # -------------------- Pauli基向量 (4维完整表示) --------------------
    # 每个向量在对应位置为1,其他位置为0
    Pauli_I = np.array([1, 0, 0, 0], dtype=np.float64)  # I基: [1,0,0,0]
    Pauli_X = np.array([0, 1, 0, 0], dtype=np.float64)  # X基: [0,1,0,0]
    Pauli_Y = np.array([0, 0, 1, 0], dtype=np.float64)  # Y基: [0,0,1,0]
    Pauli_Z = np.array([0, 0, 0, 1], dtype=np.float64)  # Z基: [0,0,0,1]
    
    # -------------------- 测量向量 --------------------
    # 测量只关心I和Z分量,因此向量为[1, 0, 0, 1]
    measure = np.array([1, 0, 0, 1], dtype=np.float64)
    
    # -------------------- 单比特Clifford门 (4x4矩阵) --------------------
    
    # Cl2: 单比特Clifford群平均化矩阵
    # 格式: [输出qubit, 输入qubit]
    Cl2_mapping = np.array([
        [0, 0],    # I -> I 1
        [1, 1],    # X -> X 1/3
        [2, 1],    # X -> Y 1/3
        [3, 1],    # X -> Z 1/3
        [1, 2],    # Y -> X 1/3
        [2, 2],    # Y -> Y 1/3
        [3, 2],    # Y -> Z 1/3
        [1, 3],    # Z -> X 1/3
        [2, 3],    # Z -> Y 1/3
        [3, 3],    # Z -> Z 1/3

    ])

    # 对应的值
    Cl2_values = np.array([1, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3])

    # 根据映射表构建 Cl2 矩阵
    Cl2 = np.zeros((4, 4), dtype=np.float64)
    Cl2[Cl2_mapping[:, 0], Cl2_mapping[:, 1]] = Cl2_values
    
    # I1: 单位矩阵 (4x4)
    # 不改变任何Pauli算符
    I1 = np.eye(4, dtype=np.float64)

    # H: Hadamard门 (4x4)
    # 实现 X <-> Z 交换,保持Y不变
    H_mapping = np.array([
        [0, 0],    # I -> I 1
        [3, 1],    # X -> Z 1
        [2, 2],    # Y -> Y 1
        [1, 3],    # Z -> Z 1
    ])
    H_values = np.array([1, 1, 1, 1])
    H = np.zeros((4, 4), dtype=np.float64)
    H[H_mapping[:, 0], H_mapping[:, 1]] = H_values
    
    # S: 相位门 (4x4))
    # 实现 X <-> Y 交换,保持Z不变
    S_mapping = np.array([
        [0, 0],    # I -> I 1
        [2, 1],    # X -> Y 1
        [1, 2],    # Y -> X 1
        [3, 3],    # Z -> Z 1
    ])
    S_values = np.array([1, 1, 1, 1])
    S = np.zeros((4, 4), dtype=np.float64)
    S[S_mapping[:, 0], S_mapping[:, 1]] = S_values


    # HSH: 组合门 (4x4)
    # 实现 Z <-> Y 交换,保持X不变
    HSH_mapping = np.array([
        [0, 0],    # I -> I 1
        [1, 1],    # X -> X 1
        [3, 2],    # Y -> Z 1
        [2, 3],    # Z -> Y 1
    ])
    HSH_values = np.array([1, 1, 1, 1])
    HSH = np.zeros((4, 4), dtype=np.float64)
    HSH[HSH_mapping[:, 0], HSH_mapping[:, 1]] = HSH_values

    # SH: 组合门 (4x4)
    # 实现循环置换 X -> Z -> Y -> X
    SH_mapping = np.array([
        [0, 0],    # I -> I 1
        [3, 1],    # X -> Z 1
        [1, 2],    # Y -> X 1
        [2, 3],    # Z -> Y 1
    ])
    SH_values = np.array([1, 1, 1, 1])
    SH = np.zeros((4, 4), dtype=np.float64)
    SH[SH_mapping[:, 0], SH_mapping[:, 1]] = SH_values

    # HS: 组合门 (4x4)
    # 实现循环置换 X -> Y -> Z -> X
    HS_mapping = np.array([
        [0, 0],    # I -> I 1
        [2, 1],    # X -> Y 1
        [3, 2],    # Y -> Z 1
        [1, 3],    # Z -> X 1
    ])
    HS_values = np.array([1, 1, 1, 1])
    HS = np.zeros((4, 4), dtype=np.float64)
    HS[HS_mapping[:, 0], HS_mapping[:, 1]] = HS_values

    # --- 双比特门 (4x4x4x4 张量) ---
    
    # I2: 双比特单位矩阵
    # 张量积形式: I ⊗ I
    # 前两个维度为输出(i,j),后两个维度为输入(k,l)
    # 当 i==k 且 j==l 时为1,否则为0
    I2 = np.eye(4)[:, None, :, None] * np.eye(4)[None, :, None, :]
    
    # SWAP: 交换门
    # 交换两个量子比特的状态
    # 输出(i,j) <- 输入(j,i)
    SWAP = np.eye(4)[None, :, :, None] * np.eye(4)[:, None, None, :]
    
    # CNOT: 受控非门
    # 使用映射表定义CNOT的作用规则
    # 格式: [输出i, 输出j, 输入k, 输入l]
    CNOT_mapping = np.array([
        [0, 0, 0, 0], [0, 1, 0, 1], [3, 2, 0, 2], [3, 3, 0, 3],  # I* (控制I)
        [1, 1, 1, 0], [1, 0, 1, 1], [2, 3, 1, 2], [2, 2, 1, 3],  # X* (控制X)
        [2, 1, 2, 0], [2, 0, 2, 1], [1, 3, 2, 2], [1, 2, 2, 3],  # Y* (控制Y)
        [3, 0, 3, 0], [3, 1, 3, 1], [0, 2, 3, 2], [0, 3, 3, 3],  # Z* (控制Z)
    ])
    
    # 根据映射表构建CNOT张量
    CNOT = np.zeros((4, 4, 4, 4), dtype=np.float64)
    CNOT[CNOT_mapping[:, 0], CNOT_mapping[:, 1], 
         CNOT_mapping[:, 2], CNOT_mapping[:, 3]] = 1
    
    # Cl4: 双比特Clifford群平均化矩阵 (占位符)
    # 此处暂时用I2代替,实际使用时可能需要完整定义
    Cl4 = I2.copy()
    
    # -------------------- 类方法: 获取门矩阵 --------------------
    
    @classmethod
    def get_gate1(cls, gate: Union[int, str]) -> np.ndarray:
        """
        获取单比特门矩阵
        
        参数:
            gate: 门的整数编码或字符串名称
            
        返回:
            np.ndarray: 4x4的门矩阵副本
        """
        if isinstance(gate, int):
            gate = cls.INT_TO_GATE_1[gate]
        return getattr(cls, gate).copy()
    
    @classmethod
    def get_gate2(cls, gate: Union[int, str]) -> np.ndarray:
        """
        获取双比特门矩阵
        
        参数:
            gate: 门的整数编码或字符串名称
            
        返回:
            np.ndarray: 4x4x4x4的门张量副本
        """
        if isinstance(gate, int):
            gate = cls.INT_TO_GATE_2[gate]
        return getattr(cls, gate).copy()
    
    @classmethod
    def get_pauli(cls, pauli: Union[int, str]) -> np.ndarray:
        """
        获取Pauli基向量
        
        参数:
            pauli: Pauli算符的整数编码或字符串名称
            
        返回:
            np.ndarray: 4维Pauli向量副本
        """
        if isinstance(pauli, int):
            pauli = cls.INT_TO_PAULI[pauli]
        return getattr(cls, pauli).copy()


class MatConfig2(MatConfigBase):
    """
    双比特量子门配置 (简化2维表示)
    
    该类使用2维向量表示Pauli基 [I分量, 非I分量],适用于快速的近似计算。
    非I分量代表X、Y、Z的平均效果,用于双比特门的Clifford平均化。
    
    维度说明:
    - Pauli向量: 2维 [I分量, X/Y/Z平均分量]
    - 双比特门: (2, 2, 2, 2) 张量
    - index: (control_out, target_out,control_in, target_in)
    """
    
    # -------------------- Pauli基向量 (2维简化表示) --------------------
    # [I分量, 非I分量(X/Y/Z的平均)]
    Pauli_I = np.array([1, 0], dtype=np.float64)  # 纯I: [1, 0]
    Pauli_X = np.array([0, 1], dtype=np.float64)  # 纯X: [0, 1]
    Pauli_Y = np.array([0, 1], dtype=np.float64)  # 纯Y: [0, 1]
    Pauli_Z = np.array([0, 1], dtype=np.float64)  # 纯Z: [0, 1]
    
    # -------------------- 测量向量 --------------------
    # 测量时I分量权重为1,非I分量权重为1/3
    measure = np.array([1, 1/3], dtype=np.float64)
    
    # -------------------- Clifford群平均化矩阵 --------------------
    # Cl4: 双比特Clifford群平均化 (2x2x2x2)
    # 定义了不同Pauli组合下的平均转换规则
    # 使用索引和数值数组定义Cl4的稀疏表示
    Cl4indices = np.array([
        [0, 0, 0, 0],  # II -> II (权重1)
        [0, 1, 0, 1],  # I* -> I* (权重1/5)
        [1, 0, 0, 1],  # I* -> *I (权重1/5)
        [1, 1, 0, 1],  # I* -> ** (权重3/5)
        [0, 1, 1, 0],  # *I -> I* (权重1/5)
        [1, 0, 1, 0],  # *I -> *I (权重1/5)
        [1, 1, 1, 0],  # *I -> ** (权重3/5)
        [1, 0, 1, 1],  # ** -> *I (权重1/5)
        [0, 1, 1, 1],  # ** -> I* (权重1/5)
        [1, 1, 1, 1]   # ** -> ** (权重3/5)
    ])
    Cl4values = np.array([1, 1/5, 1/5, 3/5, 1/5, 1/5, 3/5, 1/5,1/5,3/5])
     # 构建CNOT张量
    Cl4 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    Cl4[Cl4indices[:, 0], Cl4indices[:, 1], 
         Cl4indices[:, 2], Cl4indices[:, 3]] = Cl4values
    
    # -------------------- 双比特单位和交换门 --------------------
    
    # I2: 双比特单位矩阵 (2x2x2x2)
    I2 = np.eye(2)[:, None, :, None] * np.eye(2)[None, :, None, :]
    
    # SWAP: 交换门 (2x2x2x2)
    SWAP = np.eye(2)[None, :, :, None] * np.eye(2)[:, None, None, :]
    
    # -------------------- CNOT门 (简化版) --------------------
    # 使用索引和数值数组定义CNOT的稀疏表示
    # index: (control_out, target_out,control_in, target_in)
    CNOTindices = np.array([
        [0, 0, 0, 0],  # II -> II (权重1)
        [0, 1, 0, 1],  # I* -> I* (权重1/3)
        [1, 1, 0, 1],  # I* -> ** (权重2/3)
        [1, 0, 1, 0],  # *I -> *I (权重1/3)
        [1, 1, 1, 0],  # *I -> ** (权重2/3)
        [1, 0, 1, 1],  # ** -> *I (权重2/9)
        [0, 1, 1, 1],  # ** -> I* (权重2/9)
        [1, 1, 1, 1]   # ** -> ** (权重5/9)
    ])
    CNOTvalues = np.array([1, 1/3, 2/3, 1/3, 2/3, 2/9, 2/9, 5/9])
    
    # 构建CNOT张量
    CNOT = np.zeros((2, 2, 2, 2), dtype=np.float64)
    CNOT[CNOTindices[:, 0], CNOTindices[:, 1], 
         CNOTindices[:, 2], CNOTindices[:, 3]] = CNOTvalues
    
    # -------------------- 类方法 --------------------
    
    @classmethod
    def get_gate1(cls, gate: Union[int, str]) -> np.ndarray:
        """
        返回单位矩阵(占位符)
        
        注: MatConfig2主要用于双比特门,单比特门返回2x2单位矩阵防止报错
        """
        mat = np.eye(2, dtype=np.float64)
        return mat
    
    @classmethod
    def get_gate2(cls, gate: Union[int, str]) -> np.ndarray:
        """
        获取双比特门矩阵
        
        参数:
            gate: 门的整数编码或字符串名称
            
        返回:
            np.ndarray: 2x2x2x2的门张量副本
        """
        if isinstance(gate, int):
            gate = cls.INT_TO_GATE_2[gate]
        mat = getattr(cls, gate).copy()
        return mat
    
    @classmethod
    def get_pauli(cls, pauli: Union[int, str]) -> np.ndarray:
        """
        获取Pauli基向量
        
        参数:
            pauli: Pauli算符的整数编码或字符串名称
            
        返回:
            np.ndarray: 2维Pauli向量副本
        """
        if isinstance(pauli, int):
            pauli = cls.INT_TO_PAULI[pauli]
        return getattr(cls, pauli).copy()
    
# ============================================================================
# 第二部分: Pauli算符类
# ============================================================================
class PauliOperator:
    """
    单个Pauli算符类 (向量化版本)
    
    该类表示一个多量子比特的Pauli算符串,如 "XYZII"。
    使用整数数组存储,其中 [0,1,2,3] 对应 [I,X,Y,Z]。
    
    属性:
        pauliint: int8类型的NumPy数组,存储Pauli算符的整数表示
        weight: 权重值,用于加权求和
        num_qubits: 量子比特数量
        
    示例:
        >>> p = PauliOperator(pauli_string='XYZ')
        >>> print(p.pauliint)  # [1, 2, 3]
    """
    
    # 整数与字符的映射关系
    INT_TO_PAULI = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    PAULI_TO_INT = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    
    def __init__(self, 
                 pauliint: Union[List[int], np.ndarray] = None, 
                 pauli_string: Union[List[str], str] = None, 
                 weight: float = 1.0):
        """
        初始化Pauli算符
        
        参数:
            pauliint: 整数列表或数组,如 [0, 1, 2, 3]
            pauli_string: 字符串列表或字符串,如 ['I', 'X', 'Y', 'Z'] 或 'IXYZ'
            weight: 权重值,默认为1.0
            
        注意: pauliint 和 pauli_string 至少要提供一个
        
        异常:
            ValueError: 如果两个参数都未提供
        """
        if pauliint is None and pauli_string is None:
            raise ValueError("必须提供 pauliint 或 pauli_string 中的至少一个")
        
        # 优先使用整数数组
        if pauliint is not None:
            self.pauliint = np.asarray(pauliint, dtype=np.int8)
        else:
            # 从字符串转换为整数数组
            if isinstance(pauli_string, str):
                pauli_string = list(pauli_string)
            self.pauliint = np.array(
                [self.PAULI_TO_INT[p] for p in pauli_string], 
                dtype=np.int8
            )
        
        self.weight = weight
        self.num_qubits = len(self.pauliint)
    
    def to_string(self) -> str:
        """
        将Pauli算符转换为字符串表示
        
        返回:
            str: Pauli算符字符串,如 'XYZI'
        """
        return ''.join(self.INT_TO_PAULI[i] for i in self.pauliint)
    
    def to_list(self) -> List[str]:
        """
        将Pauli算符转换为字符列表
        
        返回:
            List[str]: Pauli算符字符列表,如 ['X', 'Y', 'Z', 'I']
        """
        return [self.INT_TO_PAULI[i] for i in self.pauliint]
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return f"PauliOperator('{self.to_string()}', weight={self.weight})"
    
    def __str__(self) -> str:
        """返回对象的可读字符串"""
        return self.to_string()
    
    def copy(self):
        """
        创建Pauli算符的深拷贝
        
        返回:
            PauliOperator: 新的Pauli算符对象
        """
        return PauliOperator(
            pauliint=self.pauliint.copy(), 
            weight=self.weight
        )
    
    def __len__(self):
        """返回量子比特数量"""
        return self.num_qubits

class PauliOperatorCollection:
    """
    Pauli算符集合类
    
    管理多个Pauli算符,提供批量操作和统计功能。
    所有Pauli算符必须具有相同的量子比特数量。
    
    属性:
        num_qubits: 量子比特数量
        operators: Pauli算符列表
        pauli_matrix: 所有Pauli算符的整数矩阵 (M x N)
        weight_vector: 所有权重的向量 (M,)
        
    示例:
        >>> coll = PauliOperatorCollection(num_qubits=3)
        >>> coll.add_from_string('XYZ', weight=1.0)
        >>> coll.add_from_string('IIZ', weight=0.5)
        >>> coll.summary()
    """
    
    def __init__(self, num_qubits: int):
        """
        初始化Pauli算符集合
        
        参数:
            num_qubits: 量子比特数量
        """
        self.num_qubits = num_qubits
        self.operators: List[PauliOperator] = []
        
        # 预留空间用于批量操作
        self.pauli_matrix = []  # shape: (M, N) 其中M是算符数量,N是量子比特数num_qubits
        self.weight_vector = []  # shape: (M,)
    
    def add(self, operator: PauliOperator):
        """
        添加一个Pauli算符
        
        参数:
            operator: PauliOperator对象
            
        异常:
            ValueError: 如果算符的量子比特数不匹配
        """
        if operator.num_qubits != self.num_qubits:
            raise ValueError(
                f"算符的量子比特数 ({operator.num_qubits}) "
                f"与集合不匹配 ({self.num_qubits})"
            )
        self.operators.append(operator)
        self._invalidate_cache()
    
    def add_from_string(self, pauli_string: str, weight: float = 1.0):
        """
        从字符串添加Pauli算符
        
        参数:
            pauli_string: Pauli算符字符串,如 'XYZI'
            weight: 权重值
            
        示例:
            >>> coll.add_from_string('XYZ', weight=2.0)
        """
        operator = PauliOperator(pauli_string=pauli_string, weight=weight)
        self.add(operator)
    
    def add_from_int(self, pauliint: Union[List[int], np.ndarray], weight: float = 1.0):
        """
        从整数数组添加Pauli算符
        
        参数:
            pauliint: 整数列表或数组
            weight: 权重值
            
        示例:
            >>> coll.add_from_int([1, 2, 3], weight=1.5)
        """
        operator = PauliOperator(pauliint=pauliint, weight=weight)
        self.add(operator)
    
    def _invalidate_cache(self):
        """
        使缓存的矩阵和向量失效
        
        当添加新算符时调用,强制重新计算批量数据
        """
        self.pauli_matrix = None
        self.weight_vector = None
    
    def _build_cache(self):
        """
        构建缓存的矩阵和向量
        
        将所有Pauli算符的整数表示堆叠成矩阵,
        将所有权重堆叠成向量,用于高效的批量计算。
        """
        if self.pauli_matrix is None:
            # 堆叠所有Pauli算符的整数数组
            self.pauli_matrix = np.array(
                [op.pauliint for op in self.operators], 
                dtype=np.int8
            )  # shape: (M, N)
            
            # 堆叠所有权重
            self.weight_vector = np.array(
                [op.weight for op in self.operators], 
                dtype=np.float64
            )  # shape: (M,)
    
    def get_pauli_matrix(self) -> np.ndarray:
        """
        获取Pauli算符整数矩阵
        
        返回:
            np.ndarray: shape (M, N) 的整数矩阵
                       M = 算符数量, N = 量子比特数
        """
        self._build_cache()
        return self.pauli_matrix
    
    def get_weight(self) -> np.ndarray:
        """
        获取权重向量
        
        返回:
            np.ndarray: shape (M,) 的权重向量
        """
        self._build_cache()
        return self.weight_vector.copy()
    
    def __len__(self) -> int:
        """返回算符数量"""
        return len(self.operators)
    
    def __getitem__(self, index: int) -> PauliOperator:
        """通过索引访问算符"""
        return self.operators[index]
    
    def __iter__(self):
        """迭代所有算符"""
        return iter(self.operators)
    
    def summary(self):
        """
        打印集合的摘要信息
        
        显示量子比特数、算符数量以及每个算符的详情
        """
        print(f"PauliOperatorCollection:")
        print(f"  量子比特数: {self.num_qubits}")
        print(f"  算符数量: {len(self)}")
        print(f"  算符列表:")
        for i, op in enumerate(self.operators):
            print(f"    [{i}] {op.to_string()} (weight={op.weight})")

# ============================================================================
# 辅助函数：MatConfig访问接口
# 
# Mode 1: MatConfig1 (4维完整表示)  Mode 2: MatConfig2 (2维简化表示)
# ============================================================================


def get_pauli(pauli: Union[int, str], mode: int = 1) -> np.ndarray:
    """
    获取Pauli基向量
    
    参数:
        pauli: Pauli算符,整数编码(0=I,1=X,2=Y,3=Z)或字符串名称
        mode: 1=4维向量(完整), 2=2维向量(简化)
    """
    if mode == 1:
        return MatConfig1.get_pauli(pauli)
    elif mode == 2:
        return MatConfig2.get_pauli(pauli)
    else:
        raise ValueError(f"mode必须是1或2,当前值: {mode}")


def get_gate1(gate: Union[int, str], mode: int = 1) -> np.ndarray:
    """
    获取单比特门矩阵
    
    参数:
        gate: 门标识,整数编码或字符串名称
        mode: 1=4x4矩阵, 2=2x2单位矩阵(占位)
    """
    if mode == 1:
        return MatConfig1.get_gate1(gate)
    elif mode == 2:
        return MatConfig2.get_gate1(gate)  # 返回2×2单位矩阵
    else:
        raise ValueError(f"mode必须是1或2,当前值: {mode}")


def get_gate2(gate: Union[int, str], mode: int = 1) -> np.ndarray:
    """
    获取双比特门张量
    
    参数:
        gate: 门标识,整数编码(0=Cl4,1=I2,2=SWAP,3=CNOT)或字符串名称
        mode: 1=4x4x4x4张量, 2=2x2x2x2张量
        index: (control_out, target_out,control_in, target_in)
    """
    if mode == 1:
        return MatConfig1.get_gate2(gate)
    elif mode == 2:
        return MatConfig2.get_gate2(gate)
    else:
        raise ValueError(f"mode必须是1或2,当前值: {mode}")


def get_measure(mode: int = 1) -> np.ndarray:
    """
    获取测量向量
    
    参数:
        mode: 1=[1,0,0,1](测量I和Z), 2=[1,1/3](I和非I平均)
    """
    if mode == 1:
        return MatConfig1.measure
    elif mode == 2:
        return MatConfig2.measure
    else:
        raise ValueError(f"mode必须是1或2,当前值: {mode}")

# ============================================================================
# 第四部分: 量子电路配置类
# ============================================================================
class CircuitConfig:
    """
    量子电路配置类
    
    管理量子电路的门配置,包括单比特门(G1)和双比特门(G2)。
    电路采用砖墙结构,双比特门分为偶数层和奇数层交替排列。
    
    电路结构:
        - 偶数层(0,2,4...): 作用在(0,1), (2,3), (4,5)...量子比特对上
        - 奇数层(1,3,5...): 作用在(1,2), (3,4), (5,6)...量子比特对上
        - 周期边界: 当量子比特数为偶数时,最后一个比特与第一个比特配对
    
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
            num_qubits: 量子比特数,默认4
            circuit_depth: 双比特门层数,默认0
        """
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        
        # 周期边界条件：仅当量子比特数为偶数时启用
        self.pbc = (num_qubits % 2 == 0)
        
        # -------------------- 单比特门配置 --------------------
        # shape: (depth+1, num_qubits)
        # 每层双比特门后都有一层单比特门,因此是depth+1层
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
        num_odd_gates = self.num_qubits // 2
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
        """创建双比特门的一维全局视图,用于按编号访问"""
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
            2. 偶数层只能放在偶数量子比特上,奇数层只能放在奇数量子比特上
            3. 无周期边界时,最后一个量子比特不能作为起始位置
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

    def copy(self):
        """
        创建当前配置的深拷贝
        
        返回:
            CircuitConfig: 新的独立配置对象
        """
        # 创建新实例
        new_config = CircuitConfig(
            num_qubits=self.num_qubits,
            circuit_depth=self.circuit_depth
        )
        
        # 深拷贝所有 NumPy 数组
        new_config.gate1q = self.gate1q.copy()
        new_config.gate2q_even = self.gate2q_even.copy()
        new_config.gate2q_odd = self.gate2q_odd.copy()
        new_config.gate2q_global = self.gate2q_global.copy()
        
        # 基本类型属性会在 __init__ 中正确设置
        # pbc, num_gates_per_even_layer 等会自动计算
        
        return new_config
# ============================================================================
# 第五部分: 量子电路转换为张量网络的函数
# ============================================================================
def measure2tn(depth: int, num_qubits: int, mode: int):
    """
    将测量操作转换为张量网络表示
    
    参数:
        depth: 电路深度
        num_qubits: 量子比特数
        mode: 计算模式 (1=完整表示, 2=简化表示)
    
    返回:
        qtn.TensorNetwork: 包含所有测量张量的张量网络
    """
    # 初始化空张量网络
    tn = qtn.TensorNetwork()
    
    # 根据模式计算最终轮次索引
    # Mode 1: 包含单比特门层,轮次=2*depth+1
    # Mode 2: 仅双比特门层,轮次=depth
    round = 2 * depth + 1 if mode == 1 else depth
    
    # 为每个量子比特添加测量张量
    for q in range(num_qubits):
        # 获取测量向量（Z基测量）
        data = get_measure(mode=mode)
        
        # 设置张量标签：用于识别和操作
        tags = {'measure', f'R{depth}Q{q}', f'measureR{depth}Q{q}'}
        # tags = {f'measureR{depth}Q{q}'}  # 备选：简化标签方案
        
        # 创建测量张量
        tensor = qtn.Tensor(
            data=data,                    # 测量向量数据
            inds=(f'q{round}_{q}',),     # 索引：连接到最后一层的量子比特q
            tags=tags                     # 标签集合
        )
        
        tn.add_tensor(tensor)
    
    return tn


def circuit2tn1(circuit_config: CircuitConfig):
    """
    将量子电路配置转换为张量网络表示 (Mode 1: 4维Pauli基)
    
    用于优化单比特量子门的精确计算模式。
    
    张量网络结构:
        - 索引命名: f'q{layer}_{qubit}'
        - 单比特门: 连接同一量子比特的相邻层
        - 双比特门: 连接相邻量子比特对
    
    参数:
        circuit_config: 量子电路配置对象
        
    返回:
        qtn.TensorNetwork: 电路的张量网络表示
    """
    tn = qtn.TensorNetwork()
    N = circuit_config.num_qubits
    
    # ========================================================================
    # 第一部分: 单比特门张量
    # ========================================================================
    # 索引结构: q{2*r}_{q} --[G1]--> q{2*r+1}_{q}
    # 每个单比特门作用在单个量子比特上,连接偶数层到奇数层
    
    for (r, q), gate_type in circuit_config.iter_gate1q():
        # 获取门矩阵和名称
        try:
            gate_data = get_gate1(gate_type, mode=1)
            gate_name = MatConfig1.INT_TO_GATE_1[gate_type]
        except KeyError:
            print(f"警告: 未找到门类型 {gate_type},跳过位置 (layer={r}, qubit={q})")
            continue
        
        # 定义输入输出索引
        input_idx = f'q{2*r}_{q}'      # 输入: 偶数层
        output_idx = f'q{2*r+1}_{q}'   # 输出: 奇数层
        inds = (output_idx, input_idx)
        
        # 创建张量标签
        tags = {
            'G1',                # 单比特门标记
            gate_name,           # 门名称 (H, S, X, Y, Z等)
            f'R{r}Q{q}',         # 位置标记
            f'G1R{r}Q{q}'        # 唯一标识
        }
        
        tn.add_tensor(qtn.Tensor(data=gate_data, inds=inds, tags=tags))
    
    # ========================================================================
    # 第二部分: 双比特门张量
    # ========================================================================
    # 索引结构: q{2*r+1}_{q0,q1} --[G2]--> q{2*r+2}_{q0,q1}
    # 每个双比特门作用在相邻量子比特对上,连接奇数层到偶数层
    
    for (r, q_start), gate_type in circuit_config.iter_gate2q():
        # 获取门张量和名称
        try:
            gate_data = get_gate2(gate_type, mode=1)
            gate_name = MatConfig1.INT_TO_GATE_2[gate_type]
        except KeyError:
            print(f"警告: 未找到门类型 {gate_type},跳过位置 (layer={r}, qubit={q_start})")
            continue
        
        # 确定作用的两个量子比特（周期边界条件）
        q0 = q_start
        q1 = (q_start + 1) % N
        
        # 定义输入输出索引
        # 索引顺序: (control_out, target_out, control_in, target_in)
        control_in = f'q{2*r+1}_{q0}'
        target_in = f'q{2*r+1}_{q1}'
        control_out = f'q{2*r+2}_{q0}'
        target_out = f'q{2*r+2}_{q1}'
        inds = (control_out, target_out, control_in, target_in)

        
        # 创建张量标签
        tags = {
            'G2',                # 双比特门标记
            gate_name,           # 门名称 (CNOT, SWAP等)
            f'R{r}Q{q0}',        # 第一个量子比特位置
            f'R{r}Q{q1}',        # 第二个量子比特位置
            f'G2R{r}Q{q0}'       # 唯一标识
        }
        
        tn.add_tensor(qtn.Tensor(data=gate_data, inds=inds, tags=tags))
    
    return tn


def circuit2tn2(circuit_config: CircuitConfig):
    """
    将量子电路配置转换为张量网络表示 (Mode 2: 2维简化基)
    
    用于优化双比特量子门的快速近似计算模式。
    
    Mode 2特点:
        - 使用2维简化Pauli表示 [I分量, 非I分量]
        - 忽略单比特门（已吸收到双比特门中）
        - 层索引直接使用r(不需要x2)
    
    参数:
        circuit_config: 量子电路配置对象
        
    返回:
        qtn.TensorNetwork: 电路的张量网络表示
    """
    tn = qtn.TensorNetwork()
    N = circuit_config.num_qubits
    
    # ========================================================================
    # 双比特门张量（Mode 2忽略单比特门）
    # ========================================================================
    # 索引结构: q{r}_{q0,q1} --[G2]--> q{r+1}_{q0,q1}
    # 层索引直接使用r,无需乘以2（因为不包含单比特门层）
    
    for (r, q_start), gate_type in circuit_config.iter_gate2q():
        # 获取门张量和名称
        try:
            gate_data = get_gate2(gate_type, mode=2)
            gate_name = MatConfig2.INT_TO_GATE_2[gate_type]
        except KeyError:
            print(f"警告: 未找到门类型 {gate_type},跳过位置 (layer={r}, qubit={q_start})")
            continue
        
        # 确定作用的两个量子比特（周期边界条件）
        q0 = q_start
        q1 = (q_start + 1) % N
        
        # 定义输入输出索引
        # 索引顺序: (output_q1, output_q0, inpu_q1, input_q0)
        # 注意: Mode 2使用简化层编号（r而非2*r）
        # index: (control_out, target_out,control_in, target_in)
        control_in = f'q{r}_{q0}'
        target_in = f'q{r}_{q1}'
        control_out = f'q{r+1}_{q0}'
        target_out = f'q{r+1}_{q1}'
        inds = (control_out, target_out, control_in, target_in)
        
        # 创建张量标签
        tags = {
            'G2',                # 双比特门标记
            gate_name,           # 门名称 (CNOT, SWAP等)
            f'R{r}Q{q0}',        # 第一个量子比特位置
            f'R{r}Q{q1}',        # 第二个量子比特位置
            f'G2R{r}Q{q0}'       # 唯一标识
        }
        
        tn.add_tensor(qtn.Tensor(data=gate_data, inds=inds, tags=tags))
    
    return tn

def circuit2tn(circuit_config: CircuitConfig, mode: int = 1):  # type: ignore
    """
    将量子电路配置转换为张量网络表示（统一接口）
    
    根据mode参数选择不同的转换方式:
        - mode=1: 4维Pauli基表示,包含单比特门和双比特门（精确计算）
        - mode=2: 2维简化表示,仅包含双比特门（快速优化）
    
    参数:
        circuit_config: 量子电路配置对象
        mode: 转换模式,1或2
        
    返回:
        qtn.TensorNetwork: 电路的张量网络表示
        
    异常:
        ValueError: mode不是1或2
    """
    if mode == 1:
        return circuit2tn1(circuit_config)
    elif mode == 2:
        return circuit2tn2(circuit_config)
    else:
        raise ValueError(f"mode必须是1或2,当前值: {mode}")

def paulistring2tn(pauli_op: PauliOperator, mode: int = 1):
    """
    将Pauli算符转换为张量网络表示
    
    将Pauli算符串(如'IXYZ')转换为张量网络,每个Pauli矩阵对应一个向量张量,
    连接到电路的输入端(第0层)。
    
    参数:
        pauli_op: Pauli算符对象
                 - pauliint: 整数数组 [0,1,2,3] 对应 [I,X,Y,Z]
                 - num_qubits: 量子比特数
        mode: 1=4维Pauli基, 2=2维简化基
    
    返回:
        qtn.TensorNetwork: Pauli算符的张量网络表示
    """
    tn = qtn.TensorNetwork()
    
    # 为每个量子比特添加对应的Pauli张量
    for q_idx in range(pauli_op.num_qubits):
        # 获取该量子比特的Pauli类型 (0=I, 1=X, 2=Y, 3=Z)
        pauli_int = int(pauli_op.pauliint[q_idx])
        
        # 获取Pauli向量数据
        try:
            data = get_pauli(pauli_int, mode=mode)
        except KeyError:
            print(f"警告: 未找到Pauli索引 {pauli_int} (mode={mode}),跳过 q_idx={q_idx}")
            continue
        
        # 获取Pauli字符串表示（用于标签）
        try:
            pauli_str = MatConfigBase.INT_TO_PAULI[pauli_int]
        except KeyError:
            pauli_str = 'UNKNOWN'
        
        # 创建张量标签和索引
        tags = {
            'Pauli',                  # Pauli算符标记
            pauli_str,                # Pauli类型 (I/X/Y/Z)
            f'R0Q{q_idx}',            # 位置标记
            f'PauliR0Q{q_idx}'        # 唯一标识
        }
        inds = (f'q0_{q_idx}',)       # 连接到第0层的量子比特q_idx
        
        # 添加Pauli张量到网络
        tensor = qtn.Tensor(data=data, inds=inds, tags=tags)
        tn.add_tensor(tensor)
    
    return tn

# ============================================================================
# 第六部分: 成本函数计算器
# ============================================================================
class PauliCostCalculator:
    """
    Pauli权重矩阵和成本函数管理器
    
    管理(M, num_measures)维的权重矩阵,并高效计算和更新成本函数值。
    成本函数定义为: C = Σ_i w_i * Π_j exp(-ε²*x_ij/2)
    
    属性:
        pauli_weight: (M, num_measures) 权重矩阵,M=Pauli串数量,num_measures=测量次数
        pauli_average_weights: (M,) Pauli算符的权重向量
        cost_function_value: 当前成本函数值（只读）
        M: Pauli算符数量
        num_measures: 测量次数
        epsilon: 成本函数超参数
    """
    
    def __init__(self, 
                 weight: np.ndarray,
                 num_measurements: int, 
                 epsilon: float):
        """
        初始化成本计算器
        
        参数:
            pauli_collection: Pauli算符集合,包含权重信息
            num_measurements: 测量次数（num_measures维度）
            epsilon: 成本函数超参数
            
        异常:
            ValueError: 如果pauli_collection为空
        """
        self.num_paulis = len(weight)  # Pauli串数量
        self.num_measures = num_measurements       # 测量次数
        self.epsilon = epsilon          # 成本函数超参数
        
        if self.num_paulis == 0:
            raise ValueError("权函数weight不能为空")
        
        # -------------------- 主要数据 --------------------
        # 权重矩阵 (M, num_measures),初始化为0
        self.pauli_weight = np.zeros((self.num_paulis, self.num_measures), dtype=np.float64)
        
        # Pauli算符的平均权重 (M,)
        self.pauli_average_weights = weight
        
        # -------------------- 内部缓存（用于高效更新）--------------------
        # 每行的乘积: Π_j exp(-ε²*x_ij/2),形状 (M,)
        self._row_products = np.zeros(self.num_paulis, dtype=np.float64)
        
        # 总成本函数值（标量）
        self._cost_function_value = 0.0
        
        # 基于初始零矩阵计算成本
        self.full_recompute_cost()
    
    # ========================================================================
    # 成本函数计算
    # ========================================================================
    
    def full_recompute_cost(self):
        """
        从头完整计算成本函数
        
        用于初始化或批量更新后的重新计算。
        计算步骤:
            1. 对每个元素计算 exp(-ε²*x/2)
            2. 计算每行的乘积
            3. 与权重向量做点积得到总成本
        """
        # 1. 计算指数矩阵 (num_paulis, num_measures)
        exp_matrix = np.exp(-self.pauli_weight * self.epsilon**2 / 2)
        
        # 2. 计算每行的乘积 (num_paulis,)
        self._row_products = np.prod(exp_matrix, axis=1)
        
        # 3. 与权重做点积得到标量成本
        self._cost_function_value = np.dot(
            self.pauli_average_weights, 
            self._row_products
        )
    
    def _recompute_row_product(self, r: int) -> float:
        """
        重新计算单行的乘积
        
        参数:
            r: 行索引 [0, M)
            
        返回:
            该行的乘积值 Π_j exp(-ε²*x_rj/2)
        """
        exp_row = np.exp(-self.pauli_weight[r, :] * self.epsilon**2 / 2)
        return np.prod(exp_row)
    
    def _recompute_cost_from_products(self) -> float:
        """
        从当前行乘积重新计算总成本
        
        返回:
            成本函数值 Σ_i w_i * row_product_i
        """
        return np.dot(self.pauli_average_weights, self._row_products)
    
    # ========================================================================
    # 高效更新接口
    # ========================================================================
    
    def update_element(self, r: int, c: int, new_value: float, copy: bool = False):
        """
        更新单个矩阵元素并同步成本函数
        
        使用完全重新计算的方式更新,避免浮点数累积误差。
        
        参数:
            r: 行索引 [0, M)
            c: 列索引 [0, num_measures)
            new_value: 新的权重值
            copy: 是否返回更新后的深拷贝（不修改原对象）
            
        返回:
            如果copy=True,返回更新后的新对象；否则返回None（就地修改）
        """
        # 如果需要拷贝,创建新对象
        if copy:
            return self._update_element_copy(r, c, new_value)
        
        # 值未变化则跳过
        old_value = self.pauli_weight[r, c]
        if new_value == old_value:
            return self
        
        # 更新矩阵值
        self.pauli_weight[r, c] = new_value
        
        # 重新计算该行的乘积
        self._row_products[r] = self._recompute_row_product(r)
        
        # 从所有行乘积重新计算总成本
        self._cost_function_value = self._recompute_cost_from_products()

        return self
        
    
    def _update_element_copy(self, r: int, c: int, new_value: float) -> 'PauliCostCalculator':
        """
        创建深拷贝并更新元素（内部方法）
        
        参数:
            r: 行索引
            c: 列索引
            new_value: 新值
            
        返回:
            更新后的新PauliCostCalculator对象
        """
        # 创建新对象（浅拷贝结构）
        new_calc = PauliCostCalculator.__new__(PauliCostCalculator)
        
        # 复制基本属性
        new_calc.num_paulis = self.num_paulis
        new_calc.num_measures = self.num_measures
        new_calc.epsilon = self.epsilon
        
        # 深拷贝数组
        new_calc.pauli_weight = self.pauli_weight.copy()
        new_calc.pauli_average_weights = self.pauli_average_weights.copy()
        new_calc._row_products = self._row_products.copy()
        new_calc._cost_function_value = self._cost_function_value
        
        # 在新对象上更新
        new_calc.update_element(r, c, new_value, copy=False)
        
        return new_calc
    
    def update_column(self, c: int, new_column_values: np.ndarray, copy: bool = False):
        """
        更新整列并同步成本函数
        
        使用完全重新计算的方式更新,避免浮点数累积误差。
        适用于DSS算法中更新单个测量电路的场景。
        
        参数:
            c: 列索引 [0, num_measures)
            new_column_values: (M,) 新列值数组
            copy: 是否返回更新后的深拷贝（不修改原对象）
            
        返回:
            如果copy=True,返回更新后的新对象；否则返回None（就地修改）
            
        异常:
            ValueError: 如果输入数组形状不匹配
        """
        # 如果需要拷贝,创建新对象
        if copy:
            return self._update_column_copy(c, new_column_values)
        
        # 确保输入是NumPy数组
        if not isinstance(new_column_values, np.ndarray):
            new_column_values = np.array(new_column_values, dtype=np.float64)
        else:
            # 深拷贝输入数组,避免外部修改影响内部状态
            new_column_values = new_column_values.copy()
        
        # 验证输入形状
        if new_column_values.shape != (self.num_paulis,):
            raise ValueError(
                f"新值数组形状应为({self.num_paulis},),实际为{new_column_values.shape}"
            )
        
        # 更新列值
        self.pauli_weight[:, c] = new_column_values
        
        # 重新计算所有受影响行的乘积
        for r in range(self.num_paulis):
            self._row_products[r] = self._recompute_row_product(r)
        
        # 重新计算总成本
        self._cost_function_value = self._recompute_cost_from_products()
        
        return self
    
    def _update_column_copy(self, c: int, new_column_values: np.ndarray) -> 'PauliCostCalculator':
        """
        创建深拷贝并更新列（内部方法）
        
        参数:
            c: 列索引
            new_column_values: 新列值
            
        返回:
            更新后的新PauliCostCalculator对象
        """
        # 创建新对象
        new_calc = PauliCostCalculator.__new__(PauliCostCalculator)
        
        # 复制基本属性
        new_calc.num_paulis = self.num_paulis
        new_calc.num_measures = self.num_measures
        new_calc.epsilon = self.epsilon
        
        # 深拷贝数组
        new_calc.pauli_weight = self.pauli_weight.copy()
        new_calc.pauli_average_weights = self.pauli_average_weights.copy()
        new_calc._row_products = self._row_products.copy()
        new_calc._cost_function_value = self._cost_function_value
        
        # 在新对象上更新
        new_calc.update_column(c, new_column_values, copy=False)
        
        return new_calc
    
    # ========================================================================
    # 属性和工具方法
    # ========================================================================
    
    @property
    def cost_function_value(self) -> float:
        """返回当前同步的成本函数值"""
        return self._cost_function_value
    
    def copy(self) -> 'PauliCostCalculator':
        """
        创建当前对象的深拷贝
        
        返回:
            新的PauliCostCalculator对象,与原对象完全独立
        """
        new_calc = PauliCostCalculator.__new__(PauliCostCalculator)
        
        # 复制基本属性
        new_calc.num_paulis = self.num_paulis
        new_calc.num_measures = self.num_measures
        new_calc.epsilon = self.epsilon
        
        # 深拷贝所有数组
        new_calc.pauli_weight = self.pauli_weight.copy()
        new_calc.pauli_average_weights = self.pauli_average_weights.copy()
        new_calc._row_products = self._row_products.copy()
        new_calc._cost_function_value = self._cost_function_value
        
        return new_calc
    
    def __repr__(self):
        """返回对象的字符串表示"""
        return (f"PauliCostCalculator(M={self.num_paulis}, num_measures={self.num_measures}, "
                f"cost={self._cost_function_value:.4e})")

# ============================================================================
# DSS张量网络收缩函数
# ============================================================================

def _contract_single_qubit_chains(combined_tn, num_qubits, depth, mode):
    """
    收缩单量子比特张量链（垂直方向）
    
    对每个量子比特将其单qubit操作收缩到右边的双qubit门上面
    Pauli算符 -> 单比特门序列 -> 双比特门 从左往右
    measure  -> 单比特门 -> 双比特门 从右往左
    
    收缩后,张量网络中只有双qubit门
    
    参数:
        combined_tn: 合并的完整张量网络
        num_qubits: 量子比特数
        depth: 电路深度
        mode: 1=包含单比特门(4维), 2=忽略单比特门(2维)
        
    返回:
        收缩后的张量网络（保留双比特门之间的连接）
    """
    for q in range(num_qubits):
        # 该量子比特的起始和终止张量标签
        pauli_tags = {f'PauliR0Q{q}'}
        measure_tags = {f'measureR{depth}Q{q}'}
        
        if mode == 1:
            # -------------------- Mode 1: 包含单比特门 --------------------
            # 收缩顺序: Pauli -> G1 -> G2 -> G1 -> G2 -> ... -> G1 -> Measure
            
            # 步骤1: Pauli + 第一个单比特门
            combined_tn.contract_between(pauli_tags, {f'G1R0Q{q}'})
            
            # 步骤2: 逐层收缩 G1 -> G2
            for r in range(depth):
                g1_tags = {f'G1R{r}Q{q}'}
                g2_tags = {'G2', f'R{r}Q{q}'}
                combined_tn.contract_between(g1_tags, g2_tags)
            
            # 步骤3: 最后一个单比特门 + 测量
            final_g1_tags = {f'G1R{depth}Q{q}'}
            combined_tn.contract_between(final_g1_tags, measure_tags)
            
            # 步骤4: 连接到最后一层双比特门
            last_g2_tags = {'G2', f'R{depth-1}Q{q}'}
            combined_tn.contract_between(final_g1_tags, last_g2_tags)
            
        elif mode == 2:
            # -------------------- Mode 2: 忽略单比特门 --------------------
            # 收缩顺序: Pauli -> G2 -> G2 -> ... -> G2 -> Measure
            
            # 步骤1: Pauli连接到第一层双比特门
            combined_tn.contract_between(pauli_tags, {'G2', f'R0Q{q}'})
            
            # 步骤2: 测量连接到最后一层双比特门
            last_g2_tags = {'G2', f'R{depth-1}Q{q}'}
            combined_tn.contract_between(measure_tags, last_g2_tags)
    
    return combined_tn


def _generate_brickwall_contraction_path(start_qubit: int, 
                                         num_qubits: int, 
                                         max_depth: int):
    """
    生成Brickwall结构的阶梯式收缩路径
    
    为避免张量维数爆炸,采用阶梯式收缩策略：
    从起始量子比特开始,沿着"之"字形路径收缩相邻的双比特门。
    
    参数:
        start_qubit: 收缩路径的起始量子比特索引
        num_qubits: 总量子比特数
        max_depth: 电路最大深度
        
    产生:
        (layer, qubit): 下一个要收缩的双比特门位置
        
    路径示例 (start_qubit=0, num_qubits=4):
        Layer 0: q0 ━━━ q1
        Layer 1:        q1 ━━━ q2
        Layer 2:                 q2 ━━━ q3
        收缩路径: (0,0) -> (1,1) -> (2,2) -> (3,3) -> ...
    """
    import itertools
    
    if max_depth <= 0:
        return
    
    # 第一段路径: 从start_qubit-1降到0
    initial_path = list(range(start_qubit - 1, -1, -1))
    
    # 根据量子比特数决定是否需要循环
    if num_qubits % 2 == 0:
        # 偶数量子比特: 需要循环模式
        # 循环路径: num_qubits-1 -> 0 -> num_qubits-1 -> 0 -> ...
        cycle_path = range(num_qubits - 1, -1, -1)
        qubit_sequence = itertools.chain(initial_path, itertools.cycle(cycle_path))
    else:
        # 奇数量子比特: 只用初始路径,不循环
        qubit_sequence = iter(initial_path)
    
    # 将层索引(1到max_depth-1)与量子比特索引配对
    yield from zip(range(1, max_depth), qubit_sequence)


def _contract_brickwall_structure(tn, num_qubits, depth):
    """
    收缩Brickwall结构的双比特门
    
    在完成单量子比特链收缩后,剩余的张量呈现Brickwall砖墙结构：
        偶数层: (0-1), (2-3), (4-5), ...
        奇数层:   (1-2), (3-4), (5-6), ...
    
    采用阶梯式收缩策略,从多个起点同时进行,避免中间张量维数过大。
    
    参数:
        tn: 已完成单量子比特链收缩的张量网络
        num_qubits: 量子比特数
        depth: 电路深度
        
    返回:
        完全收缩后的张量网络(多个高维的张量M1 M2 M3 ...,最后只需要求Tr(M1 M2 M3))
    """
    if num_qubits % 2 == 0:
        # ==================== 偶数量子比特 ====================
        # 从每个偶数位置(0,2,4,...)发起阶梯式收缩
        
        for q in range(0, num_qubits, 2):
            # 当前收缩位置
            current_layer = 0
            current_qubit = q
            
            # 沿阶梯路径收缩
            for next_layer, next_qubit in _generate_brickwall_contraction_path(
                q, num_qubits, depth
            ):
                # 收缩当前位置和下一个位置的双比特门
                current_tags = {f'G2R{current_layer}Q{current_qubit}'}
                next_tags = {f'G2R{next_layer}Q{next_qubit}'}
                tn.contract_between(current_tags, next_tags)
                
                # 移动到下一个位置
                current_layer = next_layer
                current_qubit = next_qubit
    
    else:
        # ==================== 奇数量子比特 ====================
        
        # 第一部分: 处理所有非边界的偶数起点
        for q in range(0, num_qubits - 1, 2):
            current_layer = 0
            current_qubit = q
            
            for next_layer, next_qubit in _generate_brickwall_contraction_path(
                q, num_qubits, depth
            ):
                current_tags = {f'G2R{current_layer}Q{current_qubit}'}
                next_tags = {f'G2R{next_layer}Q{next_qubit}'}
                tn.contract_between(current_tags, next_tags)
                current_layer = next_layer
                current_qubit = next_qubit
        
        # 第二部分: 处理最后一个量子比特（边界情况）
        # 注意: 从layer=1开始,因为layer=0时最后一个qubit没有双比特门
        current_layer = 1
        current_qubit = num_qubits - 1
        
        for layer, qubit in _generate_brickwall_contraction_path(
            num_qubits - 1, num_qubits, depth - 1
        ):
            next_layer = layer + 1  # 调整层索引偏移
            next_qubit = qubit
            current_tags = {f'G2R{current_layer}Q{current_qubit}'}
            next_tags = {f'G2R{next_layer}Q{next_qubit}'}
            tn.contract_between(current_tags, next_tags)
            current_layer = next_layer
            current_qubit = next_qubit
    
    return tn


def contract_dss(pauli_tn, circuit_tn, measure_tn, num_qubits, depth=1, mode=1):
    """
    DSS张量网络完整收缩函数
    
    实现高效的两阶段收缩策略：
        阶段1 - 垂直收缩: 将每个量子比特的时间演化链收缩成柱状张量
        阶段2 - 水平收缩: 按阶梯模式收缩Brickwall结构的双比特门
    
    该收缩顺序有效控制了中间张量的维数,避免指数级增长。
    
    参数:
        pauli_tn: Pauli算符的张量网络（输入态）
        circuit_tn: 量子电路的张量网络（演化算符）
        measure_tn: 测量的张量网络（测量算符）
        num_qubits: 量子比特数
        depth: 电路深度（双比特门层数）
        mode: 1=完整4维Pauli表示, 2=简化2维表示
        
    返回:
        float: 测量该Pauli算符的概率值
    """
    # ==================== 阶段0: 合并所有张量网络 ====================
    # 创建副本以保护原始网络不被修改
    full_tn = pauli_tn.copy()
    full_tn |= circuit_tn.copy()
    full_tn |= measure_tn.copy()
    
    # ==================== 阶段1: 单量子比特链收缩 ====================
    # 将每个量子比特的垂直张量链收缩成单列
    # 收缩后只剩下双比特门之间的连接
    full_tn = _contract_single_qubit_chains(
        full_tn, 
        num_qubits=num_qubits, 
        depth=depth, 
        mode=mode
    )
    
    # ==================== 阶段2: Brickwall结构收缩 ====================
    # 按阶梯模式收缩双比特门
    result_tn = _contract_brickwall_structure(
        full_tn, 
        num_qubits=num_qubits, 
        depth=depth
    )
    
    # ==================== 阶段3: 提取最终结果 ====================
    if isinstance(result_tn, qtn.TensorNetwork):
        if result_tn.num_tensors == 1:
            # 只剩一个张量,直接提取数据
            result = np.array(result_tn.tensors[0])
        else:
            # 仍有多个张量,执行最终收缩
            result = np.array(result_tn.contract(optimize='greedy'))
    else:
        # 已经浮点数
        result = np.array(result_tn)
    
    return float(result)



def PauliWeightCalculator(pauli_tn, circuit_tn, measure_tn,num_qubits,depth, mode):
    """
    计算Pauli算符的测量概率权重
    
    通过DSS张量网络收缩方法,计算给定Pauli算符在特定电路配置下的测量概率。
    该函数自动从张量网络中推断量子比特数量。
    
    参数:
        pauli_tn: Pauli算符的张量网络
        circuit_tn: 量子电路的张量网络
        measure_tn: 测量的张量网络
        depth: 电路深度,默认为1
        mode: 计算模式 (1=4维完整, 2=2维简化)
        
    返回:
        float: Pauli权重 P_i(Pauli),范围[0, 1]
    """
    result = contract_dss(
        pauli_tn, 
        circuit_tn, 
        measure_tn, 
        num_qubits=num_qubits,
        depth=depth,
        mode=mode
    )
    
    return result

# ============================================================================
# 第五部分: DSS优化算法主类
# ============================================================================
class DSS:
    """
    Derandomized Shallow Shadows (DSS) 贪心优化算法
    
    通过贪心策略优化num_measures个确定性测量电路,最小化测量Pauli算符集合的总成本。
    
    优化策略:
        1. 对每个测量电路k,按顺序优化:
           - 双比特门(G2): 从右到左,使用Mode 2(2维简化)
           - 单比特门(G1): 从左到右,使用Mode 1(4维完整)
        2. 每个门位置采用贪心选择：尝试所有确定性门,选择成本最低的
    
    属性:
        pauli_collection: Pauli算符集合
        M: Pauli算符数量
        num_qubits: 量子比特数
        num_measures: 测量电路数量
        D: 电路深度
        epsilon: 成本函数超参数
        circuit_configs: num_measures个电路配置对象
        cost_calculator: 成本函数计算器
    """
    
    def __init__(self, 
                 pauli_collection: PauliOperatorCollection,
                 circuit_depth: int,
                 num_measurements: int,
                 epsilon: float = 0.1):
        """
        初始化DSS优化器
        
        参数:
            pauli_collection: 待测量的M个Pauli算符及其权重w_P
            circuit_depth: 电路深度(双比特门层数)
            num_measurements: 测量电路数量(num_measures)
            epsilon: 成本函数超参数
            
        异常:
            ValueError: 如果Pauli集合为空或量子比特数未定义
        """
        # -------------------- 基本参数 --------------------
        self.pauli_collection = pauli_collection
        self.num_paulis = len(pauli_collection)              # Pauli算符数量
        self.num_qubits = pauli_collection.num_qubits   # 量子比特数
        self.num_measures = num_measurements                   # 测量电路数
        self.depth = circuit_depth                      # 电路深度
        self.epsilon = epsilon
        
        # 参数验证
        if self.num_qubits is None or self.num_paulis == 0:
            raise ValueError("Pauli集合不能为空且必须定义量子比特数")
        
        # 打印初始化信息
        self._print_header("DSS 优化器初始化")
        print(f"  量子比特数 (num_qubits)    : {self.num_qubits}")
        print(f"  电路深度 (Depth)      : {self.depth}")
        print(f"  测量电路数 (num_measures)    : {self.num_measures}")
        print(f"  Pauli算符数 (num_paulis)   : {self.num_paulis}")
        print(f"  超参数 (ε)        : {self.epsilon}")
        print()
        
        # -------------------- 初始化电路配置 --------------------
        # 创建num_measures个电路,所有门初始化为0(随机/Clifford平均)
        self.circuit_configs = [
            CircuitConfig(num_qubits=self.num_qubits, circuit_depth=self.depth) 
            for _ in range(self.num_measures)
        ]
        
        # -------------------- 预构建张量网络 --------------------
        print("  ⏳ 预构建张量网络...")
        
        # Mode 2 (2维简化): 用于优化双比特门
        self.pauli_tns_mode2 = [
            paulistring2tn(op, mode=2) for op in self.pauli_collection
        ]
        self.measure_tn_mode2 = measure2tn(
            depth=self.depth, 
            num_qubits=self.num_qubits, 
            mode=2
        )
        
        # Mode 1 (4维完整): 用于优化单比特门
        self.pauli_tns_mode1 = [
            paulistring2tn(op, mode=1) for op in self.pauli_collection
        ]
        self.measure_tn_mode1 = measure2tn(
            depth=self.depth, 
            num_qubits=self.num_qubits, 
            mode=1
        )
        print("  ✓ 张量网络构建完成")
        
        # -------------------- 初始化成本计算器 --------------------
        print("  ⏳ 初始化成本计算器...")
        
        # 计算初始Pauli权重(全0门对应的随机测量)
        initial_PauliWeights = self._calculate_pauli_weights_for_circuit(
            self.circuit_configs[0], 
            mode=2
        )
        
        # 创建成本计算器
        self.cost_calculator = PauliCostCalculator(
            weight=self.pauli_collection.get_weight(),
            epsilon=self.epsilon,
            num_measurements=self.num_measures
        )
        
        # 用相同的初始权重填充所有num_measures列
        for k in range(self.num_measures):
            self.cost_calculator.update_column(k, initial_PauliWeights)
        
        initial_cost = self.cost_calculator.cost_function_value
        print(f"  ✓ 成本计算器初始化完成")
        print(f"  📊 初始成本: {initial_cost:.6e}")
        self._print_separator()
    
    # ========================================================================
    # 输出格式化辅助方法
    # ========================================================================
    
    @staticmethod
    def _print_header(title: str):
        """打印标题头"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    
    @staticmethod
    def _print_separator(char: str = '='):
        """打印分隔线"""
        print(f"{char*70}")
    
    @staticmethod
    def _print_section(title: str):
        """打印章节标题"""
        print(f"\n{'-'*70}")
        print(f"  {title}")
        print(f"{'-'*70}")
    
    @staticmethod
    def _print_progress(current: int, total: int, prefix: str = "", 
                       cost: float = None, extra: str = ""):
        """打印进度信息"""
        percentage = 100 * current / total
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        progress_str = f"  {prefix} [{bar}] {current}/{total} ({percentage:.1f}%)"
        if cost is not None:
            progress_str += f" | 成本: {cost:.6e}"
        if extra:
            progress_str += f" | {extra}"
        print(progress_str)
    
    # ========================================================================
    # 内部辅助方法
    # ========================================================================
    
    def _calculate_pauli_weights_for_circuit(self, 
                                             circuit_config: CircuitConfig, 
                                             mode: int) -> np.ndarray:
        """
        计算单个电路配置下所有Pauli算符的测量概率
        
        对给定的电路配置,计算集合中每个Pauli算符的测量概率p_i(P)。
        
        参数:
            circuit_config: 待评估的电路配置
            mode: 1=4维完整表示, 2=2维简化表示
            
        返回:
            np.ndarray: (M,) 概率数组 [p_i(P_0), p_i(P_1), ..., p_i(P_{M-1})]
            
        异常:
            ValueError: 如果mode不是1或2
        """
        # 1. 将电路配置转换为张量网络
        circuit_tn = circuit2tn(circuit_config, mode=mode)
        
        # 2. 选择对应模式的测量和Pauli张量
        if mode == 1:
            measure_tn = self.measure_tn_mode1
            pauli_tns = self.pauli_tns_mode1
        elif mode == 2:
            measure_tn = self.measure_tn_mode2
            pauli_tns = self.pauli_tns_mode2
        else:
            raise ValueError("mode必须是1或2")
        
        # 3. 计算所有M个Pauli算符的权重
        PauliWeights_column = np.zeros(self.num_paulis, dtype=np.float64)
        
        for m in range(self.num_paulis):
            pauli_tn = pauli_tns[m]
            
            # 收缩张量网络得到概率
            Prob = PauliWeightCalculator(
                pauli_tn, 
                circuit_tn, 
                measure_tn,
                depth=self.depth,
                num_qubits=self.num_qubits,
                mode=mode
            )
            PauliWeights_column[m] = Prob
        
        return PauliWeights_column
    
    # ========================================================================
    # 主优化算法
    # ========================================================================
    
    def run(self) -> List[CircuitConfig]:
        """
        运行完整的DSS贪心优化算法
        
        算法流程:
            对每个测量电路k (k=0,1,...,num_measures-1):
                阶段1: 优化所有双比特门(G2) - 从右到左,使用Mode 2
                阶段2: 优化所有单比特门(G1) - 从左到右,使用Mode 1
            
            每个门位置的优化:
                1. 尝试所有确定性门选项
                2. 计算每个选项的成本
                3. 选择成本最低的门
        
        返回:
            List[CircuitConfig]: num_measures个优化后的电路配置
        """
        self._print_header("开始 DSS 优化")
        # print(f"  总电路数: {self.num_measures}")
        # print(f"  每个电路: {self.depth} 层深度 x {self.num_qubits} 量子比特")
        self._print_separator()
        
        # -------------------- 定义门选项 --------------------
        # 双比特门: I2, SWAP, CNOT (排除0=随机) 按照实验难度排序
        gate2q_option_ints = [3, 2, 1]
        # 双比特门: I2, SWAP, CNOT
        # gate2q_option_ints = [1, 2, 3]
        gate2q_option_names = {
            i: MatConfigBase.INT_TO_GATE_2[i] for i in gate2q_option_ints
        }
        
        # # 单比特门: I1, H, S, HSH, SH, HS (排除0=随机) 按照实验难度排序
        gate1q_option_ints = [4,6,5,3,2,1]
        # 单比特门: I1, H, S, HSH, SH, HS (排除0=随机) 按照实验难度排序
        # gate1q_option_ints = [1,2,3,4,5,6]
        gate1q_option_names = {
            i: MatConfigBase.INT_TO_GATE_1[i] for i in gate1q_option_ints
        }
        
        # -------------------- 主优化循环 --------------------
        for k in range(self.num_measures):
            # self._print_section(f"电路 {k+1}/{self.num_measures} 优化中")
            
            current_circuit = self.circuit_configs[k].copy()
            best_cost = self.cost_calculator.cost_function_value
            cost_calculator = self.cost_calculator.copy()
            
            # ========== 阶段1: 优化双比特门(G2, Mode 2) ==========
            # print(f"\n  🔵 阶段1: 双比特门优化 (Mode 2)")
            # print(f"  策略: 从右到左 | 门选项: {len(gate2q_option_ints)}个")
            
            # 收集所有双比特门位置
            gate2q_positions = []
            for (r, q), _ in current_circuit.iter_gate2q():
                gate2q_positions.append((r, q))
            
            # 按层降序排序(从右到左)
            gate2q_positions.sort(key=lambda x: x[0], reverse=True)
            print(f"  待优化门数: {len(gate2q_positions)}")
            
            # 逐个优化双比特门
            for i, (r, q) in enumerate(gate2q_positions):

                # 同步成本函数
                cost_calculator = self.cost_calculator.copy()
                # 初始化最优解
                best_gate = current_circuit.get_gate2q(r, q)
                best_Prob = self.cost_calculator.pauli_weight[:, k].copy()
                
                # 贪心搜索所有门选项
                for gate_int in gate2q_option_ints:
                    # 临时设置新门
                    current_circuit.set_gate2q(r, q, gate_int)
                    
                    # 重新计算Pauli权重
                    new_Prob = self._calculate_pauli_weights_for_circuit(
                        current_circuit, mode=2
                    )
                    
                    # 临时更新成本
                    cost_calculator.update_column(k, new_Prob)
                    trial_cost = cost_calculator.cost_function_value

                    # # ===== 新增：计算概率向量的变化 =====
                    # prob_diff = np.linalg.norm(new_Prob - best_Prob)
                    # prob_threshold = 1e-10
                    # or prob_diff < prob_threshold
                    
                    # #测试 强制替换随机门
                    if gate_int == gate2q_option_ints[0]:
                        best_gate=gate_int
                        best_Prob=new_Prob
                        best_cost = trial_cost

                    # 贪心选择
                    if trial_cost <= best_cost:
                    # if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_gate = gate_int
                        best_Prob = new_Prob
                    
                
                # 永久设置最优门
                self.circuit_configs[k].set_gate2q(r, q, best_gate)
                # 重新获取当前电路配置
                current_circuit = self.circuit_configs[k].copy()
                # 永久更新成本
                self.cost_calculator.update_column(k, best_Prob)
                
                
            
            # G2优化完成
            g2_cost = self.cost_calculator.cost_function_value
            # print(f"  ✓ 双比特门优化完成 | 成本: {g2_cost:.6e}")
            
            # ========== 阶段2: 优化单比特门(G1, Mode 1) ==========
            # print(f"\n  🟢 阶段2: 单比特门优化 (Mode 1)")
            # print(f"  策略: 从左到右 | 门选项: {len(gate1q_option_ints)}个")

            # 收集所有单比特门位置
            gate1q_positions = []
            for (r, q), _ in current_circuit.iter_gate1q():
                gate1q_positions.append((r, q))
            
            # 按层升序排序(从左到右)
            gate1q_positions.sort(key=lambda x: x[0], reverse=False)
            print(f"  待优化门数: {len(gate1q_positions)}")
            
            # 逐个优化单比特门
            for i, (r, q) in enumerate(gate1q_positions):

                # 同步成本函数
                cost_calculator = self.cost_calculator.copy()
                # 初始化最优解
                best_Prob = self.cost_calculator.pauli_weight[:, k].copy()
                best_gate = current_circuit.get_gate1q(r, q)
                
                # 贪心搜索所有门选项
                for gate_int in gate1q_option_ints:
                    # 临时设置新门
                    current_circuit.set_gate1q(r, q, gate_int)
                    
                    # 重新计算Pauli权重(包含已固定的G2门)
                    new_Prob = self._calculate_pauli_weights_for_circuit(
                        current_circuit, mode=1
                    )
                    
                    # 临时更新成本
                    cost_calculator.update_column(k, new_Prob)
                    trial_cost = cost_calculator.cost_function_value

                    # # ===== 新增：计算概率向量的变化 =====
                    # prob_diff = np.linalg.norm(new_Prob - best_Prob)
                    # prob_threshold = 1e-10
                    # or prob_diff < prob_threshold

                    #测试 强制替换随机门
                    if gate_int == gate1q_option_ints[0]:
                        best_gate=gate_int
                        best_Prob=new_Prob
                        best_cost = trial_cost
                    
                    # 贪心选择
                    if trial_cost <= best_cost:
                    # if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_gate = gate_int
                        best_Prob = new_Prob

                # 永久设置最优门
                self.circuit_configs[k].set_gate1q(r, q, best_gate)
                # 重新获取当前电路配置
                current_circuit = self.circuit_configs[k].copy()
                # 永久更新成本
                self.cost_calculator.update_column(k, best_Prob)
                
                
                
                # 定期输出进度
                if (i + 1) % 20 == 0 or (i + 1) == len(gate1q_positions):
                    gate_name = gate1q_option_names.get(best_gate, 'Rand')
                    self._print_progress(
                        i + 1, 
                        len(gate1q_positions),
                        prefix="G1",
                        cost=best_cost,
                        extra=f"当前门: {gate_name}"
                    )
            
            # G1优化完成
            final_cost = self.cost_calculator.cost_function_value
            # print(f"  ✓ 单比特门优化完成 | 成本: {final_cost:.6e}")
            
            # 电路优化总结
            cost_reduction = (g2_cost - final_cost) / g2_cost * 100 if g2_cost > 0 else 0
            # print(f"\n  📈 电路 {k+1} 优化总结:")
            # print(f"     - G2后成本: {g2_cost:.6e}")
            # print(f"     - 最终成本: {final_cost:.6e}")
            # print(f"     - G1改善: {cost_reduction:.2f}%")
        
        # -------------------- 优化完成 --------------------
        final_cost = self.cost_calculator.cost_function_value
        self._print_header("DSS 优化完成")
        print(f"  ✅ 成功优化 {self.num_measures} 个测量电路")
        print(f"  📊 最终成本: {final_cost:.6e}")
        self._print_separator()
        print()
        
        return self.circuit_configs
    
    # ========================================================================
    # 结果保存
    # ========================================================================
    
    def save_results(self, 
                     folder_name: str = "dss_results", 
                     report_name: str = "optimization_report.md"):
        """
        保存优化结果并生成报告
        
        生成内容:
            1. SVG格式的电路图像(保存在circuits子目录)
            2. Markdown格式的优化报告(包含成本、门配置等信息)
        
        输出目录结构:
            folder_name/
            ├── circuits/
            │   ├── circuit_0.svg
            │   ├── circuit_1.svg
            │   └── ...
            └── optimization_report.md
        
        参数:
            folder_name: 结果文件夹名称
            report_name: Markdown报告文件名
        """
        # -------------------- 创建输出目录 --------------------
        img_folder = os.path.join(folder_name, "circuits")
        os.makedirs(img_folder, exist_ok=True)
        
        self._print_header("保存优化结果")
        print(f"  📁 保存路径: {os.path.abspath(folder_name)}")
        print(f"  📄 报告文件: {report_name}")
        print()
        
        # -------------------- 生成报告头部 --------------------
        md_content = []
        md_content.append("# DSS优化报告\n")
        md_content.append(f"## 📋 配置参数\n")
        md_content.append(f"| 参数 | 值 |")
        md_content.append(f"|------|-----|")
        md_content.append(f"| 量子比特数 | {self.num_qubits} |")
        md_content.append(f"| 电路深度 | {self.depth} |")
        md_content.append(f"| 测量电路数 | {self.num_measures} |")
        md_content.append(f"| Pauli算符数 | {self.num_paulis} |")
        md_content.append(f"| 超参数 ε | {self.epsilon} |")
        md_content.append(f"\n## 🎯 优化结果\n")
        md_content.append(
            f"**最终成本函数值:** `{self.cost_calculator.cost_function_value:.6e}`\n"
        )
        
        # -------------------- 处理每个电路 --------------------
        print(f"  ⏳ 正在生成电路图像和报告...")
        from drawcircuit import draw_circuit
        for k, config in enumerate(self.circuit_configs):
            # 进度显示
            self._print_progress(k + 1, self.num_measures, prefix="电路处理")
            
            # === 生成SVG图像 ===
            svg_filename = f"circuit_{k}.svg"
            svg_relative_path = os.path.join("circuits", svg_filename)
            svg_full_path = os.path.join(img_folder, svg_filename)
            
            try:
                draw_circuit(config, svg_full_path)
            except Exception as e:
                print(f"    ⚠️  无法绘制电路 {k}: {e}")

            # === 生成PDF图像 ===
            pdf_filename = f"circuit_{k}.pdf"
            pdf_relative_path = os.path.join("pdf_circuits", pdf_filename)
            pdf_full_path = os.path.join(img_folder, pdf_filename)

            try:
                draw_circuit(config, pdf_full_path)
            except Exception as e:
                print(f"    ⚠️  无法绘制电路 {k}: {e}")
            
            # === 收集门配置信息 ===
            md_content.append(f"\n---\n")
            md_content.append(f"## 电路 {k+1}\n")
            md_content.append(f"**电路图:** [🔗 查看SVG](./{svg_relative_path})\n")
            md_content.append(f"**电路图:** [🔗 查看PDF](./{pdf_relative_path})\n")
            
            # --- 统计门数量 ---
            g1_count = sum(1 for (r, q), gt in config.iter_gate1q() 
                          if gt not in [0, 1])
            g2_count = sum(1 for (r, q), gt in config.iter_gate2q() 
                          if gt not in [0, 1])
            
            md_content.append(f"**统计:** ")
            md_content.append(f"{g1_count} 个确定性单比特门, ")
            md_content.append(f"{g2_count} 个确定性双比特门\n")
            
            # --- 单比特门配置 ---
            md_content.append("### 🔹 单比特门 (G1)\n")
            g1_details = []
            for (r, q), gate_type in config.iter_gate1q():
                if gate_type not in [0, 1]:
                    gate_name = MatConfigBase.INT_TO_GATE_1.get(
                        gate_type, 'Unknown'
                    )
                    g1_details.append(f"- Layer {r}, Qubit {q}: `{gate_name}`")
            
            if g1_details:
                md_content.extend(g1_details)
            else:
                md_content.append("- *(无特殊门)*")
            
            # --- 双比特门配置 ---
            md_content.append("\n### 🔸 双比特门 (G2)\n")
            g2_details = []
            for (r, q), gate_type in config.iter_gate2q():
                if gate_type not in [0, 1]:
                    gate_name = MatConfigBase.INT_TO_GATE_2.get(
                        gate_type, 'Unknown'
                    )
                    q2 = (q + 1) % config.num_qubits
                    g2_details.append(
                        f"- Layer {r}, Qubits ({q}, {q2}): `{gate_name}`"
                    )
            
            if g2_details:
                md_content.extend(g2_details)
            else:
                md_content.append("- *(无特殊门)*")
        
        print()  # 换行
        
        # -------------------- 写入报告文件 --------------------
        report_full_path = os.path.join(folder_name, report_name)
        try:
            with open(report_full_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_content))
            print(f"  ✓ 报告生成成功: {os.path.basename(report_full_path)}")
        except Exception as e:
            print(f"  ✗ 报告生成失败: {e}")
        
        # -------------------- 总结 --------------------
        print()
        print(f"  📊 生成统计:")
        print(f"     - 电路图像: {self.num_measures} 个")
        print(f"     - Markdown报告: 1 个")
        print(f"     - 总文件大小: ~{self._estimate_folder_size(folder_name)}")
        
        self._print_separator()
        print(f"  ✅ 所有结果已保存至: {os.path.abspath(folder_name)}")
        self._print_separator()
        print()
    
    @staticmethod
    def _estimate_folder_size(folder_path: str) -> str:
        """估算文件夹大小"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            # 转换为合适的单位
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
        except:
            return "未知"