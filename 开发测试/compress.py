import math
import torch
from typing import List, Tuple, Dict

class DeepGradientCompression:
    def __init__(self, momentum_buffer_size=32, compress_ratio=0.02, 
                 communication_threshold=0.001,
                 momentum_factor=0.9):

        """
        Args:
            momentum_buffer_size (int): 动量缓冲区大小
            compress_ratio (float): 压缩比率
            communication_threshold (float): 通信阈值
            momentum_factor (float): 动量因子
        """

        self.momentum_buffer_size = momentum_buffer_size
        self.compress_ratio = compress_ratio
        self.communication_threshold = communication_threshold
        self.momentum_factor = momentum_factor
        
        self.momentum_states = {}

        # 可压缩的最小元素数量，若小于这个数量的原始数据，取最大的一个梯度进行更新
        self.min_raw_elements = int(math.ceil(1 / compress_ratio))

    def clear(self):
        self.momentum_states = {}

    def compress_shape(self, original_shape):
        """计算压缩后的形状, 确保压缩后长度不为0

        compress_ratio = 0.02
        min_raw_elements = 50

        1. raw_elements = 49
            compressed_size = 1

        2. raw_elements = 50
            compressed_size = 1
        """
        raw_elements = math.prod(original_shape)
        if raw_elements < self.min_raw_elements:
            return (1, )
        compressed_size = int(math.prod(original_shape) * self.compress_ratio)
        return (compressed_size,)

    def compress(self, gradients, need_warm_up_steps=False):
        """压缩梯度，确保输出大小固定且不小于最小元素数量"""

        compressed_gradients = []
        compression_infos = []

        # 预热期间不压缩
        if need_warm_up_steps:
            for idx, gradient in enumerate(gradients):
                param_name = f'grad_{idx}'

                # 初始化动量状态
                if param_name not in self.momentum_states:
                    self.momentum_states[param_name] = torch.zeros_like(gradient)
                
                # 计算新的动量
                momentum_grad = (self.momentum_factor * self.momentum_states[param_name]) + gradient
                self.momentum_states[param_name] = momentum_grad
                
                compression_info = {
                    'is_full_gradient': True,
                }
                
                compressed_gradients.append(momentum_grad)
                compression_infos.append(compression_info)
            
        else:
            # 处理每个梯度
            for idx, gradient in enumerate(gradients):
                param_name = f'grad_{idx}'

                # 检查梯度 inf/nan
                if torch.isnan(gradient).any() or torch.isinf(gradient).any():
                    raise ValueError(f"梯度 {param_name} 包含 inf/nan 值")

                # 扁平化梯度
                flat_grad = gradient.view(-1)

                # 初始化动量状态
                if param_name not in self.momentum_states or self.momentum_states[param_name].shape != flat_grad.shape:
                    if param_name in self.momentum_states:
                        log(f"param {param_name} shape {self.momentum_states[param_name].shape} change to {flat_grad.shape}, most likely due to no more need warm up")
                    self.momentum_states[param_name] = torch.zeros_like(flat_grad)
                
                # 计算新的动量
                momentum_grad = (self.momentum_factor * self.momentum_states[param_name]) + flat_grad
                self.momentum_states[param_name] = momentum_grad
                
                # 计算阈值
                abs_grad = torch.abs(momentum_grad)
                threshold = max(
                    torch.quantile(abs_grad, 1 - self.compress_ratio), 
                    self.communication_threshold
                )
            
                # 获取重要梯度的掩码和索引
                important_mask = abs_grad >= threshold
                important_indices = torch.nonzero(important_mask).squeeze()

                # 校正索引
                compressed_size = self.compress_shape(gradient.shape)[0]

                if important_indices.numel() < compressed_size:
                    # 没有重要梯度 / 重要梯度数量不足
                    # 选取 topk k = compressed_size
                    topk = torch.topk(abs_grad, compressed_size)
                    important_indices = topk.indices

                elif important_indices.numel() > compressed_size:
                    # 随机抽取降采样
                    important_indices = important_indices[torch.randperm(len(important_indices))[:compressed_size]]

                # 获取重要梯度
                important_grad = momentum_grad[important_indices]
                
                compression_info = {
                    'indices': important_indices,
                    'is_full_gradient': False,
                    'original_shape': gradient.shape
                }
                
                compressed_gradients.append(important_grad)
                compression_infos.append(compression_info)

        return compressed_gradients, compression_infos

    def decompress(self, compressed_grads, compression_infos):
        decompressed_gradients = []

        for compressed_grad, comp_info in zip(compressed_grads, compression_infos):
            # 如果是全梯度,不需要解压
            if comp_info['is_full_gradient']:
                decompressed_gradients.append(compressed_grad)
                continue

            # 创建零张量
            full_gradient = torch.zeros(
                math.prod(comp_info['original_shape']), 
                device=compressed_grad.device
            )
            
            # 填充压缩后的梯度
            full_gradient[comp_info['indices']] = compressed_grad
            
            # 恢复原始形状
            decompressed_gradients.append(full_gradient.view(comp_info['original_shape']))
        
        return decompressed_gradients

CompressInfo = Dict[str, List[torch.Tensor]]

class IncrementalCompressor:
    def __init__(self, 
                 threshold: float = 1e-3,
                 sparsity_threshold: float = 0.3  # 稀疏度阈值，超过则全量更新
                ):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
            sparsity_threshold: 稀疏度阈值，当更新元素比例超过此值时切换为全量更新
        """
        self.threshold = threshold
        self.sparsity_threshold = sparsity_threshold
        self.client_params = {}  # 存储不同客户端的参数 {client_id: List[tensor]}
        
    def _init_reference(self, 
                       client_id: str,
                       tensors: List[torch.Tensor],
                      ) -> None:
        """初始化参考张量"""
        if client_id not in self.client_params:
            self.client_params[client_id] = [t.clone().detach() for t in tensors]
            return True
        
    def compress(self, 
                tensors: List[torch.Tensor],
                client_id: str,
               ) -> Tuple[List[torch.Tensor], CompressInfo]:
        """压缩张量列表"""
        init = self._init_reference(client_id, tensors)
        if init:
            return tensors, {'full': True}
        
        compressed_tensors = []
        compress_info = {
            'update_indices': [],
            'full': []
        }
        
        for curr_t, last_t in zip(tensors, self.client_params[client_id]):
            # 计算变化量
            diff = torch.abs(curr_t - last_t)
            mask = diff > self.threshold
            
            # 计算更新比例
            update_ratio = mask.sum().item() / mask.numel()
            
            # 根据更新比例决定使用全量更新还是增量更新
            if update_ratio > self.sparsity_threshold:
                # 全量更新
                compressed_tensors.append(curr_t)
                compress_info['full'].append(True)
                compress_info['update_indices'].append(None)
                last_t[:] = curr_t[:]
            else:
                # 增量更新
                update_indices = torch.where(mask)
                update_values = curr_t[mask]
                
                compressed_tensors.append(update_values)
                compress_info['update_indices'].append(torch.stack(update_indices, dim=1))
                compress_info['full'].append(False)
                
                # 更新参考张量
                last_t[mask] = curr_t[mask]
            
        return compressed_tensors, compress_info
    
    @staticmethod
    def decompress(
                  compressed_tensors: List[torch.Tensor],
                  compress_info: CompressInfo,
                  param_dict: Dict[str, torch.Tensor]
                ) -> None:
        """解压张量列表并直接更新参数字典"""
        param_names = list(param_dict.keys())
        
        if isinstance(compress_info.get('full'), bool):
            # 全部全量更新
            for param_name, compressed_t in zip(param_names, compressed_tensors):
                param_dict[param_name][:] = compressed_t[:]
            return
            
        # 混合更新模式
        for param_name, compressed_t, is_full, indices in zip(
            param_names,
            compressed_tensors,
            compress_info['full'],
            compress_info['update_indices']
        ):
            if is_full:
                # 全量更新
                param_dict[param_name][:] = compressed_t[:]
            else:
                # 增量更新
                if indices.numel() > 0:
                    param_dict[param_name][tuple(indices[:, i] for i in range(indices.shape[1]))] = compressed_t