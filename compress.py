import torch
import math
from collections import OrderedDict


class GradientCompressor:
    """
    梯度压缩器
    
    原始数据是 torch.Tensor
    压缩数据是 torch.Tensor
    """
    def __init__(self, sparsity_ratio=0.1, quantize_bits=8, pad_value=None, min_elements=0):
        """
        初始化梯度压缩器
        
        参数:
        sparsity_ratio: float - 要保留的梯度比例，默认0.1表示保留10%的梯度
        quantize_bits: int - 量化位数，默认8位
        pad_value: int - 填充值，默认为(2**quantize_bits - 1)
        min_elements: int - 压缩后最小元素数量，默认为使 int(n * sparsity_ratio)>=1 的最小n
        """
        self.sparsity_ratio = sparsity_ratio
        self.quantize_bits = quantize_bits
        self.pad_value = pad_value if pad_value is not None else (2**quantize_bits - 1)
        self.min_elements = min_elements if min_elements > 0 else int(math.ceil(1 / sparsity_ratio))
        
    def _pad_to_size(self, arr, target_size):
        """
        将张量填充或截断到目标大小
        """
        if arr.numel() >= target_size:
            return arr[:target_size]
        padded = torch.full((target_size,), self.pad_value, dtype=arr.dtype, device=arr.device)
        padded[:arr.numel()] = arr
        return padded
                
    def compress_shape(self, original_shape):
        """计算压缩后的形状，确保最小元素数量"""
        compressed_size = max(int(math.prod(original_shape) * self.sparsity_ratio), self.min_elements)
        return (compressed_size,)

    def compress(self, gradients):
        """压缩梯度，确保输出大小固定且不小于最小元素数量"""
        compressed_grads = []
        compress_info = []

        for grad in gradients:
            grad_flat = grad.flatten()
            grad_size = grad.numel()
            target_size = max(int(grad_size * self.sparsity_ratio), self.min_elements)

            # 处理小梯度的情况
            if grad_size <= target_size:
                quantized_values = torch.zeros(target_size, dtype=torch.uint8, device=grad.device)
                if grad_size > 0:
                    min_val = grad_flat.min().item()
                    max_val = grad_flat.max().item()
                    if min_val == max_val:
                        scale = 1.0
                        quantized_values[:grad_size] = 0
                    else:
                        scale = (max_val - min_val) / (2 ** self.quantize_bits - 1)
                        quantized_values[:grad_size] = torch.round((grad_flat - min_val) / scale).to(torch.uint8)
                else:
                    min_val = 0
                    scale = 1.0

                compress_info.append({
                    'shape': grad.shape,
                    'indices': torch.arange(grad_size, device=grad.device).tolist(),
                    'min_val': min_val,
                    'scale': scale,
                    'is_dominant_compressed': False,
                    'is_small_gradient': True,
                    'valid_size': grad_size
                })
                compressed_grads.append(quantized_values)
                continue

            # 处理主导值的情况
            unique_vals, counts = torch.unique(grad_flat, return_counts=True)
            total_elements = grad_size
            dominant_val_mask = counts / total_elements > (1 - self.sparsity_ratio)

            if dominant_val_mask.any():
                dominant_val = unique_vals[dominant_val_mask][0].item()
                mask = grad_flat != dominant_val
                indices = mask.nonzero().flatten()
                values = grad_flat[indices]

                if values.numel() > 0:
                    min_val = values.min().item()
                    max_val = values.max().item()
                    scale = (max_val - min_val) / (2 ** self.quantize_bits - 1)
                    quantized_values = torch.round((values - min_val) / scale).to(torch.uint8)
                else:
                    quantized_values = torch.tensor([], dtype=torch.uint8, device=grad.device)
                    min_val = dominant_val
                    scale = 1.0

                quantized_values = self._pad_to_size(quantized_values, target_size)
                
                compress_info.append({
                    'shape': grad.shape,
                    'indices': indices.tolist(),
                    'min_val': min_val,
                    'scale': scale,
                    'dominant_val': dominant_val,
                    'is_dominant_compressed': True,
                    'is_small_gradient': False,
                    'valid_size': min(values.numel(), target_size)
                })
            else:
                # 处理标准压缩的情况
                threshold = torch.quantile(torch.abs(grad_flat), 1 - self.sparsity_ratio)
                mask = torch.abs(grad_flat) >= threshold
                indices = mask.nonzero().flatten()
                values = grad_flat[indices]

                min_val = values.min().item()
                max_val = values.max().item()
                scale = (max_val - min_val) / (2 ** self.quantize_bits - 1)
                quantized_values = torch.round((values - min_val) / scale).to(torch.uint8)
                quantized_values = self._pad_to_size(quantized_values, target_size)

                compress_info.append({
                    'shape': grad.shape,
                    'indices': indices.tolist(),
                    'min_val': min_val,
                    'scale': scale,
                    'is_dominant_compressed': False,
                    'is_small_gradient': False,
                    'valid_size': min(values.numel(), target_size)
                })

            compressed_grads.append(quantized_values)

        return compressed_grads, compress_info

    def decompress(self, compressed_grads, compress_info):
        """解压梯度"""
        decompressed_grads = []

        for quantized_values, info in zip(compressed_grads, compress_info):
            # 创建目标梯度张量
            grad = torch.zeros(info['shape'], dtype=torch.float32, device=quantized_values.device)
            valid_values = quantized_values[:info['valid_size']].to(torch.float32)
            
            if info.get('is_small_gradient', False):
                if valid_values.numel() > 0:
                    values = valid_values * info['scale'] + info['min_val']
                    grad.view(-1)[:info['valid_size']] = values
            else:
                indices = torch.tensor(info['indices'][:info['valid_size']], 
                                    dtype=torch.long, 
                                    device=quantized_values.device)
                
                if info['is_dominant_compressed']:
                    grad.fill_(info['dominant_val'])
                    if valid_values.numel() > 0:
                        values = valid_values * info['scale'] + info['min_val']
                        grad.view(-1)[indices] = values
                else:
                    if valid_values.numel() > 0:
                        values = valid_values * info['scale'] + info['min_val']
                        grad.view(-1)[indices] = values

            decompressed_grads.append(grad)

        return decompressed_grads

    def get_compression_stats(self, original_grads, compressed_grads, compress_info):
        """计算压缩统计信息"""
        original_size = sum(grad.numel() * grad.element_size() for grad in original_grads)
        compressed_size = sum(grad.numel() * grad.element_size() for grad in compressed_grads)
        
        # 计算压缩信息的大小
        info_size = sum(
            len(str(info['indices'])) + 
            len(str(info['shape'])) + 
            len(str(info['min_val'])) + 
            len(str(info['scale'])) + 
            len(str(info.get('dominant_val', ''))) + 
            len(str(info['valid_size']))
            for info in compress_info
        )
        compressed_size += info_size

        valid_elements = sum(info['valid_size'] for info in compress_info)
        total_elements = sum(grad.numel() for grad in compressed_grads)

        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else float('inf'),
            'memory_saving_percentage': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
            'valid_elements': valid_elements,
            'total_elements': total_elements,
            'padding_percentage': (total_elements - valid_elements) / total_elements * 100 if total_elements > 0 else 0
        }

class ParamCompressor:
    """
    参数压缩器
    输入输出都是 torch.Tensor
    """
    def __init__(self, param_keys=None, quantize_bits=8):
        self.quantize_bits = quantize_bits
        self.param_keys = param_keys

    def compress_param(self, param):
        """压缩单个参数张量
        参数:
            param: torch.Tensor

        返回:
            压缩后的参数[torch.Tensor]，以及压缩信息字典
        """
        # 记录原始形状
        original_shape = param.shape
        
        # 展平数组以便处理
        flat_param = param.reshape(-1)
        
        # 计算量化参数
        min_val = flat_param.min().float()
        max_val = flat_param.max().float()
        
        # 避免零除错误
        if max_val == min_val:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = ((max_val - min_val) / (2**self.quantize_bits - 1)).float()
        
        # 确保 scale 有一个最小值，以避免数值溢出
        min_scale = 1e-8  # 你可以根据需要调整这个值
        scale = torch.max(scale, torch.tensor(min_scale, dtype=torch.float32))
        
        # 量化并使用 torch.clamp 确保结果在 uint8 范围内
        quantized = torch.round((flat_param - min_val) / scale)
        quantized = torch.clamp(quantized, 0, 2**self.quantize_bits - 1).byte()
        
        compress_info = {
            'shape': original_shape,
            'min_val': min_val,
            'scale': scale
        }
        
        return quantized, compress_info
    
    def decompress_param(self, quantized, compress_info):
        """解压单个参数张量"""
        # 反量化
        decompressed = (quantized.float() * compress_info['scale'] + 
                       compress_info['min_val'])
        
        # 恢复原始形状
        decompressed = decompressed.view(compress_info['shape'])
        
        return decompressed
    
    def compress_params_dict(self, params_dict):
        """压缩整个参数字典 
        参数:
            params_dict: 参数字典{k:torch.Tensor} / 参数张量的列表[torch.Tensor]

        返回是 
            压缩后的参数列表[torch.Tensor]，以及压缩信息字典
        """
        compressed_list = []
        info_list = []

        if isinstance(params_dict, dict):
            iters = list(params_dict.values())
        else:
            iters = params_dict

        for param in iters:
            quantized, compress_info = self.compress_param(param)
            compressed_list.append(quantized)
            info_list.append(compress_info)
            
        return compressed_list, info_list
    
    def decompress_params_dict(self, compressed_list, info_list):
        """
        根据 解压参数列表，压缩信息字典
        解压整个参数
        
        返回的是 解压后的参数字典[torch.Tensor]
        """
        decompressed_dict = OrderedDict()
        
        for idx, (k, info) in enumerate(zip(self.param_keys, info_list)):
            decompressed_dict[k] = self.decompress_param(compressed_list[idx], info)
            
        return decompressed_dict
    