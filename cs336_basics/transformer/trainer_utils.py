import math
import os
from typing import IO, BinaryIO, Callable, Iterable, Optional
import torch
from torch import nn
import numpy.typing as npt
import numpy as np
from einops import einsum, rearrange

def cross_entropy(logits_i: torch.Tensor, target_i: torch.Tensor) -> torch.Tensor:
    """
    计算单个样本中单个词的交叉熵

    Args:
        logits_i: [1:x_{i}]计算出的未归一化logits向量 (batch_size, ..., vocab_size)
        target_i: 表示logits中第几个是正确答案 (batch_size, ...)

    Returns:
        torch.Tensor: 平均交叉熵
    """
    # 原式：-log{ softmax(logits_i)[target_i] } 
    # 拆开softmax并化简：-logits[target_i] + log(sum(exp(logits_i)))
    
    # 对多维度输入reshape
    logits_i_reshaped = rearrange(logits_i, "b ... v -> (b ...) v")  # (batch_size, vocab_size)
    target_i_reshaped = rearrange(target_i, "b ... -> (b ...)")  # (batch_size,)

    # 对logits预处理，减去每个样本中的最大logit，防止上溢
    logits_i_stable = logits_i_reshaped - logits_i_reshaped.max(dim=-1, keepdim=True).values

    # 计算交叉熵
    targets_logit = logits_i_stable.gather(1, target_i_reshaped.unsqueeze(1)).squeeze(1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_i_stable), dim=-1))
    loss = -targets_logit + log_sum_exp
    # 平均交叉熵
    return loss.mean()

    # log_probs = torch.nn.functional.log_softmax(logits_i, dim=-1)
    # loss = torch.nn.functional.nll_loss(log_probs, target_i)
    # return loss

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
        }
        super().__init__(params, defaults=defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行一次优化步骤
        """
        loss = None if closure is None else closure()
        
        # 对每一个参数组进行梯度下降
        for group in self.param_groups:
            lr = group["lr"] # 同一个参数组有相同的学习率
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param] # 读取之前状态
                t = state.get("t", 0) # 读取当前迭代次数
                grad = param.grad.data # 读取梯度
                param.data -= lr / math.sqrt(t + 1) * grad # 根据梯度和迭代次数更新参数，lr会随着代数增加逐渐衰减
                state["t"] = t + 1 # 迭代次数+1
        
        return loss
    
class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param]
                if len(state) == 0:
                    # 初始化状态
                    state["step"] = 1   # 迭代次数
                    state["exp_avg"] = torch.zeros_like(param.data) # 一阶动量向量（指数加权平均梯度）
                    state["exp_avg_sq"] = torch.zeros_like(param.data) # 二阶动量向量（指数加权平均平方梯度）

                # 读取状态和梯度
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = betas
                grad = param.grad.data

                # 更新一阶和二阶动量
                state["exp_avg"] = beta1 * exp_avg + (1 - beta1) * grad
                state["exp_avg_sq"] = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2

                # 计算偏差矫正后的学习率
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                # 为什么需要偏差矫正？一阶梯度动量的本意是希望用一个更平滑的梯度代替原本容易震荡的梯度
                # 既然是代替梯度，我们希望就 指数加权平均梯度的期望值 == 梯度的期望值，证明两者只是方差变小而期望不变，才算等价代替。
                # 但是，我们一阶梯度动量计算得到的期望和梯度的期望还差一个系数（概率论期望计算可得）
                # 因此需要一个期望修正项。二阶动量也同理。

                # 使用学习率和动量第一次更新参数
                param.data -= lr_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                # 为什么要除sqrt(state["exp_avg_sq"])？这相当于让 步长 正比于 1/sqrt(梯度的二阶原点矩)
                # 如果梯度很大，二阶原点矩也会很大，这样就会减小步长，避免过大步长导致的震荡
                # 如果梯度很小，二阶原点矩也会很小，这样就会增大步长，避免过小步长导致的收敛速度过慢
                # 之所以不用中心矩（方差），是因为我们更关系的是梯度的具体大小，而不是梯度的离散程度，离散我们以及通过指数加权平均进行了平滑
                # 这里的eps是为了防止除0错误，以及sqrt(exp_avg_sq)过小时步长接近inf导致参数上溢

                # 使用权重衰减第二次更新参数
                param.data -= lr * weight_decay * param.data
                # 原本正则化惩罚是为了参数不至于过大，导致模型过拟合，直接加在loss函数中
                # 但是因为梯度计算是基于loss的，所以如果直接加在loss中，梯度会被影响（梯度本只应该包含当前参数值点的导数的信息，因为加正则化项反而还多了参数值本身的信息）
                # 所以AdamW要求去掉loss中的正则化项，而是直接在参数更新时加上权重衰减，这样梯度就不会被影响

                # 更新迭代次数
                state["step"] += 1
        
        return loss

def learning_rate_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    """
    计算余弦退火学习率,使得在训练前中后期动态调整学习率
    （优化器是针对某个参数的大小特调学习率，调度器则是全局迭代进度调整学习率）

    Args:
        it: 当前迭代次数
        max_learning_rate: 最大学习率
        min_learning_rate: 最小学习率
        warmup_iters: 预热结束时迭代次数
        cosine_cycle_iters: 余弦周期结束时迭代次数

    Returns:
        float: 当前学习率
    """
    if it < warmup_iters:
        # 热身阶段，线性增加学习率到最大值
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        # 余弦退火阶段，学习率从最大值逐渐平滑减小到最小值
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * cos_percent)
        )
    else:
        # 余弦周期结束后，保持最小学习率
        return min_learning_rate

def clip_grad(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0, eps: float = 1e-6):
    """
    梯度裁剪，防止梯度爆炸

    Args:
        params: 模型参数列表
        max_norm: 最大梯度范数
    """
    total_norm = 0.0
    # 计算参数梯度的L2范数
    for param in params:
        if param.grad is not None:
            total_norm += torch.sum(param.grad ** 2)
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        # 如果总范数超过最大范数，则进行裁剪
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将一个token序列数据集，通过滑动窗口法，采样batch条长度为context_length的token序列数据
    （每一条数据不一定要在<|endoftext|>截断）

    Args:
        dataset: 输入的token序列数据集，形状为一个一维数组
        batch_size: 每个batch的大小，相当于要采样的样本数量
        context_length: 上下文长度，即batch中每条数据的长度
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 返回一个元组，包含两个Tensor
            - 输入序列Tensor，形状为(batch_size, context_length)
            - 目标序列Tensor，形状为(batch_size, context_length)，每个输入序列的下一个token作为目标
    """
    dataset_len = dataset.shape[0]
    if dataset_len < context_length:
        raise ValueError(f"Dataset length {dataset_len} is less than context length {context_length}.")

    starts = np.random.randint(0, dataset_len - context_length, size=batch_size)
    inputs = np.stack([dataset[start:start + context_length] for start in starts], dtype=np.int64)
    targets = np.stack([dataset[start + 1:start + context_length + 1] for start in starts], dtype=np.int64)

    return (
        # from_numpy会使用numpy数组的内存，不会复制数据，而Tensor会复制
        torch.from_numpy(inputs).to(device),  # 输入序列
        torch.from_numpy(targets).to(device)   # 目标序列
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    """
    将模型参数、优化器状态和迭代次数储存在一个字典中，存入指定的文件或文件对象中。
    """
    return torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    """
    从指定的文件或文件对象中加载模型参数、优化器状态，并返回迭代次数。
    """
    checkpoint = torch.load(src) # 数据会被尝试移动到它们保存时所在的设备
                                 # 如果需要移动到其他指定设备，可以使用map_location参数
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def evaluate_model(model: nn.Module, dataset, device, batch_size, context_length, num_batches=10):
    """
    在验证集上评估模型性能
    
    Args:
        model: 要评估的模型
        dataset: 验证数据集，一维token序列
        device: 计算设备
        batch_size: 批次大小
        context_length: 上下文长度
        num_batches: 要评估的批次数量
        
    Returns:
        float: 验证集上的平均损失
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 不计算梯度以节省内存
        for _ in range(num_batches):
            # 从验证集中获取一批数据
            inputs, targets = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            # 前向传播
            logits = model(inputs)
            # 计算损失
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
    
    model.train()  # 恢复为训练模式
    return total_loss / num_batches  # 返回平均损失

if __name__ == "__main__":
    # 优化器示例
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGDOptimizer([weights], lr=1e1)
    opt = AdamWOptimizer([weights], lr=1e1, weight_decay=0.01)

    for t in range(100):
        opt.zero_grad()                 # 初始化(清空)梯度
        loss = (weights ** 2).mean()    # 计算损失
        print(f"t={t}, loss={loss.cpu().item()}")
        loss.backward()                 # 反向传播计算梯度
        opt.step()                      # 执行一次优化步骤