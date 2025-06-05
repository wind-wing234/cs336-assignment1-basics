import torch
from loguru import logger
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer
from cs336_basics.transformer.module import TransformerLM, softmax
from cs336_basics.transformer.trainer_utils import load_checkpoint

def nucleus_sampling(probs, top_p):
    """
    实现Nucleus sampling（也称为Top-p采样）
    将token概率从高到低排列，逐一累加，直到累积概率超过top_p为止，后面的token概率全部设为0。
    
    Args:
        probs: 形状为 (batch_size, vocab_size) 的概率分布
        top_p: 浮点数，表示累积概率的阈值（0.0 < top_p <= 1.0）
        
    Return:
        修正后的概率分布，形状与输入相同
    """
    # 对每个样本的每个token的概率进行排序（降序）
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 计算累积概率序列
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找出累积概率小于等于top_p的索引，将其掩码标为True
    nucleus_mask = cumulative_probs <= top_p
    
    # 如果概率最高的token就大于top-p，会导致全为False
    # 这里要确保至少保留概率最高的token
    nucleus_mask[:, 0] = True
    
    # 对于掩码为False的位置（概率较低的token），将其概率设为0
    sorted_probs_fixed = sorted_probs * nucleus_mask
    
    # 重新归一化概率分布
    sorted_probs_sum = torch.sum(sorted_probs_fixed, dim=-1, keepdim=True)
    sorted_probs_normalized = sorted_probs_fixed / sorted_probs_sum
    
    # 将排序后的概率分布映射回原始顺序
    probs_filtered = torch.zeros_like(probs)
    
    # 使用scatter操作还原原始顺序
    probs_filtered.scatter_(1, sorted_indices, sorted_probs_normalized)
    
    return probs_filtered



if __name__ == "__main__":

    prompts = ["The quick brown fox jumps over the lazy dog",
               "Once upon a time,",
               "Tom and Lily are best friends.",]
    
    max_new_tokens = 128
    temperature = 1.2
    top_p = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    context_length = 256

    end_token = "<|endoftext|>"

    # 初始化分词器
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./data/token/TinyStories_train_10000_token_vocab.bin",
        mergers_filepath="./data/token/TinyStories_train_10000_merges.bin",
        special_tokens=["<|endoftext|>"]
    )

    # 初始化模型
    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10000,
        device=device
    )

    # 加载模型参数
    load_checkpoint(
        src="./data/model/checkpoint_v0_16000.pt",
        model=model,
        optimizer=None  
    )

    # 将模型移动到设备
    model.to(device)
    # 设置模型为评估模式
    model.eval()

    # 对输入进行分词
    inputs_ids = []
    len_inputs_ids = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).to(device)
        inputs_ids.append(input_ids)
        len_inputs_ids.append(len(input_ids))
    
    len_inputs_ids = torch.tensor(len_inputs_ids, dtype=torch.int64, device=device)  # (batch_size,)
    
    # 如果batch中的序列长度不一致，需要进行padding填充
    # 否则无法作为一个矩阵输入到模型中
    # 这里先将所有输入左对齐，然后在右侧使用0填充
    pad_token_id = tokenizer.encode(end_token)[0]
    padded_inputs = torch.full(
        (len(inputs_ids), context_length), 
        fill_value=pad_token_id,
        dtype=torch.int32,
        device=device
    )
    for i, input_id in enumerate(inputs_ids):
        padded_inputs[i, :len(input_id)] = input_id


    end_token_id = tokenizer.encode(end_token)[0]
    is_end = torch.zeros(padded_inputs.shape[0], dtype=torch.bool, device=device)  # 记录每个序列是否已经结束

    # 生成阶段
    with torch.no_grad():
        for num in range(max_new_tokens):
            # (batch, max_seq_len) -> (batch_size, max_seq_len, vocab_size)
            logits = model(padded_inputs)
            index = len_inputs_ids - 1 + num
            index = repeat(index, 'b -> b 1 v', v=logits.shape[-1])  # (batch_size, 1, vocab_size)
            # 取出input_ids最后一个token的logits，这才是预测的token
            logits = torch.gather(logits, dim=1, index=index).squeeze(1) # (batch_size, vocab_size)

            # temperature 越大，logits更分布在数轴两端，输出越随机；
            # temperature 越小，logits分布都被压缩到0附近，输出越确定
            logits = logits / temperature
            # 计算softmax
            probs = softmax(logits, dim=-1)

            # 使用top-p去除末尾概率
            probs = nucleus_sampling(probs, top_p)
            
            # 从概率分布中采样
            next_token_ids = torch.multinomial(probs, num_samples=1).to(dtype=torch.int32) # (batch_size, 1) 

            # 将采样的token添加到输出中
            next_token_index = (len_inputs_ids + num).unsqueeze(1) # (batch_size, 1)
            padded_inputs.scatter_(1, next_token_index, next_token_ids)

            # 更新is_end标志
            is_end = is_end | (next_token_ids == end_token_id)
            # 如果所有序列都已经结束，则提前退出
            if is_end.all():
                break
    
    # 解码输出序列
    outputs = []
    for i in range(padded_inputs.shape[0]):
        output_ids = padded_inputs[i, len_inputs_ids[i]:].cpu().numpy()
        output_text = tokenizer.decode(output_ids, end_token_id=end_token_id)
        outputs.append(output_text)
    
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output}")
            
            
