import torch
import torch.nn as nn
from math import sqrt
from einops import einsum,reduce,rearrange
from jaxtyping import Float

class Linear(nn.Module):
    """
    不包含bias的线性变换层，y=Wx=xW^T
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        初始化
        """
        # 不要忘记调用父类的构造函数
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype)) # 内存中in维度是连续的，有助于加速矩阵乘计算

        # 初始化权重
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Arg:
            x: 输入张量，形状为(batch, ..., in_features)
        """
        return einsum(x, self.W, 'batch ... input, output input -> batch ... output')

class Embedding(nn.Module):
    """
    嵌入层
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        初始化嵌入层
        Args:
            num_embeddings: 嵌入的词汇表大小
            embedding_dim: 每个嵌入的向量空间的维度，也算输出嵌入向量的长度
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 每一行对应一个词的嵌入向量
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # 初始化权重
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, seq_len)，每个元素是词汇表中的索引
        """
        batch_size = x.shape[0]
        return torch.stack([torch.index_select(self.weight, dim=0, index=x[i]) for i in range(batch_size)])

class RMSNorm(nn.Module):
    """
    均方根归一化
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化RMSNorm
        Args:
            d_model: 输入特征的维度
            eps: 防止除零的极小值
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, seq_len, d_model)
        """
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)  # 计算时使用float32类型，中间处理时避免溢出

        # 计算均方根
        rms = torch.sqrt(reduce(x ** 2, "b ... d -> b ... 1",'mean') + self.eps)
        # 归一化,然后乘以可学习系数
        x = x / rms * self.weight

        return x.to(in_dtype) # dtype转化回去

class SWiGLUFeedForward(nn.Module):
    """
    使用 SWiGLU 激活函数的前馈神经网络
    """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        初始化
        Args:
            d_model: 输入特征的维度
            d_ff: SWiGLU的中间层维度，如果为None，则等于8/3 * d_model（舍入到最接近64的值）
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(8 / 3 * d_model)
            self.d_ff = (self.d_ff + 63) // 64 * 64
        else:
            self.d_ff = d_ff
        self.weight1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 W2(SiLU(W1 x) \odot (W3 x))
        Args:
            x: 输入张量，形状为(batch, ..., d_model)
        Returns:
            输出张量，形状为(batch, ..., d_model)
        """
        # 计算SWiGLU
        w1_x = self.weight1(x)
        w3_x = self.weight3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        swiglu = silu * w3_x
        return self.weight2(swiglu)

class RoPE(nn.Module):
    """
    旋转位置编码
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化RoPE
        Args:
            theta: 旋转角度基数
            d_k: 输入Q或K向量的维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算旋转复数矩阵
        self.register_buffer("rope", self._precompute_freqs_cis(), persistent=False)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        """
        预计算频率和相位
        Returns:
            形状为(max_seq_len, d_k)的张量，包含旋转位置编码
        """
        # 计算\theta_i序列，也就是频率序列
        # theta_i = 1 / { theta^{2i / d_k} }
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[:(self.d_k // 2)] / self.d_k))
        # 生成序列索引m [0, 1, ..., max_seq_len-1]
        seq_idx = torch.arange(0, self.max_seq_len, device=self.device)
        # 计算 m * \theta_i 矩阵
        freqs = einsum(seq_idx, freqs, "seq, d -> seq d")

        # 复数化
        # freqs[m][i] = m * \theta_i
        # freqs_cis[m][i] = 1 * e^{i * m * \theta_i}
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(..., seq_len, d_k)
            token_positions: 位置索引，形状为(..., seq_len)
        Returns:
            旋转位置编码后的张量，形状为(..., seq_len, d_k)
        """
        # 将维度分组
        x_ = rearrange(x, "... seq (d two) -> ... seq d two", two=2).float()
        # 转为复数(... seq (d 2) )
        x_ = torch.view_as_complex(x_)

        # 根据token_positions获取对应的位置的频率
        rope_pos = self.rope[token_positions]  # (batch, ..., seq_len, d_k // 2)

        # 旋转，之后转回实数域并展平
        x_out = rearrange(torch.view_as_real(x_ * rope_pos), "... seq d two -> ... seq (d two)", two=2)
        
        return x_out.to(x.dtype)  # 转回原始dtype

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    计算softmax
    Args:
        x: 输入张量
        dim: 计算softmax的维度
    Returns:
        softmax后的张量
    """
    # 减去最大值以防止溢出(对所有元素加上常数c不改变softmax输出)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    计算缩放点积注意力
    Args:
        Q: 查询张量，形状为(batch, ..., seq_len_q, d_k)
        K: 键张量，形状为(batch, ..., seq_len_k, d_k)
        V: 值张量，形状为(batch, ..., seq_len_k, d_v)
        mask: 可选的掩码张量，形状为(seq_len_q, seq_len_k)
    Returns:
        注意力输出张量，形状为(batch, ..., seq_len_q, d_v)
    """
    # 计算注意力分数 QK^T
    scores = einsum(Q, K, 'batch ... q d_k, batch ... k d_k -> batch ... q k')
    d_k = Q.shape[-1]
    scores /= sqrt(d_k) # 缩放 QK^T / sqrt(d_k)

    # 应用掩码,在False上 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)

    # 计算输出 
    output = einsum(attn_weights, V, 'batch ... q k, batch ... k d_v -> batch ... q d_v')
    
    return output

class MultiheadSelfAttention(nn.Module):
    """
    多头自注意力
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # d_v = d_k
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.device = device
        self.dtype = dtype

        # self.w_q = Linear(d_model, self.num_heads*self.d_k, device=device, dtype=dtype)
        # self.w_k = Linear(d_model, self.num_heads*self.d_k, device=device, dtype=dtype)
        # self.w_v = Linear(d_model, self.num_heads*self.d_k, device=device, dtype=dtype)
        # 上面三个矩阵都要和x相乘，又都有一维完全相同，可否考虑合成为一个？
        self.w_qkv = Linear(d_model, self.num_heads * self.d_k * 3, device=device, dtype=dtype)
        self.w_o = Linear(self.num_heads * self.d_k, self.d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        QKV = self.w_qkv(x)  # (batch, ..., seq_len, head * d_k * 3)
        # 分割Q、K、V
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)

        # 因果掩码：(seq_len_q, seq_len_k)
        # 位置i的query不能分配注意力给位置j的key（j>i）
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)  # (batch, ..., head, seq_len, d_k)
        
        # 将多头拼接回去
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        
        return self.w_o(atten)  # (batch, ..., seq_len, d_model)

class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    """
    带有旋转位置编码的多头自注意力
    """
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
            token_positions: 位置索引，形状为(batch, ..., seq_len)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        QKV = self.w_qkv(x)  # (batch, ..., seq_len, head * d_k * 3)
        # 分割Q、K、V
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)

        # 对Q，K使用RoPE，head视为batch维度
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # 因果掩码：(seq_len_q, seq_len_k)
        # 位置i的query不能分配注意力给位置j的key（j>i）
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)  # (batch, ..., head, seq_len, d_k)
        
        # 将多头拼接回去
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        
        return self.w_o(atten)  # (batch, ..., seq_len, d_model)

class TransformerBlock(nn.Module):
    """
    Transformer块，包含多头自注意力和前馈网络
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int = None, theta: float=10000.0, device=None):
        """
        初始化Transformer块
        Args:
            d_model: 输入特征的维度
            num_heads: 多头注意力的头数
            d_ff: 前馈网络的中间层维度
            max_seq_len: 最大序列长度，位置嵌入用
            theta: 旋转位置编码的基数
            
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attention = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device)
        self.ffn = SWiGLUFeedForward(d_model, d_ff, device=device)
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
            token_positions: 位置索引，形状为(batch, ..., seq_len)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        token_positions = torch.arange(x.shape[-2], dtype=torch.int, device=x.device)  # (batch, ..., seq_len)

        # 多头自注意力
        attn_output = self.attention(
            self.norm1(x), token_positions
        )
        x2 = x + attn_output
        # 前馈网络
        ffn_output = self.ffn(self.norm2(x2))
        return x2 + ffn_output

class TransformerLM(nn.Module):
    """
    Transformer语言模型
    """
    def __init__(self, 
                 vocab_size: int, 
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int = None,
                 rope_theta: float = 10000.0,
                 device=None,
                 dtype=None):
        """
        初始化Transformer语言模型
        Args:
            vocab_size: 词汇表大小
            context_length: 上下文长度，也就是最大序列长度
            d_model: 输入特征的维度
            num_heads: 多头注意力的头数
            num_layers: Transformer块的层数
            d_ff: 前馈网络的中间层维度
            rope_theta: 旋转位置编码的基数
        """
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device)
        self.tf_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, d_ff, rope_theta, device=device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.output_embedding = Linear(d_model, vocab_size, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, seq_len)，每个元素是词汇表中的索引
        Returns:
            输出张量，形状为(batch, seq_len, vocab_size)
        """
        # 嵌入
        x = self.token_embedding(x)
        # Transformer块
        for block in self.tf_blocks:
            x = block(x)
        # 最终归一化
        x = self.ln_final(x)
        # 输出层
        x = self.output_embedding(x)
        # # softmax，暂时先不用
        # return softmax(x, dim=-1)
        return x
