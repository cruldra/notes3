import math
import os
import random
import time
from contextlib import nullcontext
from typing import Tuple, Optional, List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
from datasets import load_dataset
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler, Sampler, DataLoader
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoTokenizer,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,  # Dropout概率，用于防止过拟合
            bos_token_id: int = 1,  # 句子开始token的ID (Begin of Sentence)
            eos_token_id: int = 2,  # 句子结束token的ID (End of Sentence)
            hidden_act: str = "silu",  # 隐藏层激活函数，如 'silu', 'gelu' 等
            hidden_size: int = 512,  # 模型的隐藏层维度 (Embedding维度)
            intermediate_size: int = None,  # FFN中间层的维度，如果为None则默认为 hidden_size * 8/3
            max_position_embeddings: int = 32768,  # 模型支持的最大序列长度
            num_attention_heads: int = 8,  # 注意力头的数量 (Query heads)
            num_hidden_layers: int = 8,  # Transformer解码器层的数量
            num_key_value_heads: int = 2,  # Key/Value头的数量 (用于GQA)，如果为None则等于num_attention_heads
            vocab_size: int = 6400,  # 词表大小
            rms_norm_eps: float = 1e-05,  # RMSNorm层的epsilon值，防止除零
            rope_theta: int = 1000000.0,  # RoPE旋转位置编码的基数 (Theta)
            inference_rope_scaling: bool = False,  # 推理时是否使用RoPE插值扩展上下文
            flash_attn: bool = True,  # 是否启用Flash Attention加速
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,  # 是否启用混合专家模型 (MoE)
            num_experts_per_tok: int = 2,  # 每个Token选择的专家数量 (Top-K)
            n_routed_experts: int = 4,  # 总的路由专家数量
            n_shared_experts: int = 1,  # 共享专家数量 (始终激活)
            scoring_func: str = "softmax",  # 门控评分函数，通常为 'softmax'
            aux_loss_alpha: float = 0.01,  # 辅助损失系数，用于平衡专家负载
            seq_aux: bool = True,  # 是否在序列级别计算辅助损失
            norm_topk_prob: bool = True,  # 是否对Top-K专家的概率进行归一化
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(
        lm_config,
        weight="full_sft",
        model=None,
        optimizer=None,
        epoch=0,
        step=0,
        wandb=None,
        save_dir="../checkpoints",
        **kwargs,
):
    """
    模型检查点保存与加载函数。

    Args:
        lm_config: 模型配置对象，用于获取 hidden_size 和是否使用 MoE。
        weight: 权重文件名标识，默认为 'full_sft'。
        model: 模型实例。如果不为 None，则执行保存操作；为 None 则执行加载操作。
        optimizer: 优化器实例，仅在保存时需要。
        epoch: 当前训练轮数。
        step: 当前训练步数。
        wandb: Weights & Biases 实例，用于记录 run id 以便断点续传。
        save_dir: 检查点保存目录。
        **kwargs: 其他需要保存的对象（如 scheduler），如果对象有 state_dict 方法会自动调用。

    Returns:
        加载模式下返回包含恢复信息的字典 (dict)，否则返回 None。
    """

    # 1. 准备保存目录和路径
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    moe_path = "_moe" if lm_config.use_moe else ""  # MoE 模型添加特殊后缀
    # 纯权重文件路径 (仅包含 model state_dict，体积小，用于推理)
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    # 恢复文件路径 (包含 model, optimizer, step, epoch 等，用于断点续训)
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        # ==================== 保存模式 ====================

        # 2. 解包模型 (Unwrap)
        # 如果是 DDP (分布式) 模型，取其 .module
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        # 如果是 torch.compile 编译后的模型，取其 _orig_mod
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

        # 3. 处理模型权重 (Save Model Weights)
        state_dict = raw_model.state_dict()
        # 将权重转为半精度 (FP16) 并移动到 CPU，节省存储空间和显存
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 4. 原子保存权重文件 (Atomic Save)
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)  # 先写入临时文件
        os.replace(ckp_tmp, ckp_path)  # 原子替换，防止写入中断导致文件损坏

        # 5. 获取 WandB Run ID (用于恢复曲线)
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        # 6. 构建恢复数据字典 (Resume Data)
        resume_data = {
            "model": state_dict,  # 模型权重
            "optimizer": optimizer.state_dict(),  # 优化器状态
            "epoch": epoch,  # 当前 Epoch
            "step": step,  # 当前 Step
            "world_size": dist.get_world_size()
            if dist.is_initialized()
            else 1,  # 保存时的 GPU 数量
            "wandb_id": wandb_id,  # WandB ID
        }

        # 7. 处理额外的 kwargs (如 LR Scheduler)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    # 如果是 DDP 或编译后的对象，同样需要解包
                    raw_value = (
                        value.module
                        if isinstance(value, DistributedDataParallel)
                        else value
                    )
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 8. 原子保存恢复文件
        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 9. 清理资源
        del state_dict, resume_data
        torch.cuda.empty_cache()  # 释放显存

    else:
        # ==================== 加载模式 ====================

        if os.path.exists(resume_path):
            # 1. 加载恢复文件到 CPU
            ckp_data = torch.load(resume_path, map_location="cpu")

            # 2. 处理 GPU 数量变化带来的 Step 差异
            # 场景：例如 4 卡变 8 卡，Global Batch Size 翻倍，总 Step 数应减半
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                print(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None


class FeedForward(nn.Module):
    """
    前馈神经网络层 (Feed-Forward Network, FFN)。
    这里采用的是 Llama 架构中经典的 SwiGLU (Swish-Gated Linear Unit) 变体。
    相比传统的 ReLU FFN (up_proj -> relu -> down_proj)，SwiGLU 增加了门控机制，性能通常更好。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果未指定中间层维度 (intermediate_size)，则自动计算。
        # 也就是 FFN 内部升维后的维度。
        if config.intermediate_size is None:
            # 通常设置为 hidden_size 的 8/3 倍 (约为 2.67 倍)，这是 Llama 的惯用比例。
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 为了硬件计算效率（如 GPU Tensor Core），将维度对齐到 64 的倍数。
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # gate_proj: 门控投影层。输入 -> 中间维度。用于计算激活值（门）。
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        # down_proj: 下行投影层。中间维度 -> 输出维度。用于将数据映射回原来的 hidden_size。
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        # up_proj: 上行投影层。输入 -> 中间维度。用于计算待被“门”控制的特征值。
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

        self.dropout = nn.Dropout(config.dropout)
        # 获取激活函数，通常是 'silu' (SiLU / Swish)。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前向传播逻辑：
        1. gate_proj(x): 算出门控信号。
        2. act_fn(...): 对门控信号进行非线性激活 (SiLU)。
        3. up_proj(x): 算出原始特征信号。
        4. (...) * up_proj(x): 逐元素相乘 (Element-wise multiplication)。
           这是 GLU (Gated Linear Unit) 的核心，用激活后的门去“控制”特征信号的通过量。
        5. down_proj(...): 将加权后的特征映射回原始维度。
        6. dropout(...): 防止过拟合。

        公式: Output = Dropout( DownProj( Act(GateProj(x)) * UpProj(x) ) )
        """
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码 (Rotary Positional Embeddings, RoPE)。
    RoPE 通过将 Query 和 Key 向量在复数域空间中进行旋转操作来注入绝对位置信息，
    这种方式能够自然地让模型捕捉到 token 之间的相对位置关系。

    Args:
        q: Query 向量，形状通常为 [batch_size, seq_len, num_heads, head_dim]
        k: Key 向量，形状通常为 [batch_size, seq_len, num_kv_heads, head_dim]
        cos: 预计算的 Cosine 值，形状通常为 [seq_len, head_dim]
        sin: 预计算的 Sine 值，形状通常为 [seq_len, head_dim]
        position_ids: (在此实现中未使用，假设 cos/sin 已根据 position_ids 提取好)
        unsqueeze_dim: 用于广播 cos/sin 的维度。因为 cos/sin 通常没有 head 维度，
                       需要 unsqueeze 插入一个维度以便与 q, k 进行广播运算。

    Returns:
        q_embed: 旋转后的 Query 向量
        k_embed: 旋转后的 Key 向量
    """

    # 辅助函数：执行“半旋转”操作
    # 对应 RoPE 数学推导中的 (-x2, x1) 变换部分
    def rotate_half(x):
        # 将向量 x 沿最后一个维度 (head_dim) 切分为前半部分 x1 和后半部分 x2
        # 例如 x = [x1, x2]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        # 返回 [-x2, x1]
        return torch.cat((-x2, x1), dim=-1)

    # 应用 RoPE 旋转公式：
    # f(x, pos) = x * cos(pos) + rotate_half(x) * sin(pos)
    #
    # 原理推导 (以 head_dim 中的一对特征 (x1, x2) 为例，旋转角度为 theta):
    # 目标是计算复数乘法: (x1 + i*x2) * e^(i*theta) = (x1 + i*x2) * (cos + i*sin)
    # 展开实部: x1*cos - x2*sin
    # 展开虚部: x2*cos + x1*sin
    #
    # 代码实现对应:
    # 第一部分 (x * cos): [x1*cos, x2*cos]
    # 第二部分 (rotate_half(x) * sin): [-x2*sin, x1*sin]
    # 相加: [x1*cos - x2*sin, x2*cos + x1*sin] -> 正好匹配旋转矩阵的结果

    # unsqueeze(unsqueeze_dim) 是为了在 num_heads 维度上广播
    # cos shape: [seq_len, head_dim] -> [seq_len, 1, head_dim] -> 广播适配 q: [batch, seq_len, n_heads, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    为了支持分组查询注意力 (Grouped Query Attention, GQA) 机制而设计的函数。
    在 GQA 中，Key (K) 和 Value (V) 的头数 (num_kv_heads) 通常少于 Query (Q) 的头数 (num_heads)。
    为了进行点积计算，我们需要将 K 和 V 在头的维度上进行“复制”扩展，使其头数与 Q 对齐。

    torch.repeat_interleave(x, dim=2, repeats=n_rep) 的手动高效实现。

    Args:
        x: 输入张量，通常是 Key 或 Value。
           Shape: [batch_size, seq_len, num_kv_heads, head_dim]
        n_rep: 复制的倍数 (repeats)。
               n_rep = num_heads // num_kv_heads

    Returns:
        扩展后的张量。
        Shape: [batch_size, seq_len, num_heads, head_dim]
               其中 num_heads = num_kv_heads * n_rep
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # 操作流程：
    # 1. x[:, :, :, None, :] -> 在 num_kv_heads 后插入一个新的维度。
    #    Shape: [bs, slen, num_kv_heads, 1, head_dim]
    # 2. .expand(..., n_rep, ...) -> 在新维度上进行广播复制。
    #    Shape: [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. .reshape(...) -> 将最后两个维度合并，完成“交织”复制。
    #    Shape: [bs, slen, num_kv_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    注意力机制模块 (Attention Mechanism)。
    本实现支持分组查询注意力 (Grouped Query Attention, GQA) 以及 Flash Attention 加速。
    同时集成了旋转位置编码 (RoPE) 和 KV Cache 机制。
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # num_key_value_heads: KV heads 的数量。
        # 如果配置中未指定 (None)，则默认为 num_attention_heads (即标准 Multi-Head Attention, MHA)。
        # 如果指定了且小于 num_attention_heads，则是 GQA。
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        # 检查 num_attention_heads 是否能被 num_key_value_heads 整除
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads  # Query 头数
        self.n_local_kv_heads = self.num_key_value_heads  # Key/Value 头数
        # n_rep: 每个 KV head 对应多少个 Q head (GQA 的重复倍数)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 线性投影层
        # q_proj: [hidden_size] -> [num_heads * head_dim]
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        # k_proj: [hidden_size] -> [num_kv_heads * head_dim] (注意维度可能小于 Q)
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        # v_proj: [hidden_size] -> [num_kv_heads * head_dim]
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        # o_proj: [num_heads * head_dim] -> [hidden_size] (输出投影)
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否支持 Flash Attention (PyTorch 2.0+ 特性)
        self.flash = (
                hasattr(torch.nn.functional, "scaled_dot_product_attention")
                and args.flash_attn
        )
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[
                torch.Tensor, torch.Tensor
            ],  # 接收预计算好的 cos, sin
            past_key_value: Optional[
                Tuple[torch.Tensor, torch.Tensor]
            ] = None,  # 历史 KV 缓存
            use_cache=False,  # 是否使用 KV Cache
            attention_mask: Optional[
                torch.Tensor
            ] = None,  # 外部传入的 mask (如 padding mask)
    ):
        bsz, seq_len, _ = x.shape

        # 1. 线性投影 (Projection)
        # xq: [batch, seq_len, n_heads * head_dim]
        # xk, xv: [batch, seq_len, n_kv_heads * head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. 重塑形状 (Reshape)
        # 将 flat 的 embedding 维度切分为 heads 和 head_dim
        # xq: [batch, seq_len, n_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 应用旋转位置编码 (RoPE)
        # 将位置信息注入到 Query 和 Key 中
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 4. KV Cache 处理 (推理优化)
        # 如果提供了过去的 KV，将其与当前的 KV 拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 如果启用缓存，保存当前的完整 KV 状态
        past_kv = (xk, xv) if use_cache else None

        # 5. 准备注意力计算 (GQA 扩展 & 维度转置)
        # GQA 关键步: 如果 KV heads 少于 Q heads，需要重复 KV 以匹配 Q 的数量
        # repeat_kv: [bs, seq, n_kv, dim] -> [bs, seq, n_heads, dim]
        # transpose: [bs, seq, n_heads, dim] -> [bs, n_heads, seq, dim] (符合 Attention 计算格式)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 6. 计算注意力 (Attention Calculation)
        if (
                self.flash
                and (seq_len > 1)
                and (past_key_value is None)
                and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # 6a. Flash Attention (快速路径)
            # PyTorch 内置的高效实现，自动处理 Mask 和 Softmax，大幅减少显存占用和计算时间
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # 自动应用因果掩码 (即左下三角 mask)
            )
        else:
            # 6b. 手动实现 Attention (慢速/兼容路径)

            # 计算 Attention Scores: Q @ K^T / sqrt(d_k)
            # scores shape: [bs, n_heads, seq_len_q, seq_len_k]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 应用因果掩码 (Causal Mask)
            # 确保当前位置只能看到自己及之前的位置，看不到未来的 token
            # mask 为上三角矩阵 (对角线以上)，填充 -inf
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            # 应用外部传入的 Attention Mask (如 Padding Mask)
            if attention_mask is not None:
                # 扩展 mask 维度以匹配 scores: [bs, 1, 1, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 将 0 (mask) 的位置变为极小的负数，Softmax 后趋近于 0
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax 归一化得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            # 加权求和: Scores @ V
            # output: [bs, n_heads, seq_len, head_dim]
            output = scores @ xv

        # 7. 输出投影 (Output Projection)
        # 恢复形状: [bs, n_heads, seq, dim] -> [bs, seq, n_heads, dim] -> [bs, seq, hidden]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class RMSNorm(torch.nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)。

    RMSNorm 是 LayerNorm 的一种变体，由 Zhang et al. (2019) 提出。
    相比于标准的 LayerNorm，RMSNorm 省略了减去均值 (Mean Centering) 的步骤，
    仅通过均方根 (RMS) 进行缩放。这减少了计算量，同时在许多大模型 (如 Llama) 中
    表现出相当甚至更好的性能。

    公式:
        RMS(x) = sqrt( mean(x^2) + eps )
        Output = x / RMS(x) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: 输入特征的维度 (hidden_size)。
            eps: 一个微小的数，用于防止除零错误 (epsilon)。
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 (Gain/Scale)，形状为 [dim]。
        # 对应 LayerNorm 中的 gamma，但 RMSNorm 没有偏置项 (beta)。
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        计算 RMSNorm 的核心标准化步骤。

        Args:
            x: 输入张量。

        Returns:
            标准化后的张量 (在乘以 weight 之前)。
        """
        # 1. x.pow(2): 计算元素的平方。
        # 2. .mean(-1, keepdim=True): 沿最后一个维度计算均值 (Mean Square)。
        # 3. + self.eps: 加上 epsilon 防止开方时出现 0。
        # 4. torch.rsqrt(...): 计算平方根的倒数 (Reciprocal Square Root)，即 1 / sqrt(...)。
        # 5. x * ...: 将输入 x 乘以这个缩放因子。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播。

        注意：为了数值稳定性，RMS 的计算通常在 float32 (FP32) 精度下进行。
        """
        # 1. x.float(): 将输入转为 FP32 精度。RMSNorm 对精度敏感，FP16/BF16 可能导致溢出或精度损失。
        # 2. self._norm(...): 计算标准化结果。
        # 3. .type_as(x): 将结果转回输入 x 的原始数据类型 (如 FP16/BF16)。
        # 4. self.weight * ...: 乘以可学习的缩放参数。
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
        dim: int,
        end: int = int(32 * 1024),
        rope_base: float = 1e6,
        rope_scaling: Optional[dict] = None,
):
    """
    预计算旋转位置编码 (RoPE) 的复数频率 (Cis) 值，即 cos 和 sin。
    该函数支持 YaRN (Yet another RoPE extension method) 长上下文外推算法。

    Args:
        dim: 每个注意力头的维度 (head_dim)。注意 RoPE 通常应用于所有维度。
        end: 预计算的最大序列长度 (max_position_embeddings)。
        rope_base: RoPE 的基数 (base)，通常为 10000.0 或 1000000.0。
                   较大的 base 有助于捕捉更长距离的依赖。
        rope_scaling: 包含 YaRN 等扩展配置的字典。如果为 None，则使用标准 RoPE。

    Returns:
        freqs_cos: 预计算的 Cosine 值，形状为 [end, dim]。
        freqs_sin: 预计算的 Sine 值，形状为 [end, dim]。
    """
    # 1. 计算基础频率 (Theta)
    # formula: theta_i = 1 / (base ^ (2i / dim)) for i in [0, dim/2)
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    # 2. 应用 YaRN (Yet another RoPE extension method) 缩放逻辑
    # YaRN 是一种无需微调即可扩展 LLM 上下文窗口的技术。
    if rope_scaling is not None:
        # 解包配置参数
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)  # 扩展倍数
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 仅当需要的长度超过原始训练长度时才应用缩放
        if end / orig_max > 1.0:
            # YaRN 插值策略：混合了直接插值 (Linear) 和外推
            # 对于高频部分 (beta_fast)，不做处理；对于低频部分 (beta_slow)，进行插值。
            # 中间部分使用线性斜坡 (Linear Ramp) 进行平滑过渡。

            # 计算频率对应的波长，并反推出维度索引边界
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                        2 * math.log(rope_base)
                )

            # 确定混合区间的上下界
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 生成斜坡函数 (Ramp Function)，值域在 [0, 1] 之间
            # 0 表示完全不插值，1 表示完全插值 (除以 factor)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 更新频率：freqs = freqs * (1 - ramp + ramp / factor)
            # 这有效地“拉伸”了低频部分的波长，使其能覆盖更长的上下文
            freqs = freqs * (1 - ramp + ramp / factor)

    # 3. 生成所有位置的频率矩阵 (Outer Product)
    # t: 位置索引序列 [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # freqs (theta): [dim/2]
    # result: [end, dim/2] -> 每一行是一个位置的所有频率 theta * t
    freqs = torch.outer(t, freqs).float()

    # 4. 构建 Cos 和 Sin 矩阵
    # 为了配合 apply_rotary_pos_emb 中的 rotate_half 实现 (x1, x2)，
    # 我们需要将频率复制一遍，使得 dim/2 变成 dim。
    # 假设输入向量是 [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
    # RoPE 旋转的是 (x_i, x_{i+d/2})这一对。
    # 所以 cos/sin 矩阵的左半部分和右半部分应该是相同的。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


class MoEGate(nn.Module):
    """
    MoE (Mixture of Experts) 模型的门控机制 (Gating / Router)。

    作用：
    对于每个输入 Token，计算它应该被路由到哪几个专家 (Expert)，以及分配给这些专家的权重。
    同时，在训练过程中计算辅助损失 (Auxiliary Loss / Load Balancing Loss)，
    以防止某些专家被过度使用而其他专家闲置（即“专家坍塌”问题）。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 Token 选择的专家数量 (Top-K)
        self.n_routed_experts = config.n_routed_experts  # 总的专家数量

        self.scoring_func = config.scoring_func  # 评分函数，通常为 'softmax'
        self.alpha = config.aux_loss_alpha  # 辅助损失的权重系数
        self.seq_aux = config.seq_aux  # 是否在序列级别 (Sequence-level) 计算辅助损失

        self.norm_topk_prob = (
            config.norm_topk_prob
        )  # 是否对选中的 Top-K 专家的权重进行归一化
        self.gating_dim = config.hidden_size  # 输入维度
        # 门控网络的权重矩阵: [n_experts, hidden_size]
        # 用于将输入特征映射到专家分数的空间
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Kaiming Uniform 初始化权重，有助于模型收敛
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播逻辑。

        Args:
            hidden_states: 输入的隐藏状态张量 [batch_size, seq_len, hidden_size]

        Returns:
            topk_idx: 选中的专家索引 [batch_size * seq_len, top_k]
            topk_weight: 选中的专家对应的权重 [batch_size * seq_len, top_k]
            aux_loss: 辅助损失标量 (用于负载均衡)
        """
        bsz, seq_len, h = hidden_states.shape
        # 1. 展平输入: [batch_size * seq_len, hidden_size]
        # 因为门控是针对每个 Token 独立进行的
        hidden_states = hidden_states.view(-1, h)

        # 2. 计算路由分数 (Routing Scores)
        # logits: [num_tokens, n_experts]
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            # 使用 Softmax 将 logits 转换为概率分布 (0~1 之间，和为 1)
            # scores: [num_tokens, n_experts]
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # 3. 选择 Top-K 专家
        # topk_weight: [num_tokens, top_k] (选中的专家的原始概率)
        # topk_idx: [num_tokens, top_k] (选中的专家的索引)
        # sorted=False 表示不需要对结果排序，通常能稍微快一点
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 4. 权重归一化 (可选)
        # 如果选中了多个专家，通常将它们的权重重新归一化，使其和为 1。
        # 这样可以保持后续加权求和时的数值稳定性。
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 5. 计算辅助损失 (Auxiliary Loss / Load Balancing Loss)
        # 仅在训练阶段且 alpha > 0 时计算
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 展平索引用于统计
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # ==================== 序列级辅助损失 ====================
                # 这种计算方式在 DeepSeek-MoE 等论文中较为常见。
                # 它可以更细粒度地在每个序列内部保证负载均衡。

                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # ce (Count of Experts): 记录每个专家被选中的次数 (实际负载)
                # shape: [batch_size, n_experts]
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )

                # 使用 scatter_add_ 将选中的专家计数加到 ce 中
                # dim=1 表示在专家维度上操作
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)

                # aux_loss = sum(实际负载 * 预期负载) * alpha
                # scores_for_seq_aux.mean(dim=1): 计算每个专家在序列上的平均概率 (预期负载)
                # 这种损失函数鼓励 "实际选中频率" 与 "模型预测概率" 分布一致，且趋向均匀。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                # ==================== 全局/Batch级辅助损失 ====================
                # 经典的 Switch Transformer 负载均衡损失

                # 1. 计算每个专家被选中的概率 (实际负载 fraction)
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)  # shape: [n_experts]

                # 2. 计算每个专家的平均路由概率 (预期负载 fraction)
                Pi = scores_for_aux.mean(0)  # shape: [n_experts]

                # 3. 归一化因子
                fi = ce * self.n_routed_experts

                # 4. 计算点积损失
                # 目标是最小化 sum(Pi * fi)。
                # 当所有专家的 Pi 和 fi 都接近 1/n_experts (均匀分布) 时，该乘积最小。
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络 (Mixture of Experts Feed-Forward Network)。

    该模块实现了 MoE 的核心逻辑，包括：
    1. 路由专家 (Routed Experts): 根据门控网络的输出，动态选择部分专家进行计算。
    2. 共享专家 (Shared Experts): (可选) 无论输入如何，都会被激活的专家。这是 DeepSeek-MoE 等架构引入的设计，用于捕捉通用知识。
    3. 门控网络 (Gate): 决定输入 Token 应该去往哪个专家。

    架构:
    Input -> Gate -> TopK Experts -> Weighted Sum -> Output
             +-> Shared Experts -> Add -------------^
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 初始化路由专家列表 (Routed Experts)
        # 每个专家都是一个独立的 FeedForward 网络
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 初始化门控网络 (Router)
        self.gate = MoEGate(config)
        # 初始化共享专家 (Shared Experts)
        # 如果配置了 n_shared_experts > 0，这些专家会对所有 Token 处于激活状态
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        """
        前向传播函数。
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 1. 门控路由 (Gating & Routing)
        # topk_idx: 每个 Token 选中的专家索引 [batch*seq, top_k]
        # topk_weight: 对应的路由权重 [batch*seq, top_k]
        # aux_loss: 辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 展平输入以方便处理: [batch*seq, hidden_size]
        x = x.view(-1, x.shape[-1])
        # 展平索引: [batch*seq*top_k]
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # ==================== 训练模式 ====================
            # 在训练时，我们通常遍历所有专家，选出属于该专家的 Token 进行计算。
            # 这种方式虽然看起来有循环，但在 experts 数量不多时效率尚可，且兼容性好（支持 Autograd）。

            # 复制输入: 因为每个 Token 被 Top-K 个专家处理，所以输入需要重复 K 次
            # x: [batch*seq, hidden] -> [batch*seq*top_k, hidden] (逻辑上的视角，实际通过掩码实现)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            # 遍历每一个专家
            for i, expert in enumerate(self.experts):
                # 找出分配给当前专家 i 的 Token
                # flat_topk_idx == i 生成一个布尔掩码
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])

                # 显式处理空分支，确保梯度流连通性 (即使某专家未被选中，也需要参与计算图构建，避免 DDP 报错)
                # + 0 * sum(...) 是一个常见的 PyTorch trick，用于挂载梯度但不影响数值
                if y[flat_topk_idx == i].shape[0] == 0:
                    y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )

            # 加权求和 (Weighted Sum)
            # y: [batch*seq*top_k, hidden] -> 调整形状与 topk_weight 对齐
            # topk_weight: [batch*seq, top_k] -> 扩展维度 -> [batch*seq, top_k, 1]
            # y * weight -> sum(dim=1) -> [batch*seq, hidden]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

            # 恢复原始形状: [batch, seq, hidden]
            y = y.view(*orig_shape)
        else:
            # ==================== 推理模式 ====================
            # 推理时使用优化的内核 (moe_infer)，避免 Python 循环，提高速度
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        # 2. 共享专家计算 (Shared Experts)
        # 共享专家的输出直接叠加到路由专家的输出上
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE 推理优化实现。
        相比于训练时的 mask 掩码方式，这里采用了基于排序 (Sort) 的方法来聚集 Token，
        从而实现更高效的批量计算。

        Args:
            x: 输入张量 [batch*seq, hidden]
            flat_expert_indices: 展平的专家索引 [batch*seq*top_k]
            flat_expert_weights: 展平的专家权重 [batch*seq*top_k, 1]
        """
        expert_cache = torch.zeros_like(x)

        # 1. 对专家索引进行排序
        # idxs: 排序后的索引，用于知道原来的 token 在哪
        # 例如 flat_expert_indices = [1, 0, 1, 2] -> sorted -> [0, 1, 1, 2], idxs=[1, 0, 2, 3]
        idxs = flat_expert_indices.argsort()

        # 2. 计算每个专家分配到的 Token 数量
        # tokens_per_expert: 例如 [token_count_exp0, token_count_exp1, ...]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 3. 映射回原始 Token 索引
        # 因为 x 是 [batch*seq]，而 idxs 是 [batch*seq*top_k]
        # 所以需要除以 top_k 来找到对应的原始输入行号
        token_idxs = idxs // self.config.num_experts_per_tok

        # 4. 遍历每个专家 (按排序后的区间处理)
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 该专家未被任何 Token 选中

            expert = self.experts[i]

            # 获取属于该专家的原始 Token 索引
            exp_token_idx = token_idxs[start_idx:end_idx]

            # 提取输入数据
            expert_tokens = x[exp_token_idx]

            # 执行专家计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # 乘上对应的路由权重
            # idxs[start_idx:end_idx] 对应 flat_expert_weights 中的正确位置
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 5. 将结果累加回输出张量 (Scatter Add)
            # 因为一个 Token 可能被多个专家处理，所以需要用 scatter_add_ 将结果累加到对应的位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    Transformer Decoder Block (解码器层)。
    这是构建大型语言模型 (LLM) 的基本单元，通常会重复堆叠多次 (例如 Llama 2 有 32 层)。

    架构遵循标准的 Pre-Norm 结构：
    Input -> RMSNorm -> Attention -> Residual -> RMSNorm -> FFN/MoE -> Residual -> Output
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 1. 自注意力机制 (Self-Attention)
        # 负责捕捉序列中 Token 之间的依赖关系
        self.self_attn = Attention(config)

        self.layer_id = layer_id

        # 2. 归一化层 (Normalization)
        # 使用 RMSNorm (Root Mean Square Normalization)，相比 LayerNorm 计算更高效且效果相当。
        # input_layernorm: 用于 Attention 之前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # post_attention_layernorm: 用于 FFN 之前的归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # 3. 前馈神经网络 (Feed-Forward Network)
        # 如果配置启用了混合专家 (MoE)，则使用 MOEFeedForward，否则使用标准的 SwiGLU FFN。
        # FFN 负责处理每个 Token 的非线性变换，增加模型的表达能力。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
            self,
            hidden_states,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=None,
    ):
        """
        前向传播函数。

        Args:
            hidden_states: 输入的隐藏状态张量 [batch_size, seq_len, hidden_size]。
            position_embeddings: 预计算的旋转位置编码 (cos, sin)。
            past_key_value: 上一步的 KV Cache，用于加速推理。
            use_cache: 是否使用 KV Cache。
            attention_mask: 注意力掩码，用于处理 padding 或因果遮蔽。
        """
        # ==================== Attention 子层 ====================

        # 保存残差连接的原始输入 (Residual Connection)
        residual = hidden_states

        # 1. Pre-Norm: 先进行 RMSNorm 归一化
        # 2. Self-Attention: 计算注意力输出
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        # 3. Add Residual: 将注意力输出与原始输入相加 (Add)
        # 公式: x = x + Attention(Norm(x))
        hidden_states += residual

        # ==================== FFN / MoE 子层 ====================

        # 保存上一阶段的输出作为新的残差基准
        residual = hidden_states

        # 1. Pre-Norm: 再次进行 RMSNorm 归一化
        # 2. MLP: 通过前馈网络 (或 MoE)
        # 3. Add Residual: 相加
        # 公式: x = x + MLP(Norm(x))
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 模型核心 (Core Model)。

    这是不包含 LM Head (语言模型头) 的基础 Transformer 模型。
    它的输出是最后一层的隐藏状态 (hidden_states)，而不是词表上的概率分布。
    通常用于特征提取、或者作为 MiniMindForCausalLM 的一部分。

    结构:
    Embeddings -> Dropout -> Decoder Layers (x N) -> Final RMSNorm
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        # 1. 词嵌入层 (Token Embeddings)
        # 将输入的 Token ID 转换为密集向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 2. Dropout 层
        self.dropout = nn.Dropout(config.dropout)

        # 3. 解码器层堆叠 (Decoder Layers Stack)
        # 由多个 MiniMindBlock 组成
        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )

        # 4. 最终归一化层 (Final RMSNorm)
        # 在输出之前进行的最后一次归一化，有助于训练稳定性
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 5. 预计算 RoPE 旋转位置编码 (Precompute RoPE Frequencies)
        # 一次性计算好所有可能位置的 cos 和 sin 值，放入缓冲区 (buffer) 中，避免在前向传播中重复计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # register_buffer 会将张量注册为模型的一部分（随模型保存/加载/移动设备），但不会被视为可训练参数 (Parameter)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            **kwargs,
    ):
        """
        前向传播函数。

        Args:
            input_ids: 输入 Token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 历史 KV Cache，用于推理加速
            use_cache: 是否启用 KV Cache

        Returns:
            hidden_states: 最后一层的输出特征 [batch_size, seq_len, hidden_size]
            presents: 当前步骤生成的新的 KV Cache
            aux_loss: 所有 MoE 层的辅助损失之和
        """
        batch_size, seq_length = input_ids.shape

        # 初始化或处理 KV Cache
        # 如果 past_key_values 不符合预期格式（如某些库传递的对象），则重置为 None
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前序列在绝对位置中的起始点
        # 如果有 cache，说明是接在之前的生成后面的，起始位置 = cache 长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # 1. 获取 Embedding 并应用 Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 2. 截取当前序列对应的 RoPE 编码
        # 根据绝对位置索引，从预计算好的 buffer 中切片出对应的 cos/sin
        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_length],
            self.freqs_sin[start_pos: start_pos + seq_length],
        )

        presents = []  # 用于收集每一层新的 KV Cache

        # 3. 逐层前向传播
        for layer_idx, (layer, past_key_value) in enumerate(
                zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)

        # 5. 收集所有 MoE 层的辅助损失
        # 遍历所有层，如果是 MoE 层，就将其 aux_loss 累加起来
        # hidden_states.new_zeros(1).squeeze() 是为了创建一个与 hidden_states 同设备、同类型的 0 标量作为初始值
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型 (Causal Language Model)。

    这是可以直接用于生成任务（如文本补全、对话）的完整模型。
    它包含：
    1. 基础模型 (MiniMindModel): 负责提取文本特征。
    2. 语言模型头 (LM Head): 负责将特征映射到词表上的概率分布，预测下一个 Token。

    继承自 PreTrainedModel 和 GenerationMixin，这意味着它集成了 Hugging Face Transformers 库的
    强大功能，如 .save_pretrained(), .from_pretrained(), .generate() 等。
    """

    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        # 如果未提供配置，则使用默认配置
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        # 1. 实例化基础 Transformer 模型
        self.model = MiniMindModel(self.config)

        # 2. 实例化语言模型头 (LM Head)
        # 这是一个线性层，将 hidden_size 映射到 vocab_size
        # bias=False: 通常为了参数共享和更稳定的训练，不使用偏置
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        # 3. 权重绑定 (Weight Tying)
        # 将 Embedding 层的权重与 LM Head 的权重共享。
        # 这是一个常见的技巧 (Attention Is All You Need, GPT-2 等)，
        # 既能减少参数量（Embedding 表通常很大），又能提升模型性能（输入和输出语义空间一致）。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **args,
    ):
        """
        前向传播函数。

        Args:
            input_ids: 输入 Token IDs
            attention_mask: 注意力掩码
            labels: 标签 Token IDs (用于训练计算 Loss)
            past_key_values: 历史 KV Cache
            use_cache: 是否使用 KV Cache
            logits_to_keep: 优化显存的参数。
                            如果为 0，则计算所有 Token 的 Logits。
                            如果为 k (int)，则只保留最后 k 个 Token 的 Logits（通常推理时只需要最后一个）。
                            这对于大词表的模型训练可以显著节省显存。
        """
        # 1. 调用基础模型获取隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 2. 切片 Hidden States (优化显存)
        # 如果 logits_to_keep 不为 0，则只保留需要计算 Logits 的部分特征
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # 应用 LM Head 计算 Logits
        # Logits: 未经 Softmax 归一化的概率分数
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # ==================== 训练模式：计算 Loss ====================

            # 3. 移位 (Shift) 以对齐预测目标
            # 因果语言模型的目标是预测下一个 Token。
            # 输入序列: [A, B, C, D]
            # 预测目标: [B, C, D, E]
            # 所以 logits 应该取 [:-1] (预测 B, C, D)，labels 应该取 [1:] (真实 B, C, D)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 4. 计算交叉熵损失 (Cross Entropy Loss)
            # view(-1, ...) 将 batch 和 seq 维度展平，符合 CrossEntropyLoss 的输入要求
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # 忽略 padding token 或其他不需要计算 loss 的位置
            )

        # 5. 封装输出
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        # 将辅助损失附加到输出对象上
        output.aux_loss = aux_loss
        return output


def get_model_params(model, config):
    """
    计算并打印模型的参数量统计信息。
    对于 MoE 模型，会分别计算总参数量 (Sparse/Total Params) 和 激活参数量 (Active Params)。

    Args:
        model: 模型实例
        config: 模型配置对象
    """
    # 1. 计算总参数量 (Total Parameters)
    # sum(p.numel()...) 遍历所有参数张量并求和
    # / 1e6 将单位转换为百万 (M)
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. 获取 MoE 相关配置
    # n_routed: 路由专家总数 (总共有多少个专家)
    # n_active: 每次激活的专家数 (top_k)
    # n_shared: 共享专家数 (总是激活)
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)

    # 3. 计算单个专家的参数量
    # 这里通过遍历 named_parameters 并匹配名字来估算
    # 'mlp.experts.0.' 是路由专家列表中第一个专家的参数前缀
    expert = (
            sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n)
            / 1e6
    )

    # 4. 计算单个共享专家的参数量
    # 'mlp.shared_experts.0.' 是共享专家列表中第一个专家的参数前缀
    shared_expert = (
            sum(
                p.numel()
                for n, p in model.named_parameters()
                if "mlp.shared_experts.0." in n
            )
            / 1e6
    )

    # 5. 计算非专家部分的参数量 (Base Params)
    # Base = Total - 所有路由专家 - 所有共享专家
    # 这部分包括 Embedding, Attention, Norm, Output Head 等，它们对每个 token 都是必算的
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 6. 计算激活参数量 (Active Parameters)
    # Active = Base + (激活的路由专家数 * 单个专家参数) + (所有共享专家参数)
    # 这是推理时处理一个 token 实际需要参与计算的参数量，直接决定了推理速度
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 7. 打印结果
    # 如果 active < total，说明是稀疏模型 (MoE)，打印 Total 和 Active 两个数值
    # 格式: Total(M)-A(Active)(M)，例如 47B-A13B
    if active < total:
        print(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        # 否则是稠密模型 (Dense)，Total == Active
        print(f"Model Params: {total:.2f}M")


def init_model(
        lm_config,
        from_weight="pretrain",
        tokenizer_path=".",
        save_dir="./dist",
        device="cuda",
):
    """
    初始化模型和分词器，并选择性地加载预训练权重。

    Args:
        lm_config: 模型配置对象
        from_weight: 预训练权重的来源名称，如 'pretrain' 或 'sft'。如果为 'none' 则随机初始化。
        tokenizer_path: 分词器配置文件的路径
        save_dir: 权重文件的保存目录
        device: 模型加载的目标设备 (如 'cuda', 'cpu')

    Returns:
        model: 加载并移动到指定设备的模型实例
        tokenizer: 加载好的分词器实例
    """
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 实例化模型架构
    # 此时模型参数是随机初始化的
    model = MiniMindForCausalLM(lm_config)

    # 3. 加载预训练权重 (如果指定)
    if from_weight != "none":
        # 构造权重文件路径，自动处理 MoE 后缀
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        # 加载 state_dict 到指定设备
        # map_location=device 确保直接加载到目标设备，避免先加载到 CPU 再移动的开销
        weights = torch.load(weight_path, map_location=device)

        # 将权重加载进模型
        # strict=False 允许忽略不匹配的键 (例如从旧版本权重加载时)
        model.load_state_dict(weights, strict=False)

    # 4. 打印参数统计信息
    get_model_params(model, lm_config)

    # 打印可训练参数量 (Trainable Params)
    # 有时我们冻结部分参数 (如 LoRA 微调)，这里只统计 requires_grad=True 的参数
    print(
        f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )

    # 5. 返回模型和分词器
    return model.to(device), tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to("cuda:0")
        labels = labels.to("cuda:0")
        lr = get_lr(epoch * iters + step, 1 * iters, 5e-4)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / 8

        scaler.scale(loss).backward()

        if (step + 1) % 8 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * 8
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            print(
                f"Epoch:[{epoch + 1}/{1}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % 1000 == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"./dist/pretrain_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight="pretrain",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="./checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    device = None
    # 初始化分布式训练环境
    # local_rank = init_distributed_mode()
    # if dist.is_initialized():
    #     device = f"cuda:{local_rank}"
    # 设置随机种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    # 定义模型配置
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=bool(0))
    ckp_data = lm_checkpoint(lm_config, weight="pretrain", save_dir="./checkpoints")
    dtype = torch.bfloat16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(device_type)
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, "none", device="cuda:0")
    train_ds = PretrainDataset("./pretrain_hq.jsonl", tokenizer, max_length=340)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=("bfloat16" == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. DDP包模型 ==========
    # if dist.is_initialized():
    #     model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    #     model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    epochs = 1
    batch_size = 32
    num_workers = 8
    for epoch in range(start_epoch, epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), batch_size, start_step + 1
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            print(
                f"Epoch [{epoch + 1}/{epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, None)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, None)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
