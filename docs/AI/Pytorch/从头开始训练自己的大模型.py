import math
import os
import random
from contextlib import nullcontext
from typing import Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
import torch.nn.functional as F

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
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
        x2 = x[..., x.shape[-1] // 2 :]
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
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


if __name__ == "__main__":
    device = None
    # 初始化分布式训练环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        device = f"cuda:{local_rank}"
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
