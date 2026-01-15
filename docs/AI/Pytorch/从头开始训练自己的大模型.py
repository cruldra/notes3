import os
import random
from contextlib import nullcontext
from logging import Logger

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
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
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
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
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
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


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None,
                  save_dir='../checkpoints', **kwargs):
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
    moe_path = '_moe' if lm_config.use_moe else ''  # MoE 模型添加特殊后缀
    # 纯权重文件路径 (仅包含 model state_dict，体积小，用于推理)
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 恢复文件路径 (包含 model, optimizer, step, epoch 等，用于断点续训)
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ==================== 保存模式 ====================

        # 2. 解包模型 (Unwrap)
        # 如果是 DDP (分布式) 模型，取其 .module
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 如果是 torch.compile 编译后的模型，取其 _orig_mod
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 3. 处理模型权重 (Save Model Weights)
        state_dict = raw_model.state_dict()
        # 将权重转为半精度 (FP16) 并移动到 CPU，节省存储空间和显存
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 4. 原子保存权重文件 (Atomic Save)
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)  # 先写入临时文件
        os.replace(ckp_tmp, ckp_path)  # 原子替换，防止写入中断导致文件损坏

        # 5. 获取 WandB Run ID (用于恢复曲线)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 6. 构建恢复数据字典 (Resume Data)
        resume_data = {
            'model': state_dict,  # 模型权重
            'optimizer': optimizer.state_dict(),  # 优化器状态
            'epoch': epoch,  # 当前 Epoch
            'step': step,  # 当前 Step
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # 保存时的 GPU 数量
            'wandb_id': wandb_id  # WandB ID
        }

        # 7. 处理额外的 kwargs (如 LR Scheduler)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 如果是 DDP 或编译后的对象，同样需要解包
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 8. 原子保存恢复文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 9. 清理资源
        del state_dict, resume_data
        torch.cuda.empty_cache()  # 释放显存

    else:
        # ==================== 加载模式 ====================

        if os.path.exists(resume_path):
            # 1. 加载恢复文件到 CPU
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 2. 处理 GPU 数量变化带来的 Step 差异
            # 场景：例如 4 卡变 8 卡，Global Batch Size 翻倍，总 Step 数应减半
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                print(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data
        return None

if __name__ == "__main__":
    device = None
    # 初始化分布式训练环境
    local_rank = init_distributed_mode()
    if dist.is_initialized(): device = f"cuda:{local_rank}"
    # 设置随机种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    # 定义模型配置
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=bool(0))
    ckp_data = lm_checkpoint(lm_config, weight="pretrain",
                             save_dir='./checkpoints')
    dtype = torch.bfloat16
    device_type ="cuda" if torch.cuda.is_available()    else "cpu"
    print(device_type)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)