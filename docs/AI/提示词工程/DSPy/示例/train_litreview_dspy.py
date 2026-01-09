"""
DSPy 文献综述生成训练脚本
使用 scripts/abstracts 和 scripts/litreviews 中的真实数据
"""

import os
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import glob

# -------------------------------------------------------------------------
# 1. 配置与设置
# -------------------------------------------------------------------------

def setup_dspy():
    """配置 DSPy 使用指定的 LLM。"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # 检查 DeepSeek 配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    
    if not api_key:
        print("警告: 环境变量中未找到 DEEPSEEK_API_KEY。")
    
    lm = dspy.LM(
        model="deepseek/deepseek-chat",
        api_key=api_key,
        api_base=base_url,
        #max_tokens=4000, # 针对文献综述增加了长度限制
        #cache=False
    )
    dspy.configure(lm=lm)
    return lm

# -------------------------------------------------------------------------
# 2. 数据加载
# -------------------------------------------------------------------------

def load_dataset():
    """从文件系统加载摘要和文献综述对。"""
    abstracts_dir = os.path.join(os.path.dirname(__file__), 'abstracts')
    litreviews_dir = os.path.join(os.path.dirname(__file__), 'litreviews')
    
    dataset = []
    
    # 列出所有摘要文件
    abstract_files = glob.glob(os.path.join(abstracts_dir, "*.txt"))
    
    print(f"找到 {len(abstract_files)} 个摘要文件。")
    
    for abs_path in abstract_files:
        filename = os.path.basename(abs_path)
        lit_path = os.path.join(litreviews_dir, filename)
        
        if os.path.exists(lit_path):
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    abstract_text = f.read().strip()
                
                with open(lit_path, 'r', encoding='utf-8') as f:
                    lit_review_text = f.read().strip()
                
                if abstract_text and lit_review_text:
                    # 创建 DSPy 样本 (Example)
                    # 输入: abstract (摘要)
                    # 标签: literature_review (文献综述)
                    example = dspy.Example(
                        abstract=abstract_text,
                        literature_review=lit_review_text
                    ).with_inputs('abstract')
                    
                    dataset.append(example)
                else:
                    print(f"跳过空文件对: {filename}")
            except Exception as e:
                print(f"读取 {filename} 时出错: {e}")
        else:
            print(f"警告: 未找到对应的文献综述文件 {filename}")
            
    print(f"成功加载 {len(dataset)} 个样本。")
    return dataset

# -------------------------------------------------------------------------
# 3. DSPy 签名 (Signature) 与 模块 (Module)
# -------------------------------------------------------------------------

class GenerateLitReview(dspy.Signature):
    """
    根据提供的【相关文献摘要列表】或【论文摘要】，撰写一份高质量的中文文献综述。
    
    要求：
    1. 如果输入包含多篇文献摘要，请综合归纳这些文献的观点、方法和结论。
    2. 结构应包含：核心概念定义、主要理论基础、研究现状（分类综述）、现有研究的不足与趋势。
    3. 语言学术规范，逻辑严密。
    """
    
    abstract = dspy.InputField(desc="文本内容，可以是单篇论文的摘要，也可以是多篇参考文献摘要的集合列表。")
    literature_review = dspy.OutputField(desc="生成的中文文献综述章节。")

class LitReviewModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought 帮助模型规划综述的结构
        self.generate = dspy.ChainOfThought(GenerateLitReview)
        
    def forward(self, abstract):
        return self.generate(abstract=abstract)

# -------------------------------------------------------------------------
# 4. 评估指标 (Metrics)
# -------------------------------------------------------------------------

def review_quality_metric(example, pred, trace=None):
    """
    评估生成综述质量的启发式指标。
    """
    generated_text = pred.literature_review
    ground_truth = example.literature_review
    
    score = 0.0
    
    # 1. 结构检查 (通常在文献综述中出现的关键词)
    keywords = ["文献综述", "定义", "现状", "理论", "研究", "不足", "趋势"]
    found_keywords = sum(1 for k in keywords if k in generated_text)
    score += (found_keywords / len(keywords)) * 0.4
    
    # 2. 长度检查 (文献综述应该有一定篇幅)
    # 惩罚过短的回答
    if len(generated_text) > 500:
        score += 0.3
    else:
        score += (len(generated_text) / 500) * 0.3
        
    # 3. 引用格式检查 (简单检查是否有括号引用)
    if "[" in generated_text and "]" in generated_text:
        score += 0.2
        
    # 4. 中文字符检查 (确保输出是中文，即使输入是英文)
    # 简单启发式：检查是否存在常见的中文助词“的”
    if "的" in generated_text:
        score += 0.1
        
    return score

# -------------------------------------------------------------------------
# 5. 训练与评估
# -------------------------------------------------------------------------

def main():
    print("正在初始化 DSPy...")
    lm = setup_dspy()
    
    print("\n正在加载数据集...")
    full_dataset = load_dataset()
    
    if not full_dataset:
        print("未找到数据。退出。")
        return

    # 分割数据集 (简单分割)
    # 假设我们有 ~8 个样本，使用 5 个用于训练，3 个用于测试
    train_size = min(5, len(full_dataset) - 1)
    if train_size < 1: train_size = 1 # 针对极少量数据的回退策略
    
    trainset = full_dataset[:train_size]
    testset = full_dataset[train_size:]
    
    print(f"训练集: {len(trainset)} 个样本, 测试集: {len(testset)} 个样本。")
    
    # 定义模块
    module = LitReviewModule()
    
    # 定义优化器 (Teleprompter)
    # BootstrapFewShot 使用评估指标从训练集中选择最佳的少样本示例 (Few-Shot Examples)
    # 并为它们生成推理过程 (Chain of Thought)。
    teleprompter = BootstrapFewShot(metric=review_quality_metric, max_bootstrapped_demos=2, max_labeled_demos=2)
    
    print("\n正在编译 (训练) 模型...")
    compiled_module = teleprompter.compile(module, trainset=trainset)
    
    print("\n正在测试集上进行评估...")
    # 设置评估器
    evaluator = Evaluate(devset=testset, metric=review_quality_metric, num_threads=1, display_progress=True)
    score = evaluator(compiled_module)
    print(f"测试集得分: {score}")
    
    # 保存编译后的程序
    save_path = os.path.join(os.path.dirname(__file__), "litreview_dspy_model.json")
    compiled_module.save(save_path)
    print(f"\n编译后的模型已保存至: {save_path}")
    
    # 导出 Prompt 供检查
    print("\n正在导出 Prompt 供检查...")
    
    # 尝试打印测试集的一个生成结果
    if testset:
        print("\n--- 生成示例 (测试集) ---")
        ex = testset[0]
        pred = compiled_module(abstract=ex.abstract)
        print(f"输入摘要 (片段): {ex.abstract[:100]}...")
        print(f"生成的综述 (片段): {pred.literature_review[:200]}...")
        print("-------------------------------------")

if __name__ == "__main__":
    main()