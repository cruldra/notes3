import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch.utils.data import Dataset, TensorDataset, ConcatDataset, Subset, random_split
    import numpy as np
    return (
        ConcatDataset,
        Dataset,
        Subset,
        TensorDataset,
        mo,
        random_split,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # torch.utils.data.Dataset 模块演示

    本笔记本演示 PyTorch 中 Dataset 相关的各个 API 用法。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 自定义 Dataset

    最基础的用法是继承 `Dataset` 类并实现 `__len__` 和 `__getitem__` 方法。
    """)
    return


@app.cell
def _(Dataset, torch):
    class CustomDataset(Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 10)
            self.labels = torch.randint(0, 2, (size,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    custom_dataset = CustomDataset(size=50)
    print(f"数据集大小: {len(custom_dataset)}")
    print(f"第一个样本: {custom_dataset[0]}")
    print(f"特征形状: {custom_dataset[0][0].shape}")
    print(f"标签: {custom_dataset[0][1].item()}")
    return CustomDataset, custom_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. TensorDataset

    `TensorDataset` 是最简单的内置 Dataset，用于包装张量数据。
    """)
    return


@app.cell
def _(TensorDataset, torch):
    features = torch.randn(100, 5)
    labels = torch.randint(0, 3, (100,))

    tensor_dataset = TensorDataset(features, labels)

    print(f"TensorDataset 大小: {len(tensor_dataset)}")
    print(f"第一个样本: {tensor_dataset[0]}")
    print(f"可以同时包装多个张量")
    return (tensor_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Subset - 数据集子集

    `Subset` 用于创建数据集的子集，通过索引列表选择特定样本。
    """)
    return


@app.cell
def _(Subset, tensor_dataset):
    indices = [0, 5, 10, 15, 20]
    subset = Subset(tensor_dataset, indices)

    print(f"原始数据集大小: {len(tensor_dataset)}")
    print(f"子集大小: {len(subset)}")
    print(f"子集第一个样本 (原始索引0): {subset[0]}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. random_split - 随机分割数据集

    `random_split` 用于将数据集随机分割成多个子集，常用于训练集/验证集/测试集划分。
    """)
    return


@app.cell
def _(random_split, tensor_dataset, torch):
    torch.manual_seed(42)

    train_size = int(0.7 * len(tensor_dataset))
    val_size = int(0.15 * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        tensor_dataset, 
        [train_size, val_size, test_size]
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"总大小: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. ConcatDataset - 连接多个数据集

    `ConcatDataset` 用于将多个数据集连接成一个大数据集。
    """)
    return


@app.cell
def _(ConcatDataset, CustomDataset):
    dataset1 = CustomDataset(size=30)
    dataset2 = CustomDataset(size=20)
    dataset3 = CustomDataset(size=50)

    concat_dataset = ConcatDataset([dataset1, dataset2, dataset3])

    print(f"数据集1大小: {len(dataset1)}")
    print(f"数据集2大小: {len(dataset2)}")
    print(f"数据集3大小: {len(dataset3)}")
    print(f"连接后数据集大小: {len(concat_dataset)}")
    print(f"访问第35个样本 (来自dataset2): {concat_dataset[35][0].shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. 数据集索引和切片

    Dataset 支持索引访问，某些实现也支持切片操作。
    """)
    return


@app.cell
def _(custom_dataset):
    print("单个索引访问:")
    sample = custom_dataset[5]
    print(f"  样本5: 特征形状={sample[0].shape}, 标签={sample[1].item()}")

    print("\n批量索引访问:")
    for i in range(3):
        sample = custom_dataset[i]
        print(f"  样本{i}: 标签={sample[1].item()}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. 实际应用示例 - 图像分类数据集

    创建一个模拟图像分类任务的数据集。
    """)
    return


@app.cell
def _(Dataset, torch):
    class ImageClassificationDataset(Dataset):
        def __init__(self, num_samples=1000, img_size=(3, 224, 224), num_classes=10):
            self.num_samples = num_samples
            self.img_size = img_size
            self.num_classes = num_classes

            self.images = torch.randn(num_samples, *img_size)
            self.labels = torch.randint(0, num_classes, (num_samples,))

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]

            image = (image - image.mean()) / (image.std() + 1e-8)

            return image, label

        def get_class_distribution(self):
            unique, counts = torch.unique(self.labels, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))

    img_dataset = ImageClassificationDataset(num_samples=500, num_classes=5)

    print(f"图像数据集大小: {len(img_dataset)}")
    print(f"图像形状: {img_dataset[0][0].shape}")
    print(f"类别分布: {img_dataset.get_class_distribution()}")
    return (img_dataset,)


@app.cell
def _(mo):
    mo.md("""
    ## 8. 数据集组合使用

    展示如何组合使用多个 Dataset 工具。
    """)
    return


@app.cell
def _(Subset, img_dataset, random_split, torch):
    torch.manual_seed(123)

    train_set, temp_set = random_split(img_dataset, [400, 100])
    val_set, test_set = random_split(temp_set, [50, 50])

    print("数据集划分:")
    print(f"  训练集: {len(train_set)} 样本")
    print(f"  验证集: {len(val_set)} 样本")
    print(f"  测试集: {len(test_set)} 样本")

    small_train_indices = list(range(0, 100))
    small_train_set = Subset(train_set, small_train_indices)
    print(f"\n小训练集 (前100个样本): {len(small_train_set)} 样本")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 总结

    本笔记本演示了 `torch.utils.data.Dataset` 模块的主要 API:

    1. **Dataset** - 基础抽象类，需要实现 `__len__` 和 `__getitem__`
    2. **TensorDataset** - 简单包装张量的数据集
    3. **Subset** - 创建数据集子集
    4. **random_split** - 随机分割数据集
    5. **ConcatDataset** - 连接多个数据集

    这些工具可以灵活组合使用，满足各种数据处理需求。
    """)
    return


if __name__ == "__main__":
    app.run()
