# PagedAttention 技术详解

## 概述

PagedAttention 是 vLLM 的核心技术创新，它借鉴了操作系统虚拟内存分页管理的思想，解决了大语言模型推理过程中 KV Cache 的内存管理难题。

## 背景：KV Cache 的挑战

### 什么是 KV Cache？

在 Transformer 模型的自注意力机制中，每个 token 都需要与之前所有 token 进行注意力计算。为了避免重复计算，模型会缓存每个 token 的 Key 和 Value 向量，这就是 KV Cache。

### 传统 KV Cache 的三大问题

#### 1. 显存占用增长快

- **问题描述**：随着生成序列长度增加，KV Cache 占用的显存呈线性增长
- **影响**：对于长文本生成任务，KV Cache 可能占用模型参数数倍的显存
- **示例**：一个 13B 参数的模型，生成 2048 个 token 时，KV Cache 可能占用 20GB+ 显存

#### 2. 内存碎片严重

- **问题描述**：不同请求的序列长度不同，导致内存分配不均，产生大量碎片
- **影响**：即使总显存充足，也可能因为碎片化而无法分配新的内存块
- **示例**：100 个请求，长度从 100 到 2000 不等，传统方法需要为每个请求预分配最大长度的内存

#### 3. 缓存难以复用

- **问题描述**：不同请求即使有相同的前缀（如系统提示词），也无法共享 KV Cache
- **影响**：浪费大量显存存储重复的计算结果
- **示例**：1000 个请求都使用相同的 500 token 系统提示词，传统方法需要存储 1000 份相同的 KV Cache

## PagedAttention 的解决方案

### 核心思想

PagedAttention 将 KV Cache 分割成固定大小的"页"（pages），类似操作系统的虚拟内存分页：

```
传统方法：
Request 1: [████████████████████████] (连续内存，2000 tokens)
Request 2: [████████] (连续内存，800 tokens)
Request 3: [████████████] (连续内存，1200 tokens)
问题：需要预分配最大长度，产生碎片

PagedAttention：
Request 1: [Page1][Page2][Page3][Page4]... (按需分配)
Request 2: [Page1][Page2]... (按需分配)
Request 3: [Page1][Page2][Page3]... (按需分配)
优势：按需分配，减少碎片，可共享页
```

### 关键机制

#### 1. 分页管理

- **页大小**：固定大小（如 16 或 32 个 token）
- **页表**：维护逻辑地址到物理地址的映射
- **按需分配**：只在需要时分配新页，避免预先分配大块连续内存

```python
# 伪代码示例
class PagedKVCache:
    def __init__(self, page_size=16):
        self.page_size = page_size
        self.physical_pages = []  # 物理页池
        self.page_tables = {}     # 每个请求的页表
    
    def allocate_page(self, request_id):
        """为请求分配新页"""
        if not self.free_pages:
            self.physical_pages.append(self.create_new_page())
        page = self.free_pages.pop()
        self.page_tables[request_id].append(page)
        return page
```

#### 2. 内存共享

不同请求可以共享相同的 KV Cache 页，特别适合以下场景：

- **系统提示词共享**：多个请求使用相同的系统提示词
- **前缀共享**：多个请求有相同的前缀
- **并行采样**：同一个请求生成多个候选序列

```
示例：3 个请求共享系统提示词

Request 1: [Shared Page 1][Shared Page 2][Private Page 1][Private Page 2]
Request 2: [Shared Page 1][Shared Page 2][Private Page 3]
Request 3: [Shared Page 1][Shared Page 2][Private Page 4][Private Page 5]

共享页只存储一次，节省显存
```

#### 3. 写时复制（Copy-on-Write）

当需要修改共享页时，采用写时复制机制：

1. 检测到写操作
2. 复制共享页到新的物理页
3. 更新页表指向新页
4. 在新页上执行写操作

```python
def write_to_page(request_id, page_index, data):
    """写入页，如果是共享页则先复制"""
    page = self.page_tables[request_id][page_index]
    
    if page.is_shared():
        # 写时复制
        new_page = page.copy()
        self.page_tables[request_id][page_index] = new_page
        page = new_page
    
    page.write(data)
```

### 技术优势

#### 1. 显存利用率提升

- **按需分配**：只分配实际需要的内存，避免预分配浪费
- **减少碎片**：固定大小的页减少内存碎片
- **提升利用率**：显存利用率从 20-40% 提升到 80-90%

#### 2. 吞吐量提升

- **更多并发**：相同显存下可以处理更多并发请求
- **批处理优化**：更高效的批处理，提升吞吐量
- **性能提升**：相比传统方法，吞吐量提升 2-24 倍

#### 3. 灵活性增强

- **动态长度**：支持动态序列长度，无需预先指定最大长度
- **内存共享**：自动检测和共享相同的前缀
- **资源调度**：更灵活的资源调度和负载均衡

## 实现细节

### 页表结构

```python
class PageTable:
    """页表：逻辑地址到物理地址的映射"""
    def __init__(self):
        self.logical_to_physical = []  # 逻辑页号 -> 物理页号
    
    def get_physical_page(self, logical_page_num):
        """获取物理页号"""
        return self.logical_to_physical[logical_page_num]
    
    def add_page(self, physical_page_num):
        """添加新页"""
        self.logical_to_physical.append(physical_page_num)
```

### 物理页池

```python
class PhysicalPagePool:
    """物理页池：管理所有物理页"""
    def __init__(self, num_pages, page_size):
        self.num_pages = num_pages
        self.page_size = page_size
        self.pages = [Page(page_size) for _ in range(num_pages)]
        self.free_pages = set(range(num_pages))
    
    def allocate(self):
        """分配一个空闲页"""
        if not self.free_pages:
            raise OutOfMemoryError("No free pages available")
        page_num = self.free_pages.pop()
        return page_num
    
    def free(self, page_num):
        """释放一个页"""
        self.free_pages.add(page_num)
```

### 注意力计算

PagedAttention 需要修改注意力计算，以支持分页的 KV Cache：

```python
def paged_attention(query, page_table, physical_pages):
    """
    分页注意力计算
    
    Args:
        query: 查询向量 [batch_size, num_heads, head_dim]
        page_table: 页表，映射逻辑页到物理页
        physical_pages: 物理页池
    
    Returns:
        attention_output: 注意力输出
    """
    attention_scores = []
    
    # 遍历所有逻辑页
    for logical_page_num in range(len(page_table)):
        # 获取物理页
        physical_page_num = page_table.get_physical_page(logical_page_num)
        page = physical_pages[physical_page_num]
        
        # 获取该页的 K, V
        keys = page.get_keys()
        values = page.get_values()
        
        # 计算注意力分数
        scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_scores.append(scores)
    
    # 合并所有页的注意力分数
    all_scores = torch.cat(attention_scores, dim=-1)
    attention_weights = torch.softmax(all_scores, dim=-1)
    
    # 计算加权和
    # ... (省略详细实现)
    
    return attention_output
```

## 性能对比

### 吞吐量对比

| 框架 | 吞吐量 (requests/s) | 相对提升 |
|------|-------------------|---------|
| HuggingFace Transformers | 1.0x | 基准 |
| vLLM (PagedAttention) | 2-24x | 2-24 倍 |

### 显存利用率对比

| 框架 | 显存利用率 | 可支持并发数 |
|------|-----------|------------|
| 传统方法 | 20-40% | 10-20 |
| PagedAttention | 80-90% | 50-100 |

### 延迟对比

| 场景 | 传统方法 | PagedAttention | 改善 |
|------|---------|---------------|------|
| 短文本 (100 tokens) | 50ms | 45ms | 10% |
| 中文本 (500 tokens) | 200ms | 150ms | 25% |
| 长文本 (2000 tokens) | 800ms | 500ms | 37.5% |

## 适用场景

### 最适合的场景

1. **高并发服务**：需要同时处理大量请求
2. **长文本生成**：生成长度不确定或较长的文本
3. **批量推理**：批量处理多个请求
4. **共享前缀**：多个请求有相同的系统提示词或前缀
5. **并行采样**：需要为同一个请求生成多个候选序列

### 不太适合的场景

1. **单请求低延迟**：单个请求的绝对延迟可能略高于优化的单请求推理
2. **极短文本**：对于极短文本（少于50 tokens），分页开销可能不划算
3. **显存充足**：如果显存非常充足，传统方法也能工作良好

## 与操作系统分页的类比

| 操作系统分页 | PagedAttention | 说明 |
|------------|---------------|------|
| 虚拟地址空间 | 逻辑 KV Cache 地址 | 程序/请求看到的地址 |
| 物理内存 | GPU 显存 | 实际存储位置 |
| 页表 | KV Cache 页表 | 地址映射关系 |
| 页 | KV Cache 页 | 固定大小的内存块 |
| 按需分页 | 按需分配 KV Cache | 只在需要时分配 |
| 页面共享 | KV Cache 共享 | 多个进程/请求共享相同页 |
| 写时复制 | 写时复制 | 修改共享页时先复制 |

## 总结

PagedAttention 是 vLLM 的核心创新，通过借鉴操作系统的虚拟内存分页思想，解决了大语言模型推理中的内存管理难题：

- ✅ **显著提升显存利用率**：从 20-40% 提升到 80-90%
- ✅ **大幅提升吞吐量**：相比传统方法提升 2-24 倍
- ✅ **支持更多并发**：相同硬件下可处理更多请求
- ✅ **减少内存碎片**：固定大小的页减少碎片化
- ✅ **实现内存共享**：自动共享相同的前缀，节省显存

PagedAttention 使得 vLLM 成为大语言模型推理的高性能引擎，特别适合企业级服务和高并发场景。

