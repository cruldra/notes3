# panic! 与异常（throw）的区别

## 🤔 为什么 Rust 的 panic! 不叫 throw？

`panic!` 和其他语言的 `throw` 虽然看起来相似，但实际上有本质区别，这也是为什么 Rust 选择了不同的名字。

---

## 🔍 本质区别

### 1. 设计哲学不同

#### 其他语言的 throw（异常机制）

- ✅ 异常是**正常的控制流**的一部分
- ✅ 可以被捕获、处理、重新抛出
- ✅ 经常用于处理**可恢复的错误**
- ✅ 例如：文件不存在、网络超时、解析失败

```java
// Java 示例
try {
    File file = new File("config.txt");
    String data = readFile(file);
} catch (FileNotFoundException e) {
    // 处理文件不存在 - 这是预期的情况
    System.out.println("文件不存在");
}
```

#### Rust 的 panic!（恐慌机制）

- ⚠️ panic 是**程序遇到了不应该发生的情况**
- ⚠️ 表示程序进入了**不可恢复的状态**
- ⚠️ 默认行为是**展开栈并终止程序**
- ⚠️ 用于表示"这不应该发生，程序有 bug"

```rust
// Rust 示例
fn get_item(index: usize, items: &[i32]) -> i32 {
    if index >= items.len() {
        panic!("索引越界！这是程序 bug！");  // 不应该发生
    }
    items[index]
}
```

---

### 2. 使用场景对比

#### ❌ 其他语言可能这样写（使用异常）

```python
# Python 示例
try:
    file = open("config.txt")
    data = parse(file.read())
except FileNotFoundError:
    # 处理文件不存在 - 这是预期的情况
    print("配置文件不存在")
except ParseError:
    # 处理解析错误 - 也是预期的情况
    print("配置文件格式错误")
```

#### ✅ Rust 的方式

```rust
// 可恢复的错误 → 使用 Result
use std::fs::File;
use std::io::Read;

fn read_config() -> Result<String, std::io::Error> {
    let mut file = File::open("config.txt")?;  // ? 操作符传播错误
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// 调用时处理错误
match read_config() {
    Ok(config) => println!("配置: {}", config),
    Err(e) => println!("读取失败: {}", e),
}
```

```rust
// panic! 只用于真正的程序错误
fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("除以零！这是程序逻辑错误！");
    }
    a / b
}

// 更好的方式：返回 Result
fn divide_safe(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("除数不能为零".to_string())
    } else {
        Ok(a / b)
    }
}
```

---

### 3. 错误处理模型不同

#### 异常模型（Java/Python/C++）

```java
// Java - 异常可以被捕获和处理
try {
    riskyOperation();
    anotherRiskyOperation();
} catch (IOException e) {
    // 捕获并处理 IO 异常
    handleIOError(e);
} catch (Exception e) {
    // 捕获所有其他异常
    handleGenericError(e);
} finally {
    // 清理资源
    cleanup();
}
```

#### Rust 的双轨模型

```rust
// 可恢复错误 → Result<T, E>
fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)  // 返回 Result
}

// 使用 ? 操作符优雅地传播错误
fn process_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = read_file(path)?;  // 如果失败，自动返回 Err
    let parsed = parse_content(&content)?;
    save_result(parsed)?;
    Ok(())
}

// 不可恢复错误 → panic!
fn assert_positive(n: i32) {
    if n <= 0 {
        panic!("数字必须为正数！调用者违反了约定！");
    }
}
```

---

### 4. 为什么叫 "panic"？

这个名字传达了几个重要信息：

#### 📢 紧急性
"panic"（恐慌）表示情况很严重，不是普通的错误。

#### 🚫 不可恢复
程序进入了混乱状态，需要立即停止。

#### 🐛 程序员错误
不是用户错误或环境问题，而是代码本身有 bug。

#### 🧠 心理暗示
让你三思："我真的需要 panic 吗？还是应该用 Result？"

```rust
// ❌ 不好的做法 - 滥用 panic
fn get_user_age(input: &str) -> u8 {
    input.parse().unwrap()  // 用户输入错误就 panic，太粗暴！
}

// ✅ 好的做法 - 使用 Result
fn get_user_age(input: &str) -> Result<u8, std::num::ParseIntError> {
    input.parse()  // 让调用者决定如何处理错误
}

// ✅ 或者提供默认值
fn get_user_age_or_default(input: &str) -> u8 {
    input.parse().unwrap_or(18)  // 解析失败返回默认值
}
```

---

## 📊 对比总结

| 特性 | panic! (Rust) | throw (其他语言) |
|------|---------------|------------------|
| **用途** | 不可恢复的程序错误 | 可恢复的异常情况 |
| **能否捕获** | 可以但不推荐（catch_unwind） | 设计为可捕获（try/catch） |
| **性能** | 展开栈有开销 | 异常处理有开销 |
| **使用频率** | 很少使用 | 经常使用 |
| **典型场景** | 数组越界、断言失败、不变量被破坏 | 文件不存在、网络错误、用户输入错误 |
| **控制流** | 终止程序（或线程） | 跳转到 catch 块 |
| **编译器检查** | 无需声明 | 某些语言需要声明（checked exception） |

---

## 💡 Rust 的错误处理哲学

Rust 强制你区分两种错误：

### 1. 预期的、可恢复的错误 → 使用 Result

```rust
use std::num::ParseIntError;

// ✅ 用户输入可能无效，这是正常的
fn parse_age(input: &str) -> Result<u8, ParseIntError> {
    input.parse()
}

// 调用时优雅地处理
fn main() {
    match parse_age("25") {
        Ok(age) => println!("年龄: {}", age),
        Err(e) => println!("解析失败: {}", e),
    }
    
    // 或者使用 ? 操作符
    let age = parse_age("25").unwrap_or(0);
}
```

### 2. 不应该发生的错误 → 使用 panic!

```rust
// ❌ 调用者应该先检查，如果到这里说明有 bug
fn get_first_element(vec: &Vec<i32>) -> i32 {
    if vec.is_empty() {
        panic!("向量为空！调用者应该先检查！");
    }
    vec[0]
}

// ✅ 更好的方式 → 返回 Option，让调用者决定
fn get_first_element_safe(vec: &Vec<i32>) -> Option<i32> {
    vec.first().copied()
}

// 使用示例
fn main() {
    let numbers = vec![1, 2, 3];
    
    // 方式 1: 使用 Option
    if let Some(first) = get_first_element_safe(&numbers) {
        println!("第一个元素: {}", first);
    }
    
    // 方式 2: 提供默认值
    let first = get_first_element_safe(&numbers).unwrap_or(0);
}
```

---

## 🎯 实际使用建议

### 1. 99% 的情况使用 `Result`

```rust
use std::fs::File;
use std::io::{self, Read};

// ✅ 文件操作
fn read_config(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// ✅ 网络请求
async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    let response = reqwest::get(url).await?;
    response.text().await
}

// ✅ 解析数据
fn parse_json(data: &str) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(data)
}
```

### 2. 很少使用 `panic!`

```rust
// ✅ 合理使用 panic! 的场景

// 场景 1: 断言不变量
fn set_percentage(value: u8) {
    assert!(value <= 100, "百分比不能超过 100");
    // ...
}

// 场景 2: 不可能的情况
fn process_enum(value: MyEnum) {
    match value {
        MyEnum::A => { /* 处理 A */ },
        MyEnum::B => { /* 处理 B */ },
        _ => unreachable!("MyEnum 只有 A 和 B 两个变体"),
    }
}

// 场景 3: 开发时的快速失败
fn main() {
    let config = std::fs::read_to_string("config.toml")
        .expect("配置文件必须存在");  // 开发时可以这样
}
```

### 3. 开发时可以用 `unwrap`/`expect`

```rust
// 开发阶段 - 快速原型
fn prototype() {
    let file = File::open("test.txt").unwrap();  // 快速测试
    let data: i32 = "42".parse().expect("应该是数字");
}

// 生产代码 - 正确处理
fn production() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("test.txt")?;  // 传播错误
    let data: i32 = "42".parse()?;
    Ok(())
}
```

### 4. 生产代码避免 `panic!`

```rust
// ❌ 不好 - 生产代码中 panic
fn process_request(data: &str) -> String {
    let parsed: i32 = data.parse().unwrap();  // 用户输入错误就崩溃！
    format!("结果: {}", parsed * 2)
}

// ✅ 好 - 返回 Result
fn process_request(data: &str) -> Result<String, String> {
    let parsed: i32 = data.parse()
        .map_err(|e| format!("解析失败: {}", e))?;
    Ok(format!("结果: {}", parsed * 2))
}
```

---

## 🔧 panic! 的高级用法

### 1. 自定义 panic 处理器

```rust
use std::panic;

fn main() {
    // 设置自定义 panic 钩子
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("程序崩溃了！");
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("panic 消息: {}", s);
        }
        if let Some(location) = panic_info.location() {
            eprintln!("位置: {}:{}", location.file(), location.line());
        }
    }));

    panic!("测试 panic");
}
```

### 2. 捕获 panic（不推荐）

```rust
use std::panic;

fn main() {
    let result = panic::catch_unwind(|| {
        panic!("出错了！");
    });

    match result {
        Ok(_) => println!("没有 panic"),
        Err(_) => println!("捕获到 panic"),
    }
}
```

### 3. 断言宏家族

```rust
fn test_assertions() {
    // assert! - 条件为 false 时 panic
    assert!(2 + 2 == 4);
    
    // assert_eq! - 两个值不相等时 panic
    assert_eq!(2 + 2, 4);
    
    // assert_ne! - 两个值相等时 panic
    assert_ne!(2 + 2, 5);
    
    // debug_assert! - 只在 debug 模式下检查
    debug_assert!(expensive_check());
    
    // unreachable! - 标记不应该到达的代码
    match some_value {
        1 => {},
        2 => {},
        _ => unreachable!("只可能是 1 或 2"),
    }
    
    // unimplemented! - 标记未实现的功能
    fn todo_function() {
        unimplemented!("稍后实现");
    }
}
```

---

## 📝 总结

### panic! 这个名字很贴切

它告诉你："这是紧急情况，程序要停止了！" 而不是像 `throw` 那样暗示"这只是个可以处理的异常"。

### Rust 的错误处理优势

1. **类型系统强制处理错误** - `Result` 必须被处理
2. **区分可恢复和不可恢复** - 清晰的错误分类
3. **零成本抽象** - `Result` 和 `?` 操作符性能优秀
4. **编译时保证** - 不会忘记处理错误

### 记住这个原则

> **如果错误是预期的、可恢复的 → 使用 `Result`**  
> **如果错误是程序 bug、不可恢复的 → 使用 `panic!`**

这就是为什么 Rust 选择了 `panic!` 这个名字，而不是 `throw` - 它们代表了完全不同的错误处理哲学！

