# Unsafe Rust 详解

## 📚 什么是 Unsafe Rust？

**Unsafe Rust** 是 Rust 中的一个特殊子集,允许你绕过编译器的某些安全检查。虽然名字听起来很危险,但它是 Rust 实现底层编程和系统编程的关键。

### 为什么需要 Unsafe？

#### 1️⃣ 编译器太保守

Rust 的编译器非常保守,有时会拒绝一些实际上是安全的代码,因为它无法证明这些代码的安全性。

```rust
// 编译器可能无法理解这段代码是安全的
// 但我们知道它是安全的
fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    // 编译器会报错:不能同时有两个可变引用
    // (&mut slice[..mid], &mut slice[mid..]) // ❌ 编译错误
}
```

#### 2️⃣ 底层编程的需要

- **系统编程** - 直接操作硬件
- **操作系统开发** - 需要访问底层资源
- **性能优化** - 某些优化需要绕过安全检查
- **FFI** - 与 C/C++ 等语言交互

---

## 🔥 Unsafe 的五种超能力

使用 `unsafe` 关键字可以获得以下 5 种能力:

### 1️⃣ 解引用裸指针

```rust
fn main() {
    let mut num = 5;
    
    // 创建裸指针(安全)
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    // 解引用裸指针(不安全)
    unsafe {
        println!("r1 is: {}", *r1);
        *r2 = 10;
        println!("r2 is: {}", *r2);
    }
}
```

### 2️⃣ 调用 unsafe 函数或方法

```rust
unsafe fn dangerous() {
    println!("This is dangerous!");
}

fn main() {
    unsafe {
        dangerous();
    }
}
```

### 3️⃣ 访问或修改可变静态变量

```rust
static mut COUNTER: u32 = 0;

fn add_to_count(inc: u32) {
    unsafe {
        COUNTER += inc;
    }
}

fn main() {
    add_to_count(3);
    
    unsafe {
        println!("COUNTER: {}", COUNTER);
    }
}
```

### 4️⃣ 实现 unsafe trait

```rust
unsafe trait Foo {
    // 方法列表
}

unsafe impl Foo for i32 {
    // 实现方法
}
```

### 5️⃣ 访问 union 的字段

```rust
#[repr(C)]
union MyUnion {
    f1: u32,
    f2: f32,
}

fn main() {
    let u = MyUnion { f1: 1 };
    
    unsafe {
        println!("f1: {}", u.f1);
    }
}
```

---

## 🎯 1. 裸指针详解

### 什么是裸指针？

裸指针类似于 C 语言中的指针,有两种类型:
- `*const T` - 不可变裸指针
- `*mut T` - 可变裸指针

### 裸指针 vs 引用

| 特性 | 引用 | 裸指针 |
|------|------|--------|
| **借用检查** | 受检查 | 不受检查 |
| **可空性** | 不可为 null | 可以为 null |
| **安全性** | 保证安全 | 不保证安全 |
| **多个可变** | 不允许 | 允许 |
| **自动回收** | 有 | 无 |

### 创建裸指针的三种方式

#### 方式 1: 从引用创建(安全)

```rust
fn main() {
    let mut num = 5;
    
    // 从引用创建裸指针是安全的
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    // 也可以隐式转换
    let r3: *const i32 = &num;
}
```

#### 方式 2: 从内存地址创建(危险)

```rust
fn main() {
    // ⚠️ 非常危险!可能导致段错误
    let address = 0x012345usize;
    let r = address as *const i32;
    
    // 不要这样做!
    // unsafe {
    //     println!("{}", *r); // 💥 段错误
    // }
}
```

#### 方式 3: 从智能指针创建

```rust
fn main() {
    let a: Box<i32> = Box::new(10);
    
    // 需要先解引用
    let b: *const i32 = &*a;
    
    // 使用 into_raw
    let c: *const i32 = Box::into_raw(a);
}
```

### 裸指针的实战示例

#### 示例 1: 实现 split_at_mut

```rust
use std::slice;

fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    
    assert!(mid <= len);
    
    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    let r = &mut v[..];
    
    let (a, b) = split_at_mut(r, 3);
    
    assert_eq!(a, &mut [1, 2, 3]);
    assert_eq!(b, &mut [4, 5, 6]);
}
```

#### 示例 2: 读取字符串内存

```rust
use std::{slice::from_raw_parts, str::from_utf8_unchecked};

fn get_memory_location() -> (usize, usize) {
    let string = "Hello World!";
    let pointer = string.as_ptr() as usize;
    let length = string.len();
    (pointer, length)
}

fn get_str_at_location(pointer: usize, length: usize) -> &'static str {
    unsafe {
        from_utf8_unchecked(from_raw_parts(pointer as *const u8, length))
    }
}

fn main() {
    let (pointer, length) = get_memory_location();
    let message = get_str_at_location(pointer, length);
    println!("The {} bytes at 0x{:X} stored: {}", length, pointer, message);
}
```

---

## 🔗 2. FFI (外部函数接口)

### 什么是 FFI？

**FFI** (Foreign Function Interface) 允许 Rust 代码调用其他语言(主要是 C)编写的函数。

### 调用 C 函数

#### 示例 1: 调用 C 标准库

```rust
use libc::size_t;

#[link(name = "snappy")]
extern "C" {
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
}

fn main() {
    let x = unsafe {
        snappy_max_compressed_length(100)
    };
    println!("max compressed length: {}", x);
}
```

#### 示例 2: 完整的 C 库绑定

```rust
use libc::{c_int, size_t};

#[link(name = "snappy")]
extern "C" {
    fn snappy_compress(
        input: *const u8,
        input_length: size_t,
        compressed: *mut u8,
        compressed_length: *mut size_t
    ) -> c_int;
    
    fn snappy_uncompress(
        compressed: *const u8,
        compressed_length: size_t,
        uncompressed: *mut u8,
        uncompressed_length: *mut size_t
    ) -> c_int;
}

pub fn compress(src: &[u8]) -> Vec<u8> {
    unsafe {
        let srclen = src.len() as size_t;
        let psrc = src.as_ptr();
        
        let mut dstlen = srclen * 2; // 简化示例
        let mut dst = Vec::with_capacity(dstlen as usize);
        let pdst = dst.as_mut_ptr();
        
        snappy_compress(psrc, srclen, pdst, &mut dstlen);
        dst.set_len(dstlen as usize);
        dst
    }
}
```

### ABI (应用二进制接口)

```rust
// C ABI (最常见)
extern "C" {
    fn abs(input: i32) -> i32;
}

// Windows API
#[cfg(all(target_os = "win32", target_arch = "x86"))]
extern "stdcall" {
    fn SetEnvironmentVariableA(n: *const u8, v: *const u8) -> i32;
}

// System ABI (根据平台自动选择)
extern "system" {
    // ...
}
```

### 从其他语言调用 Rust

```rust
#[no_mangle]
pub extern "C" fn call_from_c() {
    println!("Just called a Rust function from C!");
}
```

**C 代码**:
```c
// 声明
extern void call_from_c();

int main() {
    call_from_c();
    return 0;
}
```

### 回调函数

```rust
// Rust 代码
extern "C" fn callback(a: i32) {
    println!("I'm called from C with value {}", a);
}

#[link(name = "extlib")]
extern "C" {
    fn register_callback(cb: extern "C" fn(i32)) -> i32;
    fn trigger_callback();
}

fn main() {
    unsafe {
        register_callback(callback);
        trigger_callback();
    }
}
```

**C 代码**:
```c
typedef void (*rust_callback)(int32_t);
rust_callback cb;

int32_t register_callback(rust_callback callback) {
    cb = callback;
    return 1;
}

void trigger_callback() {
    cb(7);
}
```

---

## ⚙️ 3. 内联汇编

### 什么是内联汇编？

内联汇编允许你在 Rust 代码中直接嵌入汇编指令,用于:
- **极致性能优化**
- **访问特殊 CPU 指令**
- **底层硬件操作**

### 基本语法

```rust
use std::arch::asm;

fn main() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        asm!("nop"); // 插入一个 NOP 指令
    }
}
```

### 输入和输出

#### 示例 1: 输出

```rust
#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::asm;
    
    let x: u64;
    unsafe {
        asm!("mov {}, 5", out(reg) x);
    }
    assert_eq!(x, 5);
}
```

#### 示例 2: 输入和输出

```rust
#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::asm;
    
    let i: u64 = 3;
    let o: u64;
    unsafe {
        asm!(
            "mov {0}, {1}",
            "add {0}, 5",
            out(reg) o,
            in(reg) i,
        );
    }
    assert_eq!(o, 8);
}
```

#### 示例 3: inout (输入输出同一个变量)

```rust
#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::asm;
    
    let mut x: u64 = 3;
    unsafe {
        asm!("add {0}, 5", inout(reg) x);
    }
    assert_eq!(x, 8);
}
```

### 指定寄存器

```rust
#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::asm;
    
    let cmd = 0xd1;
    unsafe {
        asm!("out 0x64, eax", in("eax") cmd);
    }
}
```

### 实战示例: 乘法

```rust
#[cfg(target_arch = "x86_64")]
fn mul(a: u64, b: u64) -> u128 {
    use std::arch::asm;
    
    let lo: u64;
    let hi: u64;
    
    unsafe {
        asm!(
            "mul {}",
            in(reg) a,
            inlateout("rax") b => lo,
            lateout("rdx") hi
        );
    }
    
    ((hi as u128) << 64) + lo as u128
}
```

### 选项

```rust
#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::asm;
    
    let mut a: u64 = 4;
    let b: u64 = 4;
    unsafe {
        asm!(
            "add {0}, {1}",
            inlateout(reg) a,
            in(reg) b,
            options(pure, nomem, nostack), // 优化选项
        );
    }
    assert_eq!(a, 8);
}
```

**常用选项**:
- `pure` - 无副作用
- `nomem` - 不访问内存
- `readonly` - 只读内存
- `nostack` - 不使用栈
- `noreturn` - 不返回

---

## 💡 使用 Unsafe 的最佳时机

### ✅ 应该使用 Unsafe 的场景

#### 1. 性能关键路径

```rust
// 避免边界检查
fn sum_unchecked(slice: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..slice.len() {
        unsafe {
            sum += *slice.get_unchecked(i);
        }
    }
    sum
}
```

#### 2. 实现底层数据结构

```rust
pub struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Vec<T> {
    pub fn push(&mut self, elem: T) {
        if self.len == self.cap {
            self.grow();
        }
        unsafe {
            std::ptr::write(self.ptr.add(self.len), elem);
        }
        self.len += 1;
    }
}
```

#### 3. FFI 调用

```rust
extern "C" {
    fn some_c_function(x: i32) -> i32;
}

fn call_c() -> i32 {
    unsafe {
        some_c_function(42)
    }
}
```

#### 4. 内存映射 I/O

```rust
fn read_hardware_register(addr: usize) -> u32 {
    unsafe {
        std::ptr::read_volatile(addr as *const u32)
    }
}
```

### ❌ 不应该使用 Unsafe 的场景

#### 1. 简单的类型转换

```rust
// ❌ 不要用 unsafe
let x: i32 = unsafe { std::mem::transmute(5u32) };

// ✅ 使用安全的方法
let x: i32 = 5u32 as i32;
```

#### 2. 可以用安全代码实现的功能

```rust
// ❌ 不要用 unsafe
unsafe fn add(a: i32, b: i32) -> i32 {
    a + b
}

// ✅ 使用普通函数
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

---

## 🛡️ Unsafe 的安全保证

### Unsafe 不能绕过的检查

1. **借用检查** - 仍然有效
2. **生命周期检查** - 仍然有效
3. **类型检查** - 仍然有效

```rust
fn main() {
    let r;
    {
        let x = 5;
        r = &x;
    }
    // ❌ 即使在 unsafe 中也会报错
    // unsafe {
    //     println!("{}", r);
    // }
}
```

### 控制 Unsafe 边界

```rust
// ✅ 好的做法:小范围的 unsafe
pub fn safe_function(slice: &[i32], index: usize) -> Option<i32> {
    if index < slice.len() {
        Some(unsafe { *slice.get_unchecked(index) })
    } else {
        None
    }
}

// ❌ 不好的做法:大范围的 unsafe
pub unsafe fn unsafe_function(slice: &[i32], index: usize) -> i32 {
    // 很多代码...
    *slice.get_unchecked(index)
    // 更多代码...
}
```

---

## 📊 总结

### 核心要点

| 概念 | 说明 |
|------|------|
| **Unsafe 关键字** | 告诉编译器"我知道我在做什么" |
| **五种超能力** | 裸指针、unsafe 函数、静态变量、unsafe trait、union |
| **使用原则** | 能不用就不用,必须用时尽量小范围 |
| **安全保证** | 仍然有借用检查、生命周期检查 |

### 记忆口诀

- **Unsafe** = **绕过部分安全检查**
- **裸指针** = **类似 C 指针**
- **FFI** = **与 C 交互**
- **内联汇编** = **直接写汇编**
- **使用时机** = **底层编程、性能优化、FFI**

### 最佳实践

1. ✅ **最小化 unsafe 范围**
2. ✅ **用安全 API 包装 unsafe 代码**
3. ✅ **详细注释为什么需要 unsafe**
4. ✅ **充分测试 unsafe 代码**
5. ❌ **不要滥用 unsafe**

🦀✨

