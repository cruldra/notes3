# Go 语言常用语法指南

> 基于 KrillinAI 项目实战总结

## 目录

- [基础语法](#基础语法)
- [数据类型](#数据类型)
- [结构体与方法](#结构体与方法)
- [接口](#接口)
- [并发编程](#并发编程)
- [错误处理](#错误处理)
- [包管理](#包管理)
- [常用标准库](#常用标准库)
- [实战技巧](#实战技巧)

## 基础语法

### 1. 包声明和导入

```go
// 包声明（每个 Go 文件必须以 package 开头）
package main

// 导入单个包
import "fmt"

// 导入多个包（推荐）
import (
    "fmt"
    "os"
    "path/filepath"
    
    // 第三方包
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
    
    // 项目内部包
    "krillin-ai/config"
    "krillin-ai/log"
)

// 包别名
import (
    openai "github.com/sashabaranov/go-openai"
)
```

### 2. 变量声明

```go
// 方式1：var 声明（可指定类型）
var name string = "KrillinAI"
var age int = 1

// 方式2：var 声明（类型推断）
var name = "KrillinAI"
var age = 1

// 方式3：短变量声明（只能在函数内使用）
name := "KrillinAI"
age := 1

// 方式4：批量声明
var (
    host string = "127.0.0.1"
    port int    = 8888
)

// 零值（未初始化的变量会有默认值）
var s string    // ""
var i int       // 0
var b bool      // false
var p *int      // nil
```

### 3. 常量

```go
// 单个常量
const MaxSize = 100

// 批量常量
const (
    StatusPending   = "pending"
    StatusProcessing = "processing"
    StatusCompleted = "completed"
)

// iota 枚举器（自动递增）
const (
    Sunday = iota    // 0
    Monday           // 1
    Tuesday          // 2
    Wednesday        // 3
)
```

### 4. 函数

```go
// 基本函数
func add(a int, b int) int {
    return a + b
}

// 参数类型相同可简写
func add(a, b int) int {
    return a + b
}

// 多返回值
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("除数不能为零")
    }
    return a / b, nil
}

// 命名返回值
func getConfig() (host string, port int) {
    host = "127.0.0.1"
    port = 8888
    return // 自动返回 host 和 port
}

// 可变参数
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// 使用：sum(1, 2, 3, 4, 5)
```

## 数据类型

### 1. 基本类型

```go
// 整数类型
var i8 int8      // -128 到 127
var i16 int16    // -32768 到 32767
var i32 int32    // -2147483648 到 2147483647
var i64 int64    // -9223372036854775808 到 9223372036854775807
var ui uint      // 无符号整数

// 浮点数
var f32 float32  // 单精度
var f64 float64  // 双精度

// 字符串
var s string = "Hello, 世界"

// 布尔值
var b bool = true

// 字节和字符
var bt byte = 'A'    // byte 是 uint8 的别名
var r rune = '中'     // rune 是 int32 的别名，表示 Unicode 码点
```

### 2. 复合类型

#### 数组

```go
// 固定长度数组
var arr [5]int = [5]int{1, 2, 3, 4, 5}

// 自动推断长度
arr := [...]int{1, 2, 3, 4, 5}

// 访问元素
fmt.Println(arr[0])  // 1
```

#### 切片（Slice）

```go
// 创建切片
slice := []int{1, 2, 3, 4, 5}

// 使用 make 创建
slice := make([]int, 5)      // 长度为 5
slice := make([]int, 5, 10)  // 长度为 5，容量为 10

// 切片操作
slice = append(slice, 6)     // 追加元素
slice = slice[1:3]           // 切片 [2, 3]
len(slice)                   // 长度
cap(slice)                   // 容量

// 项目示例：存储字幕分段
segments := make([]types.Segment, 0)
segments = append(segments, types.Segment{
    Text:  "Hello",
    Start: 0.0,
    End:   1.5,
})
```

#### 映射（Map）

```go
// 创建 map
m := make(map[string]int)
m["age"] = 25

// 字面量创建
m := map[string]int{
    "age":  25,
    "year": 2024,
}

// 检查键是否存在
value, exists := m["age"]
if exists {
    fmt.Println(value)
}

// 删除键
delete(m, "age")

// 遍历
for key, value := range m {
    fmt.Println(key, value)
}

// 项目示例：文字替换映射
replaceWordsMap := make(map[string]string)
replaceWordsMap["AI"] = "人工智能"
replaceWordsMap["ML"] = "机器学习"
```

### 3. 指针

```go
// 声明指针
var p *int

// 取地址
x := 10
p = &x

// 解引用
fmt.Println(*p)  // 10

// 修改指针指向的值
*p = 20
fmt.Println(x)   // 20

// 项目示例：任务指针
taskPtr := &types.SubtitleTask{
    TaskId: "task123",
    Status: types.SubtitleTaskStatusProcessing,
}
```

## 结构体与方法

### 1. 结构体定义

```go
// 基本结构体
type Person struct {
    Name string
    Age  int
}

// 创建实例
p1 := Person{Name: "Alice", Age: 25}
p2 := Person{"Bob", 30}  // 按顺序赋值

// 匿名字段
type Student struct {
    Person  // 嵌入 Person
    Grade int
}

// 项目示例：配置结构体
type Config struct {
    App        App                    `toml:"app"`
    Server     Server                 `toml:"server"`
    Llm        OpenaiCompatibleConfig `toml:"llm"`
    Transcribe Transcribe             `toml:"transcribe"`
    Tts        Tts                    `toml:"tts"`
}

type Server struct {
    Host string `toml:"host"`
    Port int    `toml:"port"`
}
```

### 2. 方法

```go
// 值接收者方法
func (p Person) SayHello() {
    fmt.Printf("Hello, I'm %s\n", p.Name)
}

// 指针接收者方法（可以修改结构体）
func (p *Person) SetAge(age int) {
    p.Age = age
}

// 使用
p := Person{Name: "Alice", Age: 25}
p.SayHello()
p.SetAge(26)

// 项目示例：Client 方法
type Client struct {
    apiKey string
}

func (c *Client) ChatCompletion(query string) (string, error) {
    // 实现聊天补全逻辑
    return "response", nil
}
```

### 3. 构造函数模式

```go
// New 函数（Go 的构造函数惯例）
func NewPerson(name string, age int) *Person {
    return &Person{
        Name: name,
        Age:  age,
    }
}

// 项目示例
func NewService() *Service {
    return &Service{
        Transcriber:   whisper.NewClient(),
        ChatCompleter: openai.NewClient(),
        TtsClient:     aliyun.NewTtsClient(),
    }
}
```

## 接口

### 1. 接口定义

```go
// 接口声明
type Writer interface {
    Write([]byte) (int, error)
}

// 接口实现（隐式实现，无需声明）
type FileWriter struct {
    filename string
}

func (f *FileWriter) Write(data []byte) (int, error) {
    // 实现写入逻辑
    return len(data), nil
}

// 项目示例：核心接口
type ChatCompleter interface {
    ChatCompletion(query string) (string, error)
}

type Transcriber interface {
    Transcription(audioFile, language, wordDir string) (*TranscriptionData, error)
}

type Ttser interface {
    Text2Speech(text string, voice string, outputFile string) error
}
```

### 2. 接口组合

```go
// 组合多个接口
type ReadWriter interface {
    Reader
    Writer
}

// 空接口（可以接受任何类型）
var any interface{}
any = 42
any = "hello"
any = []int{1, 2, 3}
```

### 3. 类型断言

```go
// 类型断言
var i interface{} = "hello"

// 安全断言
s, ok := i.(string)
if ok {
    fmt.Println(s)
}

// 类型判断
switch v := i.(type) {
case string:
    fmt.Println("字符串:", v)
case int:
    fmt.Println("整数:", v)
default:
    fmt.Println("未知类型")
}
```

## 并发编程

### 1. Goroutine

```go
// 启动 goroutine
go func() {
    fmt.Println("在 goroutine 中执行")
}()

// 带参数的 goroutine
go func(msg string) {
    fmt.Println(msg)
}("Hello from goroutine")

// 项目示例：并发处理字幕
for i, sub := range subtitles {
    go func(index int, subtitle types.SrtSentence) {
        // 处理字幕
        outputFile := fmt.Sprintf("subtitle_%d.wav", index+1)
        err := s.TtsClient.Text2Speech(subtitle.Text, voiceCode, outputFile)
        if err != nil {
            log.GetLogger().Error("TTS error", zap.Error(err))
        }
    }(i, sub)
}
```

### 2. Channel

```go
// 创建 channel
ch := make(chan int)
ch := make(chan int, 10)  // 带缓冲的 channel

// 发送数据
ch <- 42

// 接收数据
value := <-ch

// 关闭 channel
close(ch)

// 遍历 channel
for value := range ch {
    fmt.Println(value)
}

// 项目示例：任务队列
pendingQueue := make(chan DataWithId[string], segmentNum)
resultQueue := make(chan DataWithId[string], segmentNum)

// 发送任务
pendingQueue <- DataWithId[string]{
    Id:   1,
    Data: "audio_file.wav",
}

// 接收结果
result := <-resultQueue
```

### 3. Select

```go
// select 多路复用
select {
case msg1 := <-ch1:
    fmt.Println("收到 ch1:", msg1)
case msg2 := <-ch2:
    fmt.Println("收到 ch2:", msg2)
case <-time.After(time.Second):
    fmt.Println("超时")
default:
    fmt.Println("没有数据")
}

// 项目示例：处理多个队列
select {
case <-ctx.Done():
    return nil
case splitItem, ok := <-pendingSplitQueue:
    if !ok {
        return nil
    }
    // 处理分割任务
case audioItem, ok := <-pendingTranscriptionQueue:
    if !ok {
        return nil
    }
    // 处理转录任务
}
```

### 4. WaitGroup

```go
import "sync"

var wg sync.WaitGroup

// 添加计数
wg.Add(3)

// 启动 goroutine
go func() {
    defer wg.Done()  // 完成时减少计数
    // 执行任务
}()

// 等待所有 goroutine 完成
wg.Wait()

// 项目示例
var wg sync.WaitGroup
for i, sub := range subtitles {
    wg.Add(1)
    go func(index int, subtitle types.SrtSentence) {
        defer wg.Done()
        // 处理字幕
    }(i, sub)
}
wg.Wait()
```

### 5. Mutex（互斥锁）

```go
import "sync"

var (
    counter int
    mu      sync.Mutex
)

// 加锁
mu.Lock()
counter++
mu.Unlock()

// 读写锁
var rwMu sync.RWMutex

// 读锁（多个 goroutine 可以同时读）
rwMu.RLock()
value := counter
rwMu.RUnlock()

// 写锁（独占）
rwMu.Lock()
counter++
rwMu.Unlock()
```

### 6. sync.Map（并发安全的 Map）

```go
import "sync"

var m sync.Map

// 存储
m.Store("key", "value")

// 读取
value, ok := m.Load("key")

// 删除
m.Delete("key")

// 遍历
m.Range(func(key, value interface{}) bool {
    fmt.Println(key, value)
    return true  // 返回 false 停止遍历
})

// 项目示例：存储任务
var SubtitleTasks = sync.Map{}
SubtitleTasks.Store(taskId, taskPtr)
```

### 7. errgroup（错误组）

```go
import (
    "context"
    "golang.org/x/sync/errgroup"
)

eg, ctx := errgroup.WithContext(context.Background())

// 启动多个任务
eg.Go(func() error {
    // 任务1
    return nil
})

eg.Go(func() error {
    // 任务2
    return nil
})

// 等待所有任务完成，任何一个返回错误都会取消其他任务
if err := eg.Wait(); err != nil {
    log.Fatal(err)
}

// 项目示例：并发处理音频分段
eg, ctx := errgroup.WithContext(context.Background())

for range runtime.NumCPU() {
    eg.Go(func() error {
        for {
            select {
            case <-ctx.Done():
                return nil
            case item, ok := <-queue:
                if !ok {
                    return nil
                }
                // 处理任务
            }
        }
    })
}

if err := eg.Wait(); err != nil {
    return err
}
```

## 错误处理

### 1. 基本错误处理

```go
import "errors"

// 创建错误
err := errors.New("发生错误")

// 格式化错误
err := fmt.Errorf("文件 %s 不存在", filename)

// 检查错误
if err != nil {
    log.Fatal(err)
}

// 返回错误
func readFile(filename string) ([]byte, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, fmt.Errorf("读取文件失败: %w", err)
    }
    return data, nil
}
```

### 2. 错误包装

```go
// 包装错误（Go 1.13+）
err := fmt.Errorf("操作失败: %w", originalErr)

// 解包错误
errors.Unwrap(err)

// 错误判断
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("文件不存在")
}

// 类型断言
var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Println("路径错误:", pathErr.Path)
}
```

### 3. 自定义错误

```go
// 自定义错误类型
type ValidationError struct {
    Field string
    Msg   string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("字段 %s 验证失败: %s", e.Field, e.Msg)
}

// 使用
func validate(age int) error {
    if age < 0 {
        return &ValidationError{
            Field: "age",
            Msg:   "年龄不能为负数",
        }
    }
    return nil
}
```

## 包管理

### 1. 模块初始化

```powershell
# 初始化模块
go mod init krillin-ai

# 添加依赖
go get github.com/gin-gonic/gin@v1.10.0

# 更新依赖
go get -u github.com/gin-gonic/gin

# 整理依赖
go mod tidy

# 下载依赖
go mod download
```

### 2. 导入路径

```go
// 标准库
import "fmt"

// 第三方库
import "github.com/gin-gonic/gin"

// 项目内部包
import "krillin-ai/config"
import "krillin-ai/internal/service"
```

## 常用标准库

### 1. fmt - 格式化输入输出

```go
import "fmt"

// 打印
fmt.Println("Hello")
fmt.Printf("Name: %s, Age: %d\n", name, age)

// 格式化字符串
s := fmt.Sprintf("Name: %s", name)

// 格式化占位符
%v    // 默认格式
%+v   // 包含字段名
%#v   // Go 语法表示
%T    // 类型
%t    // 布尔值
%d    // 十进制整数
%f    // 浮点数
%s    // 字符串
%p    // 指针
```

### 2. strings - 字符串操作

```go
import "strings"

// 常用函数
strings.Contains(s, substr)      // 包含
strings.HasPrefix(s, prefix)     // 前缀
strings.HasSuffix(s, suffix)     // 后缀
strings.Split(s, sep)            // 分割
strings.Join(slice, sep)         // 连接
strings.Replace(s, old, new, n)  // 替换
strings.ToLower(s)               // 转小写
strings.ToUpper(s)               // 转大写
strings.TrimSpace(s)             // 去除首尾空格
```

### 3. os - 操作系统功能

```go
import "os"

// 文件操作
os.ReadFile(filename)
os.WriteFile(filename, data, 0644)
os.Create(filename)
os.Open(filename)
os.Remove(filename)

// 目录操作
os.Mkdir(dirname, 0755)
os.MkdirAll(dirname, 0755)
os.RemoveAll(dirname)

// 环境变量
os.Getenv("PATH")
os.Setenv("KEY", "value")

// 进程
os.Exit(1)
os.Getwd()  // 当前目录
```

### 4. filepath - 路径操作

```go
import "path/filepath"

// 路径操作
filepath.Join("dir", "file.txt")     // 拼接路径
filepath.Ext(filename)                // 获取扩展名
filepath.Base(path)                   // 获取文件名
filepath.Dir(path)                    // 获取目录
filepath.Abs(path)                    // 绝对路径

// 项目示例
taskBasePath := filepath.Join("./tasks", taskId)
outputFile := filepath.Join(taskBasePath, "output", "video.mp4")
```

### 5. time - 时间操作

```go
import "time"

// 当前时间
now := time.Now()

// 格式化时间（Go 的特殊格式：2006-01-02 15:04:05）
now.Format("2006-01-02 15:04:05")

// 解析时间
t, _ := time.Parse("2006-01-02", "2024-01-01")

// 时间运算
future := now.Add(24 * time.Hour)
duration := future.Sub(now)

// 睡眠
time.Sleep(time.Second)

// 定时器
timer := time.NewTimer(time.Second)
<-timer.C

// 周期执行
ticker := time.NewTicker(time.Second)
for range ticker.C {
    // 每秒执行
}
```

### 6. encoding/json - JSON 处理

```go
import "encoding/json"

// 结构体转 JSON
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

p := Person{Name: "Alice", Age: 25}
data, _ := json.Marshal(p)
fmt.Println(string(data))  // {"name":"Alice","age":25}

// JSON 转结构体
jsonStr := `{"name":"Bob","age":30}`
var p2 Person
json.Unmarshal([]byte(jsonStr), &p2)

// 美化输出
data, _ := json.MarshalIndent(p, "", "  ")
```

### 7. net/http - HTTP 客户端和服务器

```go
import "net/http"

// HTTP 客户端
resp, err := http.Get("https://api.example.com")
defer resp.Body.Close()
body, _ := io.ReadAll(resp.Body)

// POST 请求
resp, err := http.Post(url, "application/json", bytes.NewBuffer(data))

// HTTP 服务器
http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
})
http.ListenAndServe(":8080", nil)
```

## 实战技巧

### 1. 泛型（Go 1.18+）

```go
// 泛型函数
func Max[T int | float64](a, b T) T {
    if a > b {
        return a
    }
    return b
}

// 泛型类型
type Queue[T any] interface {
    Enqueue(item T) bool
    Dequeue() (T, bool)
}

// 项目示例：循环队列
type CircularQueue[T any] struct {
    data  []T
    front int
    rear  int
    count int
}

func NewCircularQueue[T any](maxSize int) *CircularQueue[T] {
    return &CircularQueue[T]{
        data: make([]T, maxSize),
    }
}
```

### 2. 结构体标签

```go
// TOML 标签
type Config struct {
    Host string `toml:"host"`
    Port int    `toml:"port"`
}

// JSON 标签
type User struct {
    Name  string `json:"name"`
    Email string `json:"email,omitempty"`  // 空值时省略
    Age   int    `json:"-"`                // 忽略字段
}
```

### 3. defer 延迟执行

```go
// 延迟执行（后进先出）
func readFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()  // 函数返回前执行
    
    // 读取文件
    return nil
}

// 多个 defer
defer fmt.Println("1")
defer fmt.Println("2")
defer fmt.Println("3")
// 输出：3 2 1
```

### 4. panic 和 recover

```go
// panic 触发恐慌
func divide(a, b int) int {
    if b == 0 {
        panic("除数不能为零")
    }
    return a / b
}

// recover 恢复恐慌
func safeCall() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("恢复:", r)
        }
    }()
    
    divide(10, 0)  // 会触发 panic
}
```

### 5. 上下文（Context）

```go
import "context"

// 创建上下文
ctx := context.Background()
ctx, cancel := context.WithCancel(ctx)
ctx, cancel := context.WithTimeout(ctx, time.Second)
ctx, cancel := context.WithDeadline(ctx, time.Now().Add(time.Second))

// 使用上下文
select {
case <-ctx.Done():
    fmt.Println("上下文已取消")
case result := <-ch:
    fmt.Println("收到结果:", result)
}

// 传递值
ctx = context.WithValue(ctx, "key", "value")
value := ctx.Value("key")
```

### 6. 嵌入静态资源（embed）

```go
import "embed"

// 嵌入单个文件
//go:embed version.txt
var version string

// 嵌入多个文件
//go:embed static/*
var staticFiles embed.FS

// 项目示例
//go:embed static/index.html
var indexHTML []byte

// 使用嵌入的文件系统
http.Handle("/static/", http.FileServer(http.FS(staticFiles)))
```

### 7. 正则表达式

```go
import "regexp"

// 编译正则表达式
re := regexp.MustCompile(`\d+`)

// 匹配
matched := re.MatchString("abc123")  // true

// 查找
result := re.FindString("abc123def456")  // "123"
results := re.FindAllString("abc123def456", -1)  // ["123", "456"]

// 替换
result := re.ReplaceAllString("abc123", "XXX")  // "abcXXX"

// 项目示例：提取视频 ID
re := regexp.MustCompile(`https://(?:www\.)?bilibili\.com/(?:video/|video/av\d+/)(BV[a-zA-Z0-9]+)`)
matches := re.FindStringSubmatch(url)
if len(matches) > 1 {
    videoId := matches[1]
}
```

### 8. 命令行参数

```go
import (
    "flag"
    "os"
)

// 定义命令行参数
var (
    host = flag.String("host", "127.0.0.1", "服务器地址")
    port = flag.Int("port", 8888, "服务器端口")
    debug = flag.Bool("debug", false, "调试模式")
)

func main() {
    // 解析参数
    flag.Parse()

    fmt.Printf("Host: %s, Port: %d, Debug: %v\n", *host, *port, *debug)

    // 获取非标志参数
    args := flag.Args()
}

// 使用：go run main.go -host=0.0.0.0 -port=9000 -debug
```

### 9. 文件读写

```go
import (
    "bufio"
    "io"
    "os"
)

// 读取整个文件
data, err := os.ReadFile("file.txt")

// 写入整个文件
err := os.WriteFile("file.txt", []byte("content"), 0644)

// 逐行读取
file, _ := os.Open("file.txt")
defer file.Close()

scanner := bufio.NewScanner(file)
for scanner.Scan() {
    line := scanner.Text()
    fmt.Println(line)
}

// 逐行写入
file, _ := os.Create("file.txt")
defer file.Close()

writer := bufio.NewWriter(file)
writer.WriteString("line 1\n")
writer.WriteString("line 2\n")
writer.Flush()

// 复制文件
src, _ := os.Open("source.txt")
defer src.Close()

dst, _ := os.Create("dest.txt")
defer dst.Close()

io.Copy(dst, src)
```

### 10. 类型转换

```go
// 基本类型转换
var i int = 42
var f float64 = float64(i)
var u uint = uint(i)

// 字符串转数字
i, err := strconv.Atoi("123")
f, err := strconv.ParseFloat("3.14", 64)
b, err := strconv.ParseBool("true")

// 数字转字符串
s := strconv.Itoa(123)
s := strconv.FormatFloat(3.14, 'f', 2, 64)
s := strconv.FormatBool(true)

// 字符串和字节切片
s := "hello"
b := []byte(s)
s = string(b)
```

## 项目实战案例

### 案例 1：HTTP 路由设置

```go
package router

import (
    "krillin-ai/internal/handler"
    "github.com/gin-gonic/gin"
)

func SetupRouter(r *gin.Engine) {
    // 创建路由组
    api := r.Group("/api")

    // 创建处理器
    hdl := handler.NewHandler()

    // 注册路由
    {
        api.POST("/capability/subtitleTask", hdl.StartSubtitleTask)
        api.GET("/capability/subtitleTask", hdl.GetSubtitleTask)
        api.POST("/file", hdl.UploadFile)
        api.GET("/file/*filepath", hdl.DownloadFile)
    }

    // 静态文件服务
    r.StaticFS("/static", http.FS(staticFiles))
}
```

### 案例 2：配置加载

```go
package config

import (
    "github.com/BurntSushi/toml"
    "os"
)

type Config struct {
    App    App    `toml:"app"`
    Server Server `toml:"server"`
}

type App struct {
    SegmentDuration int    `toml:"segment_duration"`
    Proxy           string `toml:"proxy"`
}

type Server struct {
    Host string `toml:"host"`
    Port int    `toml:"port"`
}

var Conf = Config{
    App: App{
        SegmentDuration: 5,
    },
    Server: Server{
        Host: "127.0.0.1",
        Port: 8888,
    },
}

func LoadConfig() error {
    configPath := "./config/config.toml"
    if _, err := os.Stat(configPath); os.IsNotExist(err) {
        return err
    }

    _, err := toml.DecodeFile(configPath, &Conf)
    return err
}
```

### 案例 3：依赖注入

```go
package service

import (
    "krillin-ai/config"
    "krillin-ai/internal/types"
    "krillin-ai/pkg/openai"
    "krillin-ai/pkg/whisper"
)

type Service struct {
    Transcriber   types.Transcriber
    ChatCompleter types.ChatCompleter
    TtsClient     types.Ttser
}

func NewService() *Service {
    var transcriber types.Transcriber

    // 根据配置选择实现
    switch config.Conf.Transcribe.Provider {
    case "openai":
        transcriber = whisper.NewClient(
            config.Conf.Transcribe.Openai.BaseUrl,
            config.Conf.Transcribe.Openai.ApiKey,
            config.Conf.App.Proxy,
        )
    case "fasterwhisper":
        transcriber = fasterwhisper.NewProcessor(
            config.Conf.Transcribe.Fasterwhisper.Model,
        )
    }

    return &Service{
        Transcriber:   transcriber,
        ChatCompleter: openai.NewClient(),
        TtsClient:     openai.NewClient(),
    }
}
```

### 案例 4：并发任务处理

```go
package service

import (
    "context"
    "fmt"
    "golang.org/x/sync/errgroup"
    "runtime"
    "sync"
)

func (s Service) ProcessConcurrently(items []Item) error {
    // 创建错误组
    eg, ctx := errgroup.WithContext(context.Background())

    // 创建任务队列
    pendingQueue := make(chan Item, len(items))
    resultQueue := make(chan Result, len(items))

    // 发送任务到队列
    for _, item := range items {
        pendingQueue <- item
    }
    close(pendingQueue)

    // 启动工作协程（数量等于 CPU 核心数）
    for range runtime.NumCPU() {
        eg.Go(func() error {
            for {
                select {
                case <-ctx.Done():
                    return nil
                case item, ok := <-pendingQueue:
                    if !ok {
                        return nil
                    }
                    // 处理任务
                    result, err := s.processItem(item)
                    if err != nil {
                        return err
                    }
                    resultQueue <- result
                }
            }
        })
    }

    // 收集结果
    eg.Go(func() error {
        results := make([]Result, 0, len(items))
        for range items {
            result := <-resultQueue
            results = append(results, result)
        }
        close(resultQueue)
        return nil
    })

    // 等待所有任务完成
    return eg.Wait()
}
```

### 案例 5：限流器（信号量）

```go
package service

import (
    "sync"
)

func (s Service) ProcessWithRateLimit(items []Item) error {
    maxConcurrency := 3
    semaphore := make(chan struct{}, maxConcurrency)
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(index int, it Item) {
            defer wg.Done()

            // 获取信号量（限制并发数）
            semaphore <- struct{}{}
            defer func() { <-semaphore }()

            // 处理任务
            s.processItem(it)
        }(i, item)
    }

    wg.Wait()
    return nil
}
```

### 案例 6：日志记录

```go
package log

import (
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

var logger *zap.Logger

func InitLogger() {
    config := zap.NewProductionConfig()
    config.EncoderConfig.TimeKey = "timestamp"
    config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

    logger, _ = config.Build()
}

func GetLogger() *zap.Logger {
    return logger
}

// 使用示例
log.GetLogger().Info("任务开始",
    zap.String("taskId", taskId),
    zap.Int("segmentNum", segmentNum),
)

log.GetLogger().Error("任务失败",
    zap.String("taskId", taskId),
    zap.Error(err),
)
```

### 案例 7：工具函数

```go
package util

import (
    "math/rand"
    "regexp"
    "strings"
)

// 生成随机字符串
func GenerateRandString(n int) string {
    chars := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    b := make([]rune, n)
    for i := range b {
        b[i] = chars[rand.Intn(len(chars))]
    }
    return string(b)
}

// 清理路径名
func SanitizePathName(name string) string {
    // 替换非法字符
    illegalChars := regexp.MustCompile(`[<>:"/\\|?*\[\]\x00-\x1F]`)
    sanitized := illegalChars.ReplaceAllString(name, "_")

    // 去除首尾空格
    sanitized = strings.TrimSpace(sanitized)

    // 防止空字符串
    if sanitized == "" {
        sanitized = "unnamed"
    }

    return sanitized
}

// 格式化时间
func FormatTime(seconds float32) string {
    totalSeconds := int(seconds)
    milliseconds := int((seconds - float32(totalSeconds)) * 1000)

    hours := totalSeconds / 3600
    minutes := (totalSeconds % 3600) / 60
    secs := totalSeconds % 60

    return fmt.Sprintf("%02d:%02d:%02d,%03d", hours, minutes, secs, milliseconds)
}
```

## 常见陷阱和最佳实践

### 1. 切片陷阱

```go
// ❌ 错误：切片共享底层数组
a := []int{1, 2, 3, 4, 5}
b := a[1:3]  // [2, 3]
b[0] = 999
fmt.Println(a)  // [1, 999, 3, 4, 5] - a 也被修改了！

// ✅ 正确：复制切片
b := make([]int, len(a[1:3]))
copy(b, a[1:3])
b[0] = 999
fmt.Println(a)  // [1, 2, 3, 4, 5] - a 不受影响
```

### 2. 循环变量陷阱

```go
// ❌ 错误：所有 goroutine 共享同一个变量
for _, item := range items {
    go func() {
        process(item)  // 所有 goroutine 可能处理同一个 item
    }()
}

// ✅ 正确：传递参数
for _, item := range items {
    go func(it Item) {
        process(it)
    }(item)
}
```

### 3. Map 并发陷阱

```go
// ❌ 错误：普通 map 不是并发安全的
m := make(map[string]int)
go func() { m["key"] = 1 }()
go func() { m["key"] = 2 }()  // 可能导致 panic

// ✅ 正确：使用 sync.Map
var m sync.Map
go func() { m.Store("key", 1) }()
go func() { m.Store("key", 2) }()
```

### 4. defer 陷阱

```go
// ❌ 错误：defer 在循环中
for _, file := range files {
    f, _ := os.Open(file)
    defer f.Close()  // 所有文件在函数结束时才关闭
    // 处理文件
}

// ✅ 正确：使用匿名函数
for _, file := range files {
    func() {
        f, _ := os.Open(file)
        defer f.Close()  // 每次循环结束时关闭
        // 处理文件
    }()
}
```

### 5. 错误处理最佳实践

```go
// ✅ 推荐：明确的错误处理
result, err := someFunction()
if err != nil {
    log.GetLogger().Error("操作失败", zap.Error(err))
    return fmt.Errorf("someFunction 失败: %w", err)
}

// ❌ 不推荐：忽略错误
result, _ := someFunction()

// ✅ 推荐：早返回
if err != nil {
    return err
}
// 继续正常流程

// ❌ 不推荐：深层嵌套
if err == nil {
    if result != nil {
        if result.Valid {
            // 深层嵌套
        }
    }
}
```

### 6. 性能优化技巧

```go
// 预分配切片容量
slice := make([]int, 0, expectedSize)

// 使用 strings.Builder 拼接字符串
var builder strings.Builder
for _, s := range strings {
    builder.WriteString(s)
}
result := builder.String()

// 使用 sync.Pool 复用对象
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

buf := bufferPool.Get().(*bytes.Buffer)
defer bufferPool.Put(buf)
buf.Reset()
```

## 测试

### 1. 单元测试

```go
package util

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5

    if result != expected {
        t.Errorf("Add(2, 3) = %d; 期望 %d", result, expected)
    }
}

// 表格驱动测试
func TestAddTable(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"正数", 2, 3, 5},
        {"负数", -2, -3, -5},
        {"零", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("Add(%d, %d) = %d; 期望 %d",
                    tt.a, tt.b, result, tt.expected)
            }
        })
    }
}
```

### 2. 基准测试

```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(2, 3)
    }
}

// 运行：go test -bench=.
```

## 相关资源

- [Go 官方文档](https://golang.org/doc/)
- [Go by Example](https://gobyexample.com/)
- [Effective Go](https://golang.org/doc/effective_go)
- [Go 标准库](https://pkg.go.dev/std)

