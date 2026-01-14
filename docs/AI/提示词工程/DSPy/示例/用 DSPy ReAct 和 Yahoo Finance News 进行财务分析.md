本教程演示如何使用 DSPy ReAct 结合 [LangChain 的 Yahoo Finance News 工具](https://python.langchain.com/docs/integrations/tools/yahoo_finance_news/) 来构建一个用于实时市场分析的财务分析代理。

## 您将构建的内容

一个能够获取新闻、分析情绪并提供投资见解的财务代理。

## 设置

```bash
pip install dspy langchain langchain-community yfinance
```

## 第 1 步：将 LangChain 工具转换为 DSPy 工具

```python
import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf

# 配置 DSPy
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm, allow_tool_async_sync_conversion=True)

# 将 LangChain Yahoo Finance 工具转换为 DSPy 工具
yahoo_finance_tool = YahooFinanceNewsTool()
finance_news_tool = Tool.from_langchain(yahoo_finance_tool)
```

## 第 2 步：创建辅助财务工具

```python
def get_stock_price(ticker: str) -> str:
    """获取当前股票价格和基本信息。"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")
        
        if hist.empty:
            return f"Could not retrieve data for {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        result = {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_percent": round(change_pct, 2),
            "company": info.get('longName', ticker)
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error: {str(e)}"

def compare_stocks(tickers: str) -> str:
    """比较多只股票（以逗号分隔）。"""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        comparison = []
        
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                
                comparison.append({
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change_percent": round(change_pct, 2)
                })
        
        return json.dumps(comparison)
    except Exception as e:
        return f"Error: {str(e)}"
```

## 第 3 步：构建财务 ReAct 代理

```python
class FinancialAnalysisAgent(dspy.Module):
    """使用 Yahoo Finance 数据进行财务分析的 ReAct 代理。"""
    
    def __init__(self):
        super().__init__()
        
        # 组合所有工具
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
            get_stock_price,
            compare_stocks
        ]
        
        # 初始化 ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )
    
    def forward(self, financial_query: str):
        return self.react(financial_query=financial_query)
```

## 第 4 步：运行财务分析

```python
def run_financial_demo():
    """财务分析代理的演示。"""
    
    # 初始化代理
    agent = FinancialAnalysisAgent()
    
    # 示例查询
    queries = [
        "What's the latest news about Apple (AAPL) and how might it affect the stock price?",
        "Compare AAPL, GOOGL, and MSFT performance",
        "Find recent Tesla news and analyze sentiment"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        response = agent(financial_query=query)
        print(f"Analysis: {response.analysis_response}")
        print("-" * 50)

# 运行演示
if __name__ == "__main__":
    run_financial_demo()
```

## 示例输出

当您使用类似“关于苹果的最新消息是什么？”的查询运行代理时，它将：

1. 使用 Yahoo Finance News 工具获取最近的苹果新闻
2. 获取当前股票价格数据
3. 分析信息并提供见解

**响应示例：**
```
Analysis: Given the current price of Apple (AAPL) at $196.58 and the slight increase of 0.48%, it appears that the stock is performing steadily in the market. However, the inability to access the latest news means that any significant developments that could influence investor sentiment and stock price are unknown. Investors should keep an eye on upcoming announcements or market trends that could impact Apple's performance, especially in comparison to other tech stocks like Microsoft (MSFT), which is also showing a positive trend.
```

## 使用异步工具

许多 Langchain 工具使用异步操作以获得更好的性能。有关异步工具的详细信息，请参阅 [工具文档](../../learn/programming/tools.md#async-tools)。

## 主要优势

- **工具集成**：将 LangChain 工具与 DSPy ReAct 无缝结合
- **实时数据**：访问当前市场数据和新闻
- **可扩展**：易于添加更多财务分析工具
- **智能推理**：ReAct 框架提供逐步分析

本教程展示了 DSPy 的 ReAct 框架如何与 LangChain 的财务工具配合使用，以创建智能市场分析代理。
