MCP，即模型上下文协议（Model Context Protocol），是一个开放协议，用于标准化应用程序向 LLM 提供上下文的方式。尽管有一些开发开销，但 MCP 提供了一个宝贵的机会，让您可以与其他开发人员共享工具、资源和提示词，无论您使用的是哪种技术栈。同样，您也可以使用其他开发人员构建的工具，而无需重写代码。

本指南将带您了解如何在 DSPy 中使用 MCP 工具。为了演示，我们将构建一个航空公司服务代理，它可以帮助用户预订航班以及修改或取消现有预订。这将依赖于一个包含自定义工具的 MCP 服务器，但这应该很容易推广到 [社区构建的 MCP 服务器](https://modelcontextprotocol.io/examples)。

如何运行本教程

本教程无法在 Google Colab 或 Databricks notebooks 等托管的 IPython 笔记本中运行。要运行代码，您需要按照指南在本地设备上编写代码。代码已在 macOS 上测试，在 Linux 环境中应该也能以相同方式工作。

## 安装依赖

在开始之前，让我们安装所需的依赖项：

```
pip install -U "dspy[mcp]"

```

## MCP 服务器设置

首先，让我们为航空公司代理设置 MCP 服务器，它包含：

* **一组数据库**
  * 用户数据库：存储用户信息。
  * 航班数据库：存储航班信息。
  * 机票数据库：存储客户机票。
* **一组工具**
  * `fetch_flight_info`：获取特定日期的航班信息。
  * `fetch_itinerary`：获取已预订的行程信息。
  * `book_itinerary`：代表用户预订航班。
  * `modify_itinerary`：修改行程，包括更改航班或取消。
  * `get_user_info`：获取用户信息。
  * `file_ticket`：提交待办工单以寻求人工协助。

在您的工作目录中，创建一个名为 `mcp_server.py` 的文件，并将以下内容粘贴到其中：

```python
import random
import string

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# 创建一个 MCP 服务器
mcp = FastMCP("Airline Agent")

class Date(BaseModel):
    # LLM 在指定 `datetime.datetime` 方面表现不佳
    year: int
    month: int
    day: int
    hour: int

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight

class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile

user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}

@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights

@mcp.tool()
def fetch_itinerary(confirmation_number: str):
    """Fetch a booked itinerary information from database"""
    return itinery_database.get(confirmation_number)

@mcp.tool()
def pick_flight(flights: list[Flight]):
    """Pick up the best flight that matches users' request."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]

def generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))

@mcp.tool()
def book_itinerary(flight: Flight, user_profile: UserProfile):
    """Book a flight on behalf of the user."""
    confirmation_number = generate_id()
    while confirmation_number in itinery_database:
        confirmation_number = generate_id()
    itinery_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinery_database[confirmation_number]

@mcp.tool()
def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Cancel an itinerary on behalf of the user."""
    if confirmation_number in itinery_database:
        del itinery_database[confirmation_number]
        return
    raise ValueError("Cannot find the itinerary, please check your confirmation number.")

@mcp.tool()
def get_user_info(name: str):
    """Fetch the user profile from database with given name."""
    return user_database.get(name)

@mcp.tool()
def file_ticket(user_request: str, user_profile: UserProfile):
    """File a customer support ticket if this is something the agent cannot handle."""
    ticket_id = generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id

if __name__ == "__main__":
    mcp.run()

```

在启动服务器之前，让我们先看一下代码。

我们首先创建了一个 `FastMCP` 实例，这是一个帮助快速构建 MCP 服务器的实用工具：

```python
mcp = FastMCP("Airline Agent")

```

然后我们定义数据结构，在实际应用中，这将是数据库模式（Schema），例如：

```python
class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

```

接下来，我们初始化数据库实例。在实际应用中，这些将是连接到实际数据库的连接器，但为了简单起见，我们只是使用字典：

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

```

下一步是定义工具并用 `@mcp.tool()` 标记它们，以便 MCP 客户端可以将它们作为 MCP 工具发现：

```python
@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights

```

最后一步是启动服务器：

```python
if __name__ == "__main__":
    mcp.run()

```

现在我们已经完成了服务器的编写！让我们启动它：

```bash
python path_to_your_working_directory/mcp_server.py

```

## 编写利用 MCP 服务器中工具的 DSPy 程序

现在服务器正在运行，让我们构建实际的航空公司服务代理，它利用我们服务器中的 MCP 工具来协助用户。在您的工作目录中，创建一个名为 `dspy_mcp_agent.py` 的文件，并按照指南向其中添加代码。

### 从 MCP 服务器收集工具

我们首先需要从 MCP 服务器收集所有可用工具，并使它们可供 DSPy 使用。DSPy 提供了一个 API [`dspy.Tool`](https://dspy.ai/api/primitives/Tool/) 作为标准工具接口。让我们将所有 MCP 工具转换为 `dspy.Tool`。

我们需要创建一个 MCP 客户端实例来与 MCP 服务器通信，获取所有可用工具，并使用静态方法 `from_mcp_tool` 将它们转换为 `dspy.Tool`：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 为 stdio 连接创建服务器参数
server_params = StdioServerParameters(
    command="python",  # 可执行文件
    args=["path_to_your_working_directory/mcp_server.py"],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            # 列出可用工具
            tools = await session.list_tools()

            # 将 MCP 工具转换为 DSPy 工具
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())

```

通过上面的代码，我们成功收集了所有可用的 MCP 工具并将它们转换为 DSPy 工具。

### 构建 DSPy 代理以处理客户请求

现在我们将使用 `dspy.ReAct` 来构建处理客户请求的代理。`ReAct` 代表 "推理和行动（Reasoning and Acting）"，它要求 LLM 决定是调用工具还是结束流程。如果需要工具，LLM 负责决定调用哪个工具并提供适当的参数。

像往常一样，我们需要创建一个 `dspy.Signature` 来定义我们代理的输入和输出：

```python
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests. You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )

```

并为我们的代理选择一个 LM：

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

```

然后我们通过将工具和签名传递给 `dspy.ReAct` API 来创建 ReAct 代理。我们可以将完整的代码脚本组合在一起：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy

# 为 stdio 连接创建服务器参数
server_params = StdioServerParameters(
    command="python",  # 可执行文件
    args=["script_tmp/mcp_server.py"],  # 可选的命令行参数
    env=None,  # 可选的环境变量
)

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            # 列出可用工具
            tools = await session.list_tools()

            # 将 MCP 工具转换为 DSPy 工具
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # 创建代理
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))

```

注意，我们必须调用 `react.acall`，因为 MCP 工具默认是异步的。让我们执行该脚本：

```bash
python path_to_your_working_directory/dspy_mcp_agent.py

```

您应该会看到类似于以下的输出：

```
Prediction(
    trajectory={'thought_0': 'I need to fetch flight information for Adam from SFO to JFK on 09/01/2025 to find available flights for booking.', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': ['{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}', '{"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}'], ..., 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},
    reasoning="I successfully booked a flight for Adam from SFO to JFK on 09/01/2025. I found two available flights, selected the more economical option (flight DA123 at 1 AM for $200), retrieved Adam's user profile, and completed the booking process. The confirmation number for the flight is 8h7clk3q.",
    process_result='Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 8h7clk3q.'
)

```

`trajectory` 字段包含整个思考和行动过程。如果您对底层发生的事情感到好奇，请查看 [可观测性指南](https://dspy.ai/tutorials/observability/) 以设置 MLflow，它可以可视化 `dspy.ReAct` 内部发生的每一步！

## 总结

在本指南中，我们构建了一个利用自定义 MCP 服务器和 `dspy.ReAct` 模块的航空公司服务代理。在 MCP 支持的背景下，DSPy 为与 MCP 工具交互提供了一个简单的接口，使您可以灵活地实现所需的任何功能。
