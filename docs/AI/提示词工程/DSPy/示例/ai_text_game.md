# 使用 DSPy 构建创意文本 AI 游戏

本教程演示了如何使用 DSPy 的模块化编程方法创建一个交互式文本冒险游戏。你将构建一个动态游戏，其中 AI 负责处理叙事生成、角色互动和自适应游戏玩法。

## 你将构建什么

一个智能文本冒险游戏，具有以下特点：

- 动态故事生成和分支叙事
- AI 驱动的角色互动和对话
- 响应玩家选择的自适应游戏玩法
- 物品栏和角色成长系统
- 保存/加载游戏状态功能

## 设置

```bash
pip install dspy rich typer
```

## 第 1 步：核心游戏框架

```python
import dspy
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import typer

# 配置 DSPy
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

console = Console()

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    INVENTORY = "inventory"
    CHARACTER = "character"
    GAME_OVER = "game_over"

@dataclass
class Player:
    name: str
    health: int = 100
    level: int = 1
    experience: int = 0
    inventory: list[str] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=lambda: {
        "strength": 10,
        "intelligence": 10,
        "charisma": 10,
        "stealth": 10
    })
    
    def add_item(self, item: str):
        self.inventory.append(item)
        console.print(f"[green]已将 {item} 添加到物品栏！[/green]")
    
    def remove_item(self, item: str) -> bool:
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False
    
    def gain_experience(self, amount: int):
        self.experience += amount
        old_level = self.level
        self.level = 1 + (self.experience // 100)
        if self.level > old_level:
            console.print(f"[bold yellow]升级了！你现在是 {self.level} 级！[/bold yellow]")

@dataclass
class GameContext:
    current_location: str = "村庄广场" # Village Square
    story_progress: int = 0
    visited_locations: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    game_flags: dict[str, bool] = field(default_factory=dict)
    
    def add_flag(self, flag: str, value: bool = True):
        self.game_flags[flag] = value
    
    def has_flag(self, flag: str) -> bool:
        return self.game_flags.get(flag, False)

class GameEngine:
    def __init__(self):
        self.player = None
        self.context = GameContext()
        self.state = GameState.MENU
        self.running = True
        
    def save_game(self, filename: str = "savegame.json"):
        """保存当前游戏状态。"""
        save_data = {
            "player": {
                "name": self.player.name,
                "health": self.player.health,
                "level": self.player.level,
                "experience": self.player.experience,
                "inventory": self.player.inventory,
                "skills": self.player.skills
            },
            "context": {
                "current_location": self.context.current_location,
                "story_progress": self.context.story_progress,
                "visited_locations": self.context.visited_locations,
                "npcs_met": self.context.npcs_met,
                "completed_quests": self.context.completed_quests,
                "game_flags": self.context.game_flags
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        console.print(f"[green]游戏已保存到 {filename}！[/green]")
    
    def load_game(self, filename: str = "savegame.json") -> bool:
        """从文件加载游戏状态。"""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            # 重构玩家
            player_data = save_data["player"]
            self.player = Player(
                name=player_data["name"],
                health=player_data["health"],
                level=player_data["level"],
                experience=player_data["experience"],
                inventory=player_data["inventory"],
                skills=player_data["skills"]
            )
            
            # 重构上下文
            context_data = save_data["context"]
            self.context = GameContext(
                current_location=context_data["current_location"],
                story_progress=context_data["story_progress"],
                visited_locations=context_data["visited_locations"],
                npcs_met=context_data["npcs_met"],
                completed_quests=context_data["completed_quests"],
                game_flags=context_data["game_flags"]
            )
            
            console.print(f"[green]已从 {filename} 加载游戏！[/green]")
            return True
            
        except FileNotFoundError:
            console.print(f"[red]未找到存档文件 {filename}！[/red]")
            return False
        except Exception as e:
            console.print(f"[red]加载游戏出错: {e}！[/red]")
            return False

# 初始化游戏引擎
game = GameEngine()
```

## 第 2 步：AI 驱动的故事生成

```python
class StoryGenerator(dspy.Signature):
    """根据当前游戏状态生成动态故事内容。"""
    location: str = dspy.InputField(desc="当前位置")
    player_info: str = dspy.InputField(desc="玩家信息和属性")
    story_progress: int = dspy.InputField(desc="当前故事进度等级")
    recent_actions: str = dspy.InputField(desc="玩家最近的行动")
    
    scene_description: str = dspy.OutputField(desc="当前场景的生动描述")
    available_actions: list[str] = dspy.OutputField(desc="可能的玩家行动列表")
    npcs_present: list[str] = dspy.OutputField(desc="该位置出现的 NPC")
    items_available: list[str] = dspy.OutputField(desc="可以找到或互动的物品")

class DialogueGenerator(dspy.Signature):
    """生成 NPC 对话和回应。"""
    npc_name: str = dspy.InputField(desc="NPC 的名字和类型")
    npc_personality: str = dspy.InputField(desc="NPC 的个性和背景")
    player_input: str = dspy.InputField(desc="玩家说的话或做的事")
    context: str = dspy.InputField(desc="当前游戏上下文和历史")
    
    npc_response: str = dspy.OutputField(desc="NPC 的对话回应")
    mood_change: str = dspy.OutputField(desc="NPC 的情绪变化 (积极/消极/中性)")
    quest_offered: bool = dspy.OutputField(desc="NPC 是否提供任务")
    information_revealed: str = dspy.OutputField(desc="分享的任何重要信息")

class ActionResolver(dspy.Signature):
    """解决玩家行动并确定结果。"""
    action: str = dspy.InputField(desc="玩家选择的行动")
    player_stats: str = dspy.InputField(desc="玩家当前的属性和技能")
    context: str = dspy.InputField(desc="当前游戏上下文")
    difficulty: str = dspy.InputField(desc="行动的难度等级")
    
    success: bool = dspy.OutputField(desc="行动是否成功")
    outcome_description: str = dspy.OutputField(desc="发生的事情的描述")
    stat_changes: dict[str, int] = dspy.OutputField(desc="玩家属性的变化")
    items_gained: list[str] = dspy.OutputField(desc="从此行动中获得的物品")
    experience_gained: int = dspy.OutputField(desc="获得的经验值")

class GameAI(dspy.Module):
    """用于游戏逻辑和叙事的主要 AI 模块。"""
    
    def __init__(self):
        super().__init__()
        self.story_gen = dspy.ChainOfThought(StoryGenerator)
        self.dialogue_gen = dspy.ChainOfThought(DialogueGenerator)
        self.action_resolver = dspy.ChainOfThought(ActionResolver)
    
    def generate_scene(self, player: Player, context: GameContext, recent_actions: str = "") -> Dict:
        """生成当前场景描述和选项。"""
        
        player_info = f"等级 {player.level} {player.name}, 生命值: {player.health}, 技能: {player.skills}"
        
        scene = self.story_gen(
            location=context.current_location,
            player_info=player_info,
            story_progress=context.story_progress,
            recent_actions=recent_actions
        )
        
        return {
            "description": scene.scene_description,
            "actions": scene.available_actions,
            "npcs": scene.npcs_present,
            "items": scene.items_available
        }
    
    def handle_dialogue(self, npc_name: str, player_input: str, context: GameContext) -> Dict:
        """处理与 NPC 的对话。"""
        
        # 根据名字和上下文创建 NPC 个性
        personality_map = {
            "Village Elder": "睿智，知识渊博，说话像打谜语，拥有古老的知识", # Village Elder
            "Merchant": "贪婪但公平，喜欢讨价还价，了解贵重物品", # Merchant
            "Guard": "尽职尽责，怀疑陌生人，严格遵守规则", # Guard
            "Thief": "鬼鬼祟祟，不值得信任，掌握隐藏事物的信息", # Thief
            "Wizard": "神秘，强大，谈论魔法和古老力量" # Wizard
        }
        
        personality = personality_map.get(npc_name, "拥有当地知识的友好村民")
        game_context = f"地点: {context.current_location}, 故事进度: {context.story_progress}"
        
        response = self.dialogue_gen(
            npc_name=npc_name,
            npc_personality=personality,
            player_input=player_input,
            context=game_context
        )
        
        return {
            "response": response.npc_response,
            "mood": response.mood_change,
            "quest": response.quest_offered,
            "info": response.information_revealed
        }
    
    def resolve_action(self, action: str, player: Player, context: GameContext) -> Dict:
        """解决玩家行动并确定结果。"""
        
        player_stats = f"等级 {player.level}, 生命值 {player.health}, 技能: {player.skills}"
        game_context = f"地点: {context.current_location}, 进度: {context.story_progress}"
        
        # 根据行动类型确定难度
        difficulty = "medium"
        if any(word in action.lower() for word in ["fight", "battle", "attack", "战斗", "攻击"]):
            difficulty = "hard"
        elif any(word in action.lower() for word in ["look", "examine", "talk", "看", "检查", "交谈"]):
            difficulty = "easy"
        
        result = self.action_resolver(
            action=action,
            player_stats=player_stats,
            context=game_context,
            difficulty=difficulty
        )
        
        return {
            "success": result.success,
            "description": result.outcome_description,
            "stat_changes": result.stat_changes,
            "items": result.items_gained,
            "experience": result.experience_gained
        }

# 初始化 AI
ai = GameAI()
```

## 第 3 步：游戏界面和交互

```python
def display_game_header():
    """显示游戏标题。"""
    header = Text("🏰 神秘领域冒险 (MYSTIC REALM ADVENTURE) 🏰", style="bold magenta")
    console.print(Panel(header, style="bright_blue"))

def display_player_status(player: Player):
    """显示玩家状态面板。"""
    status = f"""
[bold]姓名:[/bold] {player.name}
[bold]等级:[/bold] {player.level} (XP: {player.experience})
[bold]生命值:[/bold] {player.health}/100
[bold]技能:[/bold]
  • 力量 (Strength): {player.skills['strength']}
  • 智力 (Intelligence): {player.skills['intelligence']}
  • 魅力 (Charisma): {player.skills['charisma']}
  • 潜行 (Stealth): {player.skills['stealth']}
[bold]物品栏:[/bold] {len(player.inventory)} 件物品
    """
    console.print(Panel(status.strip(), title="玩家状态", style="green"))

def display_location(context: GameContext, scene: Dict):
    """显示当前位置和场景。"""
    location_panel = f"""
[bold yellow]{context.current_location}[/bold yellow]

{scene['description']}
    """
    
    if scene['npcs']:
        location_panel += f"\n\n[bold]出现的 NPC:[/bold] {', '.join(scene['npcs'])}"
    
    if scene['items']:
        location_panel += f"\n[bold]可见物品:[/bold] {', '.join(scene['items'])}"
    
    console.print(Panel(location_panel.strip(), title="当前位置", style="cyan"))

def display_actions(actions: list[str]):
    """显示可用行动。"""
    action_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
    console.print(Panel(action_text, title="可用行动", style="yellow"))

def get_player_choice(max_choices: int) -> int:
    """获取玩家选择并验证输入。"""
    while True:
        try:
            choice = typer.prompt("选择一个行动 (数字)")
            choice_num = int(choice)
            if 1 <= choice_num <= max_choices:
                return choice_num - 1
            else:
                console.print(f"[red]请输入 1 到 {max_choices} 之间的数字[/red]")
        except ValueError:
            console.print("[red]请输入有效的数字[/red]")

def show_inventory(player: Player):
    """显示玩家物品栏。"""
    if not player.inventory:
        console.print(Panel("你的物品栏是空的。", title="物品栏", style="red"))
    else:
        items = "\n".join([f"• {item}" for item in player.inventory])
        console.print(Panel(items, title="物品栏", style="green"))

def main_menu():
    """显示主菜单并处理选择。"""
    console.clear()
    display_game_header()
    
    menu_options = [
        "1. 新游戏 (New Game)",
        "2. 加载游戏 (Load Game)", 
        "3. 玩法说明 (How to Play)",
        "4. 退出 (Exit)"
    ]
    
    menu_text = "\n".join(menu_options)
    console.print(Panel(menu_text, title="主菜单", style="bright_blue")
    
    choice = typer.prompt("选择一个选项")
    return choice

def show_help():
    """显示帮助信息。"""
    help_text = """
[bold]玩法说明:[/bold]

• 这是一个由 AI 驱动的文本冒险游戏
• 通过选择编号选项来做出决定
• 与 NPC 交谈以了解世界并获取任务
• 探索不同地点以寻找物品和冒险
• 你的选择会影响故事和角色发展
• 使用 'inventory' (物品栏) 查看你的物品
• 使用 'status' (状态) 查看角色信息
• 输入 'save' (保存) 保存进度
• 输入 'quit' (退出) 返回主菜单

[bold]提示:[/bold]
• 不同的技能会影响你在各种行动中的成功率
• NPC 会记住你们之前的互动
• 彻底探索——这里有隐藏的秘密！
• 你的声誉会影响 NPC 对你的态度
    """
    console.print(Panel(help_text.strip(), title="游戏帮助", style="blue")
    typer.prompt("按回车键继续")
```

## 第 4 步：主游戏循环

```python
def create_new_character():
    """创建新玩家角色。"""
    console.clear()
    display_game_header()
    
    name = typer.prompt("输入你的角色名字")
    
    # 角色创建与技能点分配
    console.print("\n[bold]角色创建[/bold]")
    console.print("你有 10 点额外技能点可以分配给你的技能。")
    console.print("基础技能各从 10 点开始。\n")
    
    skills = {"strength": 10, "intelligence": 10, "charisma": 10, "stealth": 10}
    points_remaining = 10
    
    for skill in skills.keys():
        if points_remaining > 0:
            console.print(f"剩余点数: {points_remaining}")
            while True:
                try:
                    points = int(typer.prompt(f"添加到 {skill} 的点数 (0-{points_remaining})"))
                    if 0 <= points <= points_remaining:
                        skills[skill] += points
                        points_remaining -= points
                        break
                    else:
                        console.print(f"[red]请输入 0 到 {points_remaining} 之间的数字[/red]")
                except ValueError:
                    console.print("[red]请输入有效的数字[/red]")
    
    player = Player(name=name, skills=skills)
    console.print(f"\n[green]欢迎来到神秘领域，{name}！[/green]")
    return player

def game_loop():
    """主游戏循环。"""
    recent_actions = ""
    
    while game.running and game.state == GameState.PLAYING:
        console.clear()
        display_game_header()
        
        # 生成当前场景
        scene = ai.generate_scene(game.player, game.context, recent_actions)
        
        # 显示游戏状态
        display_player_status(game.player)
        display_location(game.context, scene)
        
        # 添加标准行动
        all_actions = scene['actions'] + ["Check inventory", "Character status", "Save game", "Quit to menu"]
        display_actions(all_actions)
        
        # 获取玩家选择
        choice_idx = get_player_choice(len(all_actions))
        chosen_action = all_actions[choice_idx]
        
        # 处理特殊命令
        if chosen_action == "Check inventory":
            show_inventory(game.player)
            typer.prompt("按回车键继续")
            continue
        elif chosen_action == "Character status":
            display_player_status(game.player)
            typer.prompt("按回车键继续")
            continue
        elif chosen_action == "Save game":
            game.save_game()
            typer.prompt("按回车键继续")
            continue
        elif chosen_action == "Quit to menu":
            game.state = GameState.MENU
            break
        
        # 处理游戏行动
        if chosen_action in scene['actions']:
            # 检查是否与 NPC 对话
            npc_target = None
            for npc in scene['npcs']:
                if npc.lower() in chosen_action.lower():
                    npc_target = npc
                    break
            
            if npc_target:
                # 处理 NPC 互动
                console.print(f"\n[bold]正在与 {npc_target} 交谈...[/bold]")
                dialogue = ai.handle_dialogue(npc_target, chosen_action, game.context)
                
                console.print(f"\n[italic]{npc_target}:[/italic] \"{dialogue['response']}\""
                
                if dialogue['quest']:
                    console.print(f"[yellow]💼 检测到任务机会！[/yellow]")
                
                if dialogue['info']:
                    console.print(f"[blue]ℹ️  {dialogue['info']}[/blue]")
                    
                # 将 NPC 添加到已遇列表
                if npc_target not in game.context.npcs_met:
                    game.context.npcs_met.append(npc_target)
                
                recent_actions = f"与 {npc_target} 交谈: {chosen_action}"
            else:
                # 处理一般行动
                result = ai.resolve_action(chosen_action, game.player, game.context)
                
                console.print(f"\n{result['description']}")
                
                # 应用结果
                if result['success']:
                    console.print("[green]✅ 成功！[/green]")
                    
                    # 应用属性变化
                    for stat, change in result['stat_changes'].items():
                        if stat in game.player.skills:
                            game.player.skills[stat] += change
                            if change > 0:
                                console.print(f"[green]{stat.title()} 增加了 {change}！[/green]")
                        elif stat == "health":
                            game.player.health = max(0, min(100, game.player.health + change))
                            if change > 0:
                                console.print(f"[green]生命值恢复了 {change}！[/green]")
                            elif change < 0:
                                console.print(f"[red]生命值减少了 {abs(change)}！[/red]")
                    
                    # 添加物品
                    for item in result['items']:
                        game.player.add_item(item)
                    
                    # 给予经验
                    if result['experience'] > 0:
                        game.player.gain_experience(result['experience'])
                    
                    # 更新故事进度
                    game.context.story_progress += 1
                else:
                    console.print("[red]❌ 行动没能按计划进行...[/red]")
                
                recent_actions = f"尝试: {chosen_action}"
            
            # 检查游戏结束条件
            if game.player.health <= 0:
                console.print("\n[bold red]💀 你死了！游戏结束！[/bold red]")
                game.state = GameState.GAME_OVER
                break
            
            typer.prompt("\n按回车键继续")

def main():
    """主游戏函数。"""
    while game.running:
        if game.state == GameState.MENU:
            choice = main_menu()
            
            if choice == "1":
                game.player = create_new_character()
                game.context = GameContext()
                game.state = GameState.PLAYING
                console.print("\n[italic]你的冒险开始了...[/italic]")
                typer.prompt("按回车键开始")
                
            elif choice == "2":
                if game.load_game():
                    game.state = GameState.PLAYING
                typer.prompt("按回车键继续")
                
            elif choice == "3":
                show_help()
                
            elif choice == "4":
                game.running = False
                console.print("[bold]感谢游玩！再见！[/bold]")
            
        elif game.state == GameState.PLAYING:
            game_loop()
            
        elif game.state == GameState.GAME_OVER:
            console.print("\n[bold]游戏结束[/bold]")
            restart = typer.confirm("你想返回主菜单吗？")
            if restart:
                game.state = GameState.MENU
            else:
                game.running = False

if __name__ == "__main__":
    main()
```

## 示例游戏玩法 (Example Gameplay)

当你运行游戏时，你将体验到：

**角色创建:**
```
🏰 神秘领域冒险 (MYSTIC REALM ADVENTURE) 🏰

输入你的角色名字: Aria

角色创建
你有 10 点额外技能点可以分配给你的技能。
基础技能各从 10 点开始。

剩余点数: 10
添加到 strength 的点数 (0-10): 2
添加到 intelligence 的点数 (0-8): 4
添加到 charisma 的点数 (0-4): 3
添加到 stealth 的点数 (0-1): 1

欢迎来到神秘领域，Aria！
```

**动态场景生成:**
```
┌──────────── 当前位置 ────────────────────┐
│ 村庄广场                                 │
│                                          │
│ 你站在威洛布鲁克村 (Willowbrook Village) │
│ 熙熙攘攘的中心。古老的石制喷泉欢快地冒着 │
│ 泡，商人们在兜售商品，孩子们在玩耍。一个 │
│ 神秘的戴着兜帽的人潜伏在老橡树的阴影附近。│
│                                          │
│ 出现的 NPC: 村长 (Village Elder), 商人   │
│ 可见物品: 奇怪的奖章, 草药               │
└──────────────────────────────────────────┘

┌────────── 可用行动 ──────────────────────┐
│ 1. 接近那个戴着兜帽的人                  │
│ 2. 与村长交谈                            │
│ 3. 浏览商人的商品                        │
│ 4. 检查那个奇怪的奖章                    │
│ 5. 在喷泉附近采集草药                    │
│ 6. 前往森林小径                          │
└──────────────────────────────────────────┘
```

**AI 生成的对话:**
```
正在与 Village Elder 交谈...

Village Elder: "啊，年轻的旅行者，我感觉到了像晨雾一样环绕着你的
伟大命运。古老的预言说，将会有一个带着勇气印记的人到来。告诉我，
你在旅途中通过有什么... 不寻常的发现吗？"

💼 检测到任务机会！
ℹ️ 村长知道一个可能与你有关的古老预言
```

## 下一步

- **战斗系统**: 增加带有策略的回合制战斗
- **魔法系统**: 带有资源管理的施法系统
- **多人游戏**: 支持合作冒险的网络功能
- **任务系统**: 具有分支结果的复杂多步骤任务
- **世界构建**: 程序化生成的地点和角色
- **音频**: 添加音效和背景音乐

本教程演示了 DSPy 的模块化方法如何实现复杂的交互式系统，其中 AI 处理创意内容生成，同时保持一致的游戏逻辑和玩家代理。