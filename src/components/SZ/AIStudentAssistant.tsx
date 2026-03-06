import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Bot, User, Clock, ChevronRight, BookOpen, 
  HeartHandshake, Home, ShieldPlus, Sparkles, AlertCircle
} from 'lucide-react';

export default function AIStudentAssistant() {
  // 预设的快捷问题分类，对应客户需求中的六大板块
  const quickCategories = [
    { icon: <HeartHandshake size={18} />, label: "心理支持", prompt: "最近压力有点大，学校有心理辅导吗？" },
    { icon: <BookOpen size={18} />, label: "生涯与资助", prompt: "我想了解一下今年的国家奖学金申请条件。" },
    { icon: <Home size={18} />, label: "宿舍管理", prompt: "宿舍空调坏了，怎么报修？" },
    { icon: <ShieldPlus size={18} />, label: "医保政策", prompt: "异地就医怎么用学校的医保报销？" }
  ];

  // 初始聊天记录
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'ai',
      type: 'text',
      text: '同学你好！我是你的社区专属“知心朋友”AI小助手。无论你是遇到宿舍报修难题、想了解奖助学金政策，还是最近心情有些烦闷想找人倾诉，都可以随时跟我说哦。今天有什么我可以帮你的吗？',
      time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
    }
  ]);

  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // 自动滚动到最新消息
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // 处理发送消息
  const handleSend = (text = inputValue) => {
    if (!text.trim()) return;

    // 添加用户消息
    const newUserMsg = {
      id: Date.now(),
      sender: 'user',
      type: 'text',
      text: text,
      time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
    };
    
    setMessages(prev => [...prev, newUserMsg]);
    setInputValue('');
    setIsTyping(true);

    // 模拟AI思考和回复延迟
    setTimeout(() => {
      generateAIResponse(text);
    }, 1500);
  };

  // 模拟AI根据关键词回复（体现打通系统的价值）
  const generateAIResponse = (userText) => {
    setIsTyping(false);
    let aiResponse = {};
    const textStr = userText.toLowerCase();

    if (textStr.includes('报修') || textStr.includes('空调')) {
      aiResponse = {
        id: Date.now() + 1,
        sender: 'ai',
        type: 'action_card',
        title: '🏠 宿舍智能报修引流',
        text: '检测到您需要进行宿舍报修。我已经为您拉取了后勤系统的报修单。您的当前位置是【南区8栋302】，请确认是否为您本人宿舍报修？',
        actionLabel: '一键提交后勤工单'
      };
    } else if (textStr.includes('医保') || textStr.includes('报销')) {
      aiResponse = {
        id: Date.now() + 1,
        sender: 'ai',
        type: 'text',
        text: '关于大学生医保异地报销：\n1. 需要先在“国家医保服务平台”APP进行异地就医备案。\n2. 保留好发票、费用明细单、病历底方。\n3. 返校后于每周三下午交至校医院医保办（笃行楼102）。\n需要我帮您下载《异地备案操作指南.pdf》吗？',
      };
    } else if (textStr.includes('压力') || textStr.includes('心理') || textStr.includes('心情')) {
      aiResponse = {
        id: Date.now() + 1,
        sender: 'ai',
        type: 'action_card',
        title: '💚 心理中心绿色通道',
        text: '抱歉听到你最近压力比较大。学业和生活的确会让人感到疲惫，你愿意多跟我说说吗？\n另外，学校的心理健康中心有专业的老师可以提供帮助，我可以帮你隐秘地预约本周的沙盘体验或一对一咨询。',
        actionLabel: '查看可预约的心理老师'
      };
    } else {
      aiResponse = {
        id: Date.now() + 1,
        sender: 'ai',
        type: 'text',
        text: '我已经理解了您的诉求。目前系统正在对接该板块的数据，您可以尝试询问关于“宿舍报修”、“医保政策”或“心理支持”方面的问题体验智能服务流程。',
      };
    }

    setMessages(prev => [...prev, aiResponse]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen w-full bg-gray-50 text-gray-800 font-sans">
      
      {/* 左侧边栏 - 历史记录与导航 (PC端显示) */}
      <div className="hidden md:flex w-64 flex-col bg-white border-r border-gray-200">
        <div className="p-5 border-b border-gray-100">
          <h1 className="text-xl font-bold text-blue-600 flex items-center gap-2">
            <Sparkles size={24} />
            一站式智慧社区
          </h1>
          <p className="text-xs text-gray-500 mt-1">AI智能体服务矩阵 V1.0</p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <div className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wider">近期对话记录</div>
          <div className="space-y-2">
            <button className="w-full text-left p-2 rounded-lg bg-blue-50 text-blue-700 text-sm font-medium flex items-center gap-2">
              <Clock size={16} /> 了解奖学金政策
            </button>
            <button className="w-full text-left p-2 rounded-lg text-gray-600 hover:bg-gray-50 text-sm flex items-center gap-2">
              <Clock size={16} /> 办理走读申请流程
            </button>
            <button className="w-full text-left p-2 rounded-lg text-gray-600 hover:bg-gray-50 text-sm flex items-center gap-2">
              <Clock size={16} /> 校园卡挂失办理
            </button>
          </div>
        </div>

        {/* 底部学生信息卡片 (模拟已打通统一身份认证) */}
        <div className="p-4 border-t border-gray-100 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold">
              张
            </div>
            <div>
              <div className="font-semibold text-sm">张同学</div>
              <div className="text-xs text-gray-500">计算机学院 2024级</div>
            </div>
          </div>
          <div className="mt-3 flex gap-1 flex-wrap">
            <span className="px-2 py-1 bg-green-100 text-green-700 text-[10px] rounded-full">状态正常</span>
            <span className="px-2 py-1 bg-gray-200 text-gray-600 text-[10px] rounded-full">未住宿</span>
          </div>
        </div>
      </div>

      {/* 右侧主聊天区域 */}
      <div className="flex-1 flex flex-col h-full bg-[#f4f7f9] relative">
        
        {/* 顶部标题栏 */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 shadow-sm z-10">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 flex items-center justify-center text-white">
              <Bot size={18} />
            </div>
            <div>
              <h2 className="font-semibold text-gray-800">AI 学生助手</h2>
              <p className="text-xs text-green-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500 block"></span> 24小时在线
              </p>
            </div>
          </div>
          <div className="hidden sm:flex items-center text-sm text-gray-500 gap-4">
            <button className="hover:text-blue-600 flex items-center gap-1"><AlertCircle size={16}/> 清空对话</button>
          </div>
        </header>

        {/* 聊天消息流 */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {/* 欢迎引导卡片 */}
          <div className="max-w-3xl mx-auto mb-8 bg-white p-5 rounded-2xl shadow-sm border border-blue-100 flex flex-col items-center text-center">
            <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-3 text-blue-600">
              <Bot size={32} />
            </div>
            <h3 className="text-lg font-bold text-gray-800 mb-2">欢迎进入一站式服务大厅</h3>
            <p className="text-sm text-gray-500 mb-5 max-w-md">
              我是基于校园大模型构建的AI社区助手。已为您对接学工、后勤、教务等系统，您可以直接提问或点击下方快捷选项。
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 w-full">
              {quickCategories.map((cat, idx) => (
                <button 
                  key={idx}
                  onClick={() => handleSend(cat.prompt)}
                  className="flex flex-col items-center justify-center gap-2 p-3 rounded-xl border border-gray-100 bg-gray-50 hover:bg-blue-50 hover:border-blue-200 transition-colors"
                >
                  <div className="text-blue-500">{cat.icon}</div>
                  <span className="text-xs font-medium text-gray-700">{cat.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* 动态消息列表 */}
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                <div className={`flex gap-3 max-w-[85%] ${msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  
                  {/* 头像 */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 ${
                    msg.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                  }`}>
                    {msg.sender === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>

                  {/* 消息体 */}
                  <div className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
                    
                    {/* 文本类消息 */}
                    {msg.type === 'text' && (
                      <div className={`px-4 py-3 rounded-2xl text-[15px] leading-relaxed whitespace-pre-line shadow-sm ${
                        msg.sender === 'user' 
                          ? 'bg-blue-600 text-white rounded-tr-sm' 
                          : 'bg-white text-gray-800 rounded-tl-sm border border-gray-100'
                      }`}>
                        {msg.text}
                      </div>
                    )}

                    {/* 卡片类消息 (模拟系统对接成果) */}
                    {msg.type === 'action_card' && (
                      <div className="bg-white rounded-2xl p-4 shadow-md border border-gray-100 w-full sm:min-w-[320px] rounded-tl-sm">
                        <div className="font-bold text-gray-800 mb-2 border-b border-gray-50 pb-2">{msg.title}</div>
                        <div className="text-sm text-gray-600 mb-4 whitespace-pre-line">
                          {msg.text}
                        </div>
                        <button className="w-full py-2 bg-blue-50 hover:bg-blue-100 text-blue-600 rounded-lg text-sm font-semibold transition-colors flex justify-center items-center gap-1">
                          {msg.actionLabel} <ChevronRight size={16} />
                        </button>
                      </div>
                    )}

                    <span className="text-[11px] text-gray-400 mt-1.5 mx-1">{msg.time}</span>
                  </div>
                </div>
              </div>
            ))}

            {/* 输入中动画 */}
            {isTyping && (
              <div className="flex justify-start">
                <div className="flex gap-3 max-w-[80%]">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 flex items-center justify-center text-white">
                    <Bot size={16} />
                  </div>
                  <div className="bg-white border border-gray-100 px-4 py-4 rounded-2xl rounded-tl-sm flex items-center gap-1 shadow-sm">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 底部输入区 */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="max-w-3xl mx-auto relative flex items-end gap-2">
            <div className="relative flex-1 bg-gray-50 border border-gray-200 rounded-2xl focus-within:border-blue-500 focus-within:bg-white transition-all shadow-inner">
              <textarea 
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="描述你的问题，例如：宿舍空调坏了怎么报修..."
                className="w-full bg-transparent p-4 pr-12 text-sm outline-none resize-none max-h-32 min-h-[56px] rounded-2xl"
                rows="1"
              />
              <button 
                onClick={() => handleSend()}
                disabled={!inputValue.trim() || isTyping}
                className="absolute right-2 bottom-2 w-10 h-10 bg-blue-600 text-white rounded-xl flex items-center justify-center disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors shadow-sm"
              >
                <Send size={18} className={inputValue.trim() ? "translate-x-0.5 -translate-y-0.5" : ""} />
              </button>
            </div>
          </div>
          <div className="text-center mt-2">
            <p className="text-[11px] text-gray-400">系统解答由AI大模型生成，涉及重大转专业、休学等事宜请最终以辅导员确认意见为准。</p>
          </div>
        </div>

      </div>
    </div>
  );
}