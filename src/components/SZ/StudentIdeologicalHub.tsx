import React, { useState, useEffect, useRef } from 'react';
import { 
  Bell, BookOpen, Award, Map, Sparkles, Send, 
  BrainCircuit, User, ArrowRight, PlayCircle, 
  CheckCircle2, Target, MessageSquare, ChevronLeft,
  Medal, Trophy, Star, Shield, Flag, Lock, Zap,
  TrendingUp, History
} from 'lucide-react';

export default function StudentIdeologicalHub() {
  const [activeView, setActiveView] = useState('dashboard'); // 'dashboard', 'learning_case', 'knowledge_graph', 'achievements'
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  // 模拟学生收到的由导师端推送过来的个性化教案
  const pushedTask = {
    title: "新质生产力与青年担当 (计算机专业定制版)",
    teacher: "刘导师",
    time: "10分钟前",
    points: "+50 积分",
    content: "结合近期大模型与国产芯片突破热点，探讨数字经济转型下，工科生如何将个人职业规划与国家科技自立自强战略结合。"
  };

  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      sender: 'ai',
      text: `李明同学你好！我是你的专属 AI 思政导师。刘导师刚刚为你推送了本周的定制学习案例《${pushedTask.title}》。\n\n作为电子信息工程专业的学生，你最近肯定也关注到了 ChatGPT 和国产 AI 芯片的发展。你认为在未来的“新质生产力”浪潮中，我们工科生最大的机遇和挑战是什么？`,
    }
  ]);

  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current && (activeView === 'learning_case')) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isTyping, activeView]);

  const handleSendMessage = () => {
    if (!chatInput.trim()) return;
    
    // 添加学生回复
    setChatMessages(prev => [...prev, { id: Date.now(), sender: 'student', text: chatInput }]);
    setChatInput('');
    setIsTyping(true);

    // 模拟 AI 启发式点评
    setTimeout(() => {
      setChatMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: "你的思考非常深入！确实，单纯的代码编写可能会被 AI 替代（挑战），但底层的算法创新和硬核的芯片设计（机遇）恰恰是国家目前最需要的“硬科技”。\n\n这其实就印证了马克思主义政治经济学中关于“生产力决定生产关系”的论断。恭喜你完成了本次深度互动思考！已为你发放 50 个思政学习积分。🌟",
        isReward: true
      }]);
      setIsTyping(false);
    }, 2000);
  };

  // 渲染头部标题逻辑
  const renderHeaderTitle = () => {
    switch(activeView) {
      case 'dashboard': return '我的学习空间';
      case 'learning_case': return '沉浸式 AI 伴读';
      case 'knowledge_graph': return '思政知识星图';
      case 'achievements': return '我的成就与激励';
      default: return 'AI 思政空间';
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#f8fafc] text-slate-800 font-sans">
      
      {/* 极简侧边栏 */}
      <div className="w-20 md:w-64 bg-white border-r border-slate-200 flex flex-col items-center md:items-stretch shadow-sm z-20 transition-all">
        <div className="h-16 flex items-center justify-center md:justify-start md:px-6 border-b border-slate-100">
          <div className="w-8 h-8 bg-red-600 rounded-lg flex items-center justify-center text-white shadow-md">
            <Target size={18} />
          </div>
          <span className="font-bold text-slate-800 ml-3 hidden md:block">AI 思政空间</span>
        </div>
        
        <nav className="flex-1 py-6 flex flex-col gap-2 px-3">
          <button 
            onClick={() => setActiveView('dashboard')}
            className={`p-3 md:px-4 md:py-3 rounded-xl flex items-center justify-center md:justify-start gap-3 transition-colors ${activeView === 'dashboard' ? 'bg-red-50 text-red-600 font-bold' : 'text-slate-500 hover:bg-slate-50 font-medium'}`}
          >
            <BookOpen size={20} /> <span className="hidden md:block">学习主页</span>
          </button>
          <button 
            onClick={() => setActiveView('knowledge_graph')}
            className={`p-3 md:px-4 md:py-3 rounded-xl flex items-center justify-center md:justify-start gap-3 transition-colors ${activeView === 'knowledge_graph' ? 'bg-red-50 text-red-600 font-bold' : 'text-slate-500 hover:bg-slate-50 font-medium'}`}
          >
            <Map size={20} /> <span className="hidden md:block">知识图谱</span>
          </button>
          <button 
            onClick={() => setActiveView('achievements')}
            className={`p-3 md:px-4 md:py-3 rounded-xl flex items-center justify-center md:justify-start gap-3 transition-colors ${activeView === 'achievements' ? 'bg-red-50 text-red-600 font-bold' : 'text-slate-500 hover:bg-slate-50 font-medium'}`}
          >
            <Award size={20} /> <span className="hidden md:block">我的成就</span>
          </button>
        </nav>

        <div className="p-4 border-t border-slate-100 flex justify-center md:justify-start items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold">李</div>
          <div className="hidden md:block">
            <div className="text-sm font-bold text-slate-800">李明</div>
            <div className="text-[10px] text-slate-500">电子信息工程</div>
          </div>
        </div>
      </div>

      {/* 主内容区 */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        
        {/* 顶栏 */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center justify-between px-6 z-10 sticky top-0">
          <div className="flex items-center gap-2">
            {activeView === 'learning_case' && (
              <button onClick={() => setActiveView('dashboard')} className="mr-2 p-1.5 hover:bg-slate-100 rounded-lg text-slate-500 transition-colors">
                <ChevronLeft size={20} />
              </button>
            )}
            <h1 className="text-lg font-bold text-slate-800">
              {renderHeaderTitle()}
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-amber-50 text-amber-600 rounded-full border border-amber-100 text-sm font-bold shadow-sm">
              <Award size={16} /> 1,250 积分
            </div>
            <button className="relative text-slate-500 hover:text-red-600 transition-colors">
              <Bell size={20} />
              {activeView === 'dashboard' && <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white animate-pulse"></span>}
            </button>
          </div>
        </header>

        {/* ================= 视图 1：学生主页 ================= */}
        {activeView === 'dashboard' && (
          <main className="flex-1 overflow-y-auto p-6 md:p-8">
            <div className="max-w-4xl mx-auto space-y-8">
              <div className="flex items-end justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-slate-800 mb-1">下午好，李明！</h2>
                  <p className="text-sm text-slate-500">本周你已在【历史唯物主义】模块击败了 85% 的同学，继续保持！</p>
                </div>
              </div>

              {/* 导师推送任务卡片 */}
              <div className="bg-gradient-to-br from-indigo-600 to-blue-700 rounded-3xl p-1 shadow-lg relative overflow-hidden animate-in fade-in slide-in-from-bottom-4">
                <div className="absolute right-0 top-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
                <div className="bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] absolute inset-0 opacity-20"></div>
                
                <div className="bg-white/10 backdrop-blur-sm rounded-[22px] p-6 relative z-10 border border-white/20">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-1 bg-red-500 text-white text-[10px] font-bold rounded-md shadow-sm animate-pulse">
                        NEW 导师定制推送
                      </span>
                      <span className="text-xs text-indigo-100 flex items-center gap-1"><User size={12}/> {pushedTask.teacher}</span>
                    </div>
                    <span className="text-amber-300 font-bold text-sm flex items-center gap-1"><Award size={16}/> {pushedTask.points}</span>
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-2">{pushedTask.title}</h3>
                  <p className="text-indigo-100 text-sm leading-relaxed mb-6 max-w-2xl">
                    {pushedTask.content}
                  </p>
                  
                  <button 
                    onClick={() => setActiveView('learning_case')}
                    className="group bg-white text-indigo-700 px-6 py-3 rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-indigo-50 transition-all shadow-md"
                  >
                    立即开启 AI 互动学习 
                    <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>

              {/* 常规学习路径快捷入口 */}
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                    <Map className="text-red-500" size={20}/> 探索知识图谱
                  </h3>
                  <button onClick={() => setActiveView('knowledge_graph')} className="text-sm text-blue-600 font-medium hover:text-blue-800">
                    查看完整图谱 &rarr;
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div onClick={() => setActiveView('knowledge_graph')} className="bg-white border border-slate-200 rounded-2xl p-5 flex gap-4 hover:border-red-300 hover:shadow-md transition-all cursor-pointer">
                    <div className="w-12 h-12 bg-red-50 text-red-600 rounded-xl flex items-center justify-center flex-shrink-0">
                      <BookOpen size={24} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800 mb-1">《资本论》选读：价值规律</h4>
                      <p className="text-xs text-slate-500 mb-3 line-clamp-2">理解商品、价值与劳动的关系，构建马克思主义政治经济学底层思维。</p>
                      <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden"><div className="bg-red-500 h-full w-[100%]"></div></div>
                      <div className="flex justify-between text-[10px] text-slate-400 mt-1"><span>已完成</span><span>100%</span></div>
                    </div>
                  </div>
                  <div onClick={() => setActiveView('knowledge_graph')} className="bg-white border border-slate-200 rounded-2xl p-5 flex gap-4 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer">
                    <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center flex-shrink-0">
                      <PlayCircle size={24} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800 mb-1">科技伦理与人工智能治理</h4>
                      <p className="text-xs text-slate-500 mb-3 line-clamp-2">从马克思主义科技观出发，探讨AI算法偏见等技术伦理问题。</p>
                      <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden"><div className="bg-blue-500 h-full w-[0%]"></div></div>
                      <div className="flex justify-between text-[10px] text-slate-400 mt-1"><span>待学习</span><span>0%</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </main>
        )}

        {/* ================= 视图 2：沉浸式 AI 伴读 ================= */}
        {activeView === 'learning_case' && (
          <div className="flex-1 flex flex-col md:flex-row overflow-hidden bg-white animate-in slide-in-from-right-8">
            <div className="w-full md:w-1/2 border-r border-slate-100 flex flex-col bg-[#fafcff]">
              <div className="p-6 md:p-8 overflow-y-auto flex-1">
                <div className="inline-block px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-bold rounded-full mb-4">
                  刘导师 定制推送
                </div>
                <h2 className="text-2xl font-black text-slate-800 mb-4 leading-tight">{pushedTask.title}</h2>
                <div className="prose prose-sm md:prose-base prose-slate max-w-none text-slate-600 space-y-4">
                  <p className="font-medium text-slate-700">{pushedTask.content}</p>
                  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm my-6">
                    <h4 className="font-bold text-slate-800 mb-2 flex items-center gap-2"><Sparkles className="text-amber-500" size={18}/> 核心知识点关联</h4>
                    <ul className="list-disc pl-5 space-y-1 text-sm text-slate-600">
                      <li>马克思主义政治经济学：生产力与生产关系</li>
                      <li>科技强国战略与创新驱动发展</li>
                      <li>工程伦理与技术工作者的社会责任</li>
                    </ul>
                  </div>
                  <p>
                    当前，以大模型为代表的人工智能技术正在重塑千行百业。作为电子信息工程、软件工程等相关专业的青年学子，不能仅仅将自己定位为“代码的搬运工”，更应站在国家科技自立自强的高度思考问题...
                  </p>
                </div>
              </div>
            </div>

            <div className="w-full md:w-1/2 flex flex-col h-[50vh] md:h-auto bg-slate-50 border-t md:border-t-0 border-slate-200">
              <div className="p-4 bg-white border-b border-slate-100 flex items-center gap-2 shadow-sm z-10">
                <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white">
                  <BrainCircuit size={16} />
                </div>
                <div>
                  <div className="font-bold text-sm text-slate-800">AI 思政伴读助手</div>
                  <div className="text-[10px] text-green-500 flex items-center gap-1"><span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span> 启发式问答进行中</div>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
                {chatMessages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.sender === 'student' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                    <div className={`flex gap-3 max-w-[90%] ${msg.sender === 'student' ? 'flex-row-reverse' : 'flex-row'}`}>
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 ${msg.sender === 'student' ? 'bg-blue-600 text-white' : 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white'}`}>
                        {msg.sender === 'student' ? <User size={16} /> : <BrainCircuit size={16} />}
                      </div>
                      <div className="flex flex-col">
                        <div className={`px-4 py-3 rounded-2xl text-[14px] leading-relaxed whitespace-pre-line shadow-sm ${msg.sender === 'student' ? 'bg-blue-600 text-white rounded-tr-sm' : 'bg-white text-slate-700 rounded-tl-sm border border-slate-200'}`}>
                          {msg.text}
                        </div>
                        {msg.isReward && (
                          <div className="mt-2 flex items-center gap-1 text-xs font-bold text-amber-500 bg-amber-50 px-3 py-1.5 rounded-lg border border-amber-100 self-start">
                            <CheckCircle2 size={14} /> 任务达成，已获得 +50 学习积分
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="flex gap-3 max-w-[80%]">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center text-white"><BrainCircuit size={16} /></div>
                      <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-tl-sm flex items-center gap-1 shadow-sm">
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              <div className="p-4 bg-white border-t border-slate-200">
                <div className="relative flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-2xl focus-within:border-indigo-400 focus-within:bg-white transition-all shadow-inner p-1">
                  <textarea 
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="输入你的思考观点..."
                    className="w-full bg-transparent p-3 pr-12 text-sm outline-none resize-none max-h-32 min-h-[50px] rounded-2xl"
                    rows="2"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                  />
                  <button 
                    onClick={handleSendMessage}
                    disabled={!chatInput.trim() || isTyping}
                    className="absolute right-2 bottom-2 w-9 h-9 bg-indigo-600 text-white rounded-xl flex items-center justify-center disabled:bg-slate-300 hover:bg-indigo-700 transition-colors shadow-sm"
                  >
                    <Send size={16} className={chatInput.trim() ? "translate-x-0.5 -translate-y-0.5" : ""} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================= 视图 3：知识图谱 (技能树模式) ================= */}
        {activeView === 'knowledge_graph' && (
          <main className="flex-1 overflow-y-auto bg-slate-900 animate-in fade-in relative">
            {/* 科技感星空背景 */}
            <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-30"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl aspect-square bg-blue-500/10 rounded-full blur-[100px] pointer-events-none"></div>

            <div className="relative z-10 p-6 md:p-10 max-w-5xl mx-auto h-full flex flex-col">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-black text-white mb-3 tracking-wide">思政知识星图</h2>
                <p className="text-slate-400 text-sm max-w-xl mx-auto">以马克思主义理论为核心，解锁你的思政知识宇宙。点亮节点，获取专属学习任务。</p>
              </div>

              {/* 模拟的技能树/节点关系图 */}
              <div className="flex-1 relative min-h-[500px] flex items-center justify-center">
                
                {/* 伪节点连线 */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                  {/* 中心连左上 */}
                  <path d="M 50% 50% L 25% 30%" stroke="#4ade80" strokeWidth="3" opacity="0.6" strokeDasharray="6,4" className="animate-[dash_10s_linear_infinite]" />
                  {/* 中心连右上 */}
                  <path d="M 50% 50% L 75% 30%" stroke="#3b82f6" strokeWidth="2" opacity="0.3" />
                  {/* 左上连左边 */}
                  <path d="M 25% 30% L 15% 50%" stroke="#4ade80" strokeWidth="2" opacity="0.6" />
                  {/* 右上连右下 */}
                  <path d="M 75% 30% L 85% 60%" stroke="#475569" strokeWidth="2" strokeDasharray="4,4" opacity="0.5" />
                  {/* 中心连正下 */}
                  <path d="M 50% 50% L 50% 80%" stroke="#3b82f6" strokeWidth="3" opacity="0.8" className="animate-pulse" />
                </svg>

                {/* --- 节点 1：核心已掌握 --- */}
                <div className="absolute top-[30%] left-[25%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10">
                  <div className="w-16 h-16 bg-green-500/20 border-2 border-green-500 rounded-full flex items-center justify-center text-green-400 shadow-[0_0_15px_rgba(74,222,128,0.5)] transition-transform group-hover:scale-110">
                    <CheckCircle2 size={24} />
                  </div>
                  <div className="mt-3 px-3 py-1 bg-slate-800/80 backdrop-blur border border-slate-700 rounded-lg text-white text-xs font-bold text-center">
                    历史唯物主义<br/><span className="text-[10px] text-green-400 font-normal">已解锁 100%</span>
                  </div>
                </div>

                {/* --- 节点 2：次核心已掌握 --- */}
                <div className="absolute top-[50%] left-[15%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10">
                  <div className="w-12 h-12 bg-green-500/20 border-2 border-green-500 rounded-full flex items-center justify-center text-green-400 shadow-[0_0_10px_rgba(74,222,128,0.3)] transition-transform group-hover:scale-110">
                    <CheckCircle2 size={18} />
                  </div>
                  <div className="mt-2 px-2 py-1 bg-slate-800/80 backdrop-blur border border-slate-700 rounded-lg text-white text-[10px] font-bold text-center">
                    政治经济学
                  </div>
                </div>

                {/* --- 中心节点：起点 --- */}
                <div className="absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10">
                  <div className="w-24 h-24 bg-gradient-to-br from-red-500 to-rose-600 border-4 border-red-300/50 rounded-full flex items-center justify-center text-white shadow-[0_0_30px_rgba(239,68,68,0.6)] transition-transform group-hover:scale-110">
                    <Star size={36} className="fill-white" />
                  </div>
                  <div className="mt-4 px-4 py-1.5 bg-red-900/80 backdrop-blur border border-red-700 rounded-xl text-white text-sm font-black tracking-widest text-center shadow-lg">
                    马克思主义核心
                  </div>
                </div>

                {/* --- 节点 3：正在学习 (闪烁) --- */}
                <div className="absolute top-[80%] left-[50%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10" onClick={() => setActiveView('learning_case')}>
                  <div className="relative">
                    <div className="absolute inset-0 bg-blue-500 rounded-full animate-ping opacity-20"></div>
                    <div className="w-20 h-20 bg-blue-600/30 border-2 border-blue-400 rounded-full flex items-center justify-center text-blue-300 shadow-[0_0_20px_rgba(59,130,246,0.5)] transition-transform group-hover:scale-110">
                      <Zap size={28} className="animate-pulse" />
                    </div>
                  </div>
                  <div className="mt-3 px-3 py-1 bg-blue-900/80 backdrop-blur border border-blue-700 rounded-lg text-white text-xs font-bold text-center">
                    新质生产力与数字经济<br/><span className="text-[10px] text-blue-300 font-normal">本周导师推荐 · 学习中</span>
                  </div>
                </div>

                {/* --- 节点 4：未解锁 --- */}
                <div className="absolute top-[30%] left-[75%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10">
                  <div className="w-16 h-16 bg-slate-800 border-2 border-slate-600 rounded-full flex items-center justify-center text-slate-500 transition-transform group-hover:scale-105">
                    <Lock size={20} />
                  </div>
                  <div className="mt-3 px-3 py-1 bg-slate-800/50 backdrop-blur border border-slate-700 rounded-lg text-slate-400 text-xs font-bold text-center">
                    科技伦理与AI治理<br/><span className="text-[10px] font-normal">需先完成前置任务</span>
                  </div>
                </div>

                {/* --- 节点 5：未解锁深层 --- */}
                <div className="absolute top-[60%] left-[85%] -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer z-10">
                  <div className="w-12 h-12 bg-slate-800 border-2 border-slate-700 rounded-full flex items-center justify-center text-slate-600">
                    <Lock size={16} />
                  </div>
                  <div className="mt-2 text-slate-500 text-[10px] font-bold">
                    数字鸿沟
                  </div>
                </div>

              </div>
            </div>
          </main>
        )}

        {/* ================= 视图 4：我的成就与激励 (游戏化) ================= */}
        {activeView === 'achievements' && (
          <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50 animate-in fade-in slide-in-from-bottom-4">
            <div className="max-w-5xl mx-auto space-y-8">
              
              {/* 顶部：积分与等级卡片 */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="col-span-1 md:col-span-2 bg-gradient-to-r from-amber-500 to-orange-500 rounded-3xl p-8 text-white shadow-lg relative overflow-hidden flex items-center justify-between">
                  <div className="absolute right-0 bottom-0 opacity-10 translate-x-1/4 translate-y-1/4">
                    <Trophy size={180} />
                  </div>
                  <div className="relative z-10">
                    <p className="text-amber-100 font-medium mb-1 flex items-center gap-2"><Award size={18}/> 当前累计积分</p>
                    <h2 className="text-5xl font-black tracking-tight mb-2">1,250 <span className="text-lg font-normal text-amber-100">Pts</span></h2>
                    <p className="text-sm text-amber-50">距升级至 Lv.5 [理论达人] 还需 250 积分</p>
                    
                    <div className="mt-6 flex gap-3">
                      <button className="px-5 py-2 bg-white text-orange-600 rounded-xl font-bold text-sm hover:bg-amber-50 shadow-sm transition-colors">兑换奖励</button>
                      <button className="px-5 py-2 bg-orange-600/50 border border-orange-400 text-white rounded-xl font-bold text-sm hover:bg-orange-600 transition-colors">积分规则</button>
                    </div>
                  </div>
                  <div className="relative z-10 hidden md:flex flex-col items-center justify-center bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-6 mr-4">
                    <Shield size={48} className="text-white mb-2" />
                    <div className="text-sm font-bold">Lv.4</div>
                    <div className="text-xs text-amber-100">求知学子</div>
                  </div>
                </div>

                {/* 排行榜简述卡片 */}
                <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="font-bold text-slate-800 flex items-center gap-2"><TrendingUp className="text-blue-500" size={18}/> 学院战力榜</h3>
                    <span className="text-xs text-slate-400">本周</span>
                  </div>
                  <div className="flex-1 flex flex-col justify-center gap-4">
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-full bg-amber-100 text-amber-600 font-bold flex items-center justify-center text-sm">1</div>
                      <div className="flex-1 text-sm font-bold text-slate-700">张同学 <span className="text-xs text-slate-400 font-normal">/ 软工</span></div>
                      <div className="text-sm font-bold text-amber-500">2,100</div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-full bg-slate-100 text-slate-500 font-bold flex items-center justify-center text-sm">2</div>
                      <div className="flex-1 text-sm font-bold text-slate-700">陈同学 <span className="text-xs text-slate-400 font-normal">/ 信安</span></div>
                      <div className="text-sm font-bold text-slate-500">1,980</div>
                    </div>
                    <div className="flex items-center gap-4 bg-blue-50 p-2 -mx-2 rounded-lg border border-blue-100">
                      <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-bold flex items-center justify-center text-sm">12</div>
                      <div className="flex-1 text-sm font-bold text-blue-800">我 (李明)</div>
                      <div className="text-sm font-bold text-blue-600">1,250</div>
                    </div>
                  </div>
                  <button className="mt-4 w-full py-2 text-xs font-bold text-slate-500 bg-slate-100 rounded-xl hover:bg-slate-200 transition-colors">查看完整榜单</button>
                </div>
              </div>

              {/* 中间：徽章墙 */}
              <div>
                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <Medal className="text-red-500" size={20}/> 荣誉徽章墙
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {/* 已解锁徽章 */}
                  <div className="bg-white border border-slate-200 rounded-2xl p-5 flex flex-col items-center text-center shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-8 h-8 bg-green-500 text-white flex items-center justify-center rounded-bl-xl font-bold text-xs"><CheckCircle2 size={14}/></div>
                    <div className="w-16 h-16 rounded-full bg-amber-100 text-amber-500 flex items-center justify-center mb-3">
                      <Flag size={32} />
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm">初露锋芒</h4>
                    <p className="text-[10px] text-slate-500 mt-1">首次完成AI伴读</p>
                  </div>
                  
                  <div className="bg-white border border-slate-200 rounded-2xl p-5 flex flex-col items-center text-center shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-8 h-8 bg-green-500 text-white flex items-center justify-center rounded-bl-xl font-bold text-xs"><CheckCircle2 size={14}/></div>
                    <div className="w-16 h-16 rounded-full bg-blue-100 text-blue-500 flex items-center justify-center mb-3">
                      <BookOpen size={32} />
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm">理论先锋</h4>
                    <p className="text-[10px] text-slate-500 mt-1">累计阅读超10篇原著</p>
                  </div>

                  <div className="bg-white border border-slate-200 rounded-2xl p-5 flex flex-col items-center text-center shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-8 h-8 bg-green-500 text-white flex items-center justify-center rounded-bl-xl font-bold text-xs"><CheckCircle2 size={14}/></div>
                    <div className="w-16 h-16 rounded-full bg-red-100 text-red-500 flex items-center justify-center mb-3">
                      <Sparkles size={32} />
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm">深度思考者</h4>
                    <p className="text-[10px] text-slate-500 mt-1">获得3次AI高级评价</p>
                  </div>

                  {/* 未解锁徽章 */}
                  <div className="bg-slate-100/50 border border-slate-200 border-dashed rounded-2xl p-5 flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-slate-200 text-slate-400 flex items-center justify-center mb-3 grayscale">
                      <Target size={32} />
                    </div>
                    <h4 className="font-bold text-slate-500 text-sm">知行合一</h4>
                    <p className="text-[10px] text-slate-400 mt-1">未解锁：需参与一次线下实践</p>
                  </div>

                  <div className="bg-slate-100/50 border border-slate-200 border-dashed rounded-2xl p-5 flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-slate-200 text-slate-400 flex items-center justify-center mb-3 grayscale">
                      <Trophy size={32} />
                    </div>
                    <h4 className="font-bold text-slate-500 text-sm">榜样力量</h4>
                    <p className="text-[10px] text-slate-400 mt-1">未解锁：登顶周榜Top1</p>
                  </div>
                </div>
              </div>

              {/* 底部：积分获取明细 */}
              <div>
                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <History className="text-indigo-500" size={20}/> 近期积分明细
                </h3>
                <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
                  <div className="divide-y divide-slate-100">
                    <div className="p-4 flex justify-between items-center hover:bg-slate-50 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-indigo-50 text-indigo-500 flex items-center justify-center"><BrainCircuit size={18}/></div>
                        <div>
                          <p className="font-bold text-slate-800 text-sm">参与苏格拉底式 AI 互动答疑</p>
                          <p className="text-xs text-slate-400 mt-0.5">今天 14:30</p>
                        </div>
                      </div>
                      <span className="font-bold text-green-500">+50 Pts</span>
                    </div>
                    <div className="p-4 flex justify-between items-center hover:bg-slate-50 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-blue-50 text-blue-500 flex items-center justify-center"><BookOpen size={18}/></div>
                        <div>
                          <p className="font-bold text-slate-800 text-sm">完成《历史唯物主义》第一章学习</p>
                          <p className="text-xs text-slate-400 mt-0.5">昨天 20:15</p>
                        </div>
                      </div>
                      <span className="font-bold text-green-500">+100 Pts</span>
                    </div>
                    <div className="p-4 flex justify-between items-center hover:bg-slate-50 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-orange-50 text-orange-500 flex items-center justify-center"><TrendingUp size={18}/></div>
                        <div>
                          <p className="font-bold text-slate-800 text-sm">连续登录 7 天奖励</p>
                          <p className="text-xs text-slate-400 mt-0.5">昨天 08:00</p>
                        </div>
                      </div>
                      <span className="font-bold text-green-500">+30 Pts</span>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </main>
        )}

      </div>
    </div>
  );
}