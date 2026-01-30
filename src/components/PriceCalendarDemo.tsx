import React, { useState, useEffect, useRef } from 'react';
import { 
  Calendar, 
  Clock, 
  DollarSign, 
  Tag, 
  MessageSquare, 
  Settings, 
  CheckCircle, 
  AlertCircle,
  Zap,
  ChevronRight,
  Database
} from 'lucide-react';

// --- Constants ---
const BASE_PRICE = 5980;

const PRICE_RULES = [
  {
    id: 'early_bird',
    name: '早鸟优惠期',
    type: 'DISCOUNT',
    startTime: '2023-11-01 00:00',
    endTime: '2023-11-10 23:59',
    price: 4980,
    tags: ['限时立减', '赠送教材'],
    script: '现在是早鸟优惠期，立减 1000 元，到手仅需 4980！还额外赠送全套实体书哦。',
    color: 'bg-blue-100 text-blue-700 border-blue-200'
  },
  {
    id: 'double_11',
    name: '双11 疯抢秒杀',
    type: 'FLASH_SALE',
    startTime: '2023-11-11 00:00',
    endTime: '2023-11-11 23:59',
    price: 3999,
    tags: ['全年底价', '分期免息'],
    script: '您赶上了双11 全年底价！仅限今天 3999 元（原价5980），而且支持 12 期免息，每天只要一杯奶茶钱！',
    color: 'bg-red-100 text-red-700 border-red-200'
  },
  {
    id: 'normal',
    name: '日常销售期',
    type: 'NORMAL',
    startTime: '2023-11-12 00:00',
    endTime: '2023-11-30 23:59',
    price: 5980,
    tags: ['正价', '可申请分期'],
    script: '目前课程恢复正价 5980 元。不过如果您预算紧张，我可以帮您申请分期付款，或者留意下个月的活动。',
    color: 'bg-slate-100 text-slate-600 border-slate-200'
  }
];

export default function PriceCalendarDemo() {
  // --- State ---
  const [currentTime, setCurrentTime] = useState(new Date('2023-11-05T10:00:00')); // Default: Early Bird
  const [activeRule, setActiveRule] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);

  const chatRef = useRef(null);

  // --- Logic ---
  useEffect(() => {
    // 1. Find matching rule based on currentTime
    const match = PRICE_RULES.find(rule => {
      const start = new Date(rule.startTime);
      const end = new Date(rule.endTime);
      return currentTime >= start && currentTime <= end;
    });
    setActiveRule(match || PRICE_RULES[2]); // Default to Normal if no match (simplified)
  }, [currentTime]);

  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const addLog = (msg) => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg }, ...prev]);
  };

  const handleUserQuery = async () => {
    if (isProcessing) return;
    setIsProcessing(true);
    
    // 1. User asks
    setChatHistory(prev => [...prev, { role: 'user', text: '现在买多少钱？有什么优惠吗？' }]);
    
    // 2. AI "Thinks" (Simulate backend lookup)
    await new Promise(r => setTimeout(r, 600));
    addLog(`🔍 查询价格日历: Timestamp [${currentTime.toLocaleString()}]`);
    
    await new Promise(r => setTimeout(r, 400));
    addLog(`✅ 命中策略: ID=[${activeRule.id}] Price=[${activeRule.price}]`);

    // 3. AI Replies
    await new Promise(r => setTimeout(r, 600));
    setChatHistory(prev => [...prev, { 
      role: 'ai', 
      text: activeRule.script,
      priceTag: activeRule.price,
      ruleName: activeRule.name
    }]);

    setIsProcessing(false);
  };

  const changeDate = (dateStr) => {
    const newDate = new Date(dateStr);
    setCurrentTime(newDate);
    addLog(`🕒 系统时间跳跃至: ${newDate.toLocaleString()}`);
    setChatHistory([]); // Clear chat on time jump for clarity
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Calendar className="text-indigo-600" />
              动态价格日历 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">P0 Core</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于时间轴的自动化定价引擎 • 确保 AI 报价严谨合规
            </p>
          </div>
          
          {/* Time Controller */}
          <div className="flex items-center gap-3 bg-slate-100 p-2 rounded-xl border border-slate-200">
            <Clock className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-mono font-bold text-indigo-700 min-w-[160px]">
              {currentTime.toLocaleString('zh-CN', { month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[700px]">
          
          {/* Left: Calendar Configuration */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <span className="font-bold text-slate-700 flex items-center gap-2">
                  <Settings className="w-4 h-4" /> 价格策略配置
                </span>
              </div>
              
              <div className="p-4 space-y-4 overflow-y-auto flex-1">
                {PRICE_RULES.map((rule) => {
                  const isActive = activeRule?.id === rule.id;
                  return (
                    <div 
                      key={rule.id}
                      onClick={() => changeDate(rule.startTime)}
                      className={`
                        relative p-4 rounded-xl border-2 transition-all cursor-pointer group
                        ${isActive ? `${rule.color} shadow-md scale-105 z-10` : 'bg-white border-slate-100 text-slate-400 hover:border-indigo-200'}
                      `}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-bold text-sm flex items-center gap-2">
                           {rule.name}
                           {isActive && <span className="text-[10px] bg-white/50 px-1.5 rounded animate-pulse">ACTIVE</span>}
                        </div>
                        <div className="text-lg font-black font-mono">¥{rule.price}</div>
                      </div>
                      
                      <div className="text-xs space-y-1 opacity-90 font-mono">
                         <div className="flex items-center gap-1">
                           <Clock className="w-3 h-3" /> {rule.startTime.split(' ')[0]} ~ {rule.endTime.split(' ')[0]}
                         </div>
                      </div>

                      <div className="mt-3 flex gap-2 flex-wrap">
                        {rule.tags.map(tag => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded border border-current opacity-70">
                            {tag}
                          </span>
                        ))}
                      </div>

                      {/* Hover hint */}
                      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                         <button className="text-[10px] bg-indigo-600 text-white px-2 py-1 rounded">
                           Jump to Date
                         </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="p-4 bg-slate-50 text-xs text-slate-400 border-t border-slate-100">
                提示：点击上方卡片可快速“穿越”到该时间段进行测试。
              </div>
            </div>
          </div>

          {/* Middle: Logic & Logs */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
             
             {/* Active Status Display */}
             <div className="bg-slate-900 text-white p-6 rounded-xl shadow-lg border border-slate-800 relative overflow-hidden">
                <div className="relative z-10">
                  <h2 className="text-xs font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                    <Database className="w-4 h-4" /> 实时生效价格 (Atomic Fact)
                  </h2>
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-light text-slate-400">¥</span>
                    <span className="text-5xl font-black text-white tracking-tight">{activeRule?.price}</span>
                  </div>
                  <div className="mt-4 flex items-center gap-2">
                     <span className={`px-2 py-0.5 rounded text-xs font-bold ${activeRule?.id === 'double_11' ? 'bg-red-500' : 'bg-blue-500'}`}>
                       {activeRule?.type}
                     </span>
                     <span className="text-sm text-slate-300">
                       策略: {activeRule?.name}
                     </span>
                  </div>
                </div>
                
                {/* Visual Flair */}
                <div className="absolute -right-6 -bottom-6 opacity-10">
                   <DollarSign className="w-32 h-32" />
                </div>
             </div>

             {/* System Logs */}
             <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
                <div className="p-3 border-b border-slate-100 bg-slate-50/50">
                  <span className="text-xs font-bold text-slate-500 uppercase flex items-center gap-2">
                    <Zap className="w-3 h-3" /> 逻辑执行日志
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-2 bg-slate-50/30">
                  {logs.length === 0 && <span className="text-slate-400 italic">等待查询请求...</span>}
                  {logs.map((log, i) => (
                    <div key={i} className="flex gap-2 text-slate-600 animate-in slide-in-from-left-2">
                      <span className="text-slate-400">[{log.time}]</span>
                      <span>{log.msg}</span>
                    </div>
                  ))}
                </div>
             </div>
          </div>

          {/* Right: AI Chat Preview */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col relative overflow-hidden">
               <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center z-10">
                 <h2 className="font-bold text-slate-700 flex items-center gap-2">
                   <MessageSquare className="w-4 h-4" /> 用户咨询窗口
                 </h2>
               </div>

               {/* Chat Area */}
               <div className="flex-1 bg-slate-50 p-4 overflow-y-auto space-y-4 z-10">
                 {chatHistory.length === 0 && (
                   <div className="text-center text-slate-400 mt-20 text-sm">
                     <p>请点击下方按钮，询问价格。</p>
                     <p className="text-xs mt-2 opacity-70">观察 AI 回复如何随左侧时间变化。</p>
                   </div>
                 )}
                 {chatHistory.map((msg, idx) => (
                   <div key={idx} className={`flex flex-col ${msg.role === 'ai' ? 'items-start' : 'items-end'}`}>
                     {msg.role === 'ai' && (
                       <span className="text-[10px] text-slate-400 mb-1 ml-1 flex items-center gap-1">
                         <Tag className="w-3 h-3" /> 基于策略: {msg.ruleName}
                       </span>
                     )}
                     <div className={`
                       max-w-[90%] p-3 rounded-2xl text-sm leading-relaxed shadow-sm
                       ${msg.role === 'ai' 
                         ? 'bg-white border border-slate-200 text-slate-700 rounded-tl-none' 
                         : 'bg-indigo-600 text-white rounded-tr-none'}
                     `}>
                       {msg.text}
                     </div>
                   </div>
                 ))}
                 {isProcessing && (
                   <div className="flex items-start">
                     <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none shadow-sm w-12 flex justify-center">
                        <span className="text-xs text-slate-400 animate-pulse">...</span>
                     </div>
                   </div>
                 )}
                 <div ref={chatRef} />
               </div>
               
               {/* User Simulator */}
               <div className="p-4 bg-white border-t border-slate-100 z-10">
                 <button 
                   onClick={handleUserQuery}
                   disabled={isProcessing}
                   className="w-full py-3 rounded-xl bg-indigo-600 text-white text-sm font-bold hover:bg-indigo-700 shadow-lg shadow-indigo-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                 >
                   <MessageSquare className="w-4 h-4" /> 模拟提问: "现在买多少钱？"
                 </button>
               </div>

            </div>
          </div>

        </div>
      </div>
    </div>
  );
}