import React, { useState, useEffect, useRef } from 'react';
import { 
  Zap, 
  Activity, 
  ShoppingCart, 
  Wifi, 
  CreditCard, 
  AlertTriangle, 
  Radio, 
  MessageSquare,
  Siren,
  Clock,
  CheckCircle,
  XCircle,
  TrendingDown
} from 'lucide-react';

// --- Configuration ---
const EVENTS = {
  INVENTORY_LOW: {
    id: 'EVT_INV_LOW',
    type: 'URGENCY',
    trigger: 'Stock <= 3',
    scriptName: '🔥 库存逼单剧本',
    color: 'bg-red-500',
    icon: <ShoppingCart className="w-5 h-5 text-white" />,
    message: '【系统急报】老师刚通知，本期训练营名额只剩最后 1 个了！系统即将关闭报名通道，现在付款能锁住优惠，手慢无！'
  },
  LIVE_LAG: {
    id: 'EVT_LIVE_LAG',
    type: 'CRISIS',
    trigger: 'Latency > 5000ms',
    scriptName: '🙏 安抚与补偿剧本',
    color: 'bg-orange-500',
    icon: <Wifi className="w-5 h-5 text-white" />,
    message: '非常抱歉！监测到直播间信号有点波动 😖。技术小哥正在紧急修复！您可以先点击这个备用链接观看图文版，稍后我们会在群里补发高清录播。'
  },
  PAYMENT_FAIL: {
    id: 'EVT_PAY_FAIL',
    type: 'TRANSACTION',
    trigger: 'Webhook: Insufficient Balance',
    scriptName: '💳 支付挽回剧本',
    color: 'bg-blue-500',
    icon: <CreditCard className="w-5 h-5 text-white" />,
    message: '检测到您的支付未成功，是花呗额度不够吗？没关系的，我们支持“组合支付”或者“分3期”免息。点这个专属码试试？'
  },
  NORMAL: {
    id: 'NORMAL',
    type: 'NORMAL',
    trigger: 'None',
    scriptName: '💬 常规答疑剧本',
    color: 'bg-slate-500',
    icon: <MessageSquare className="w-5 h-5 text-white" />,
    message: '我们的课程主要涵盖 Python 基础和 AI Agent 实战，非常适合新手入门。您还有什么具体想了解的吗？'
  }
};

export default function EventRoutingDemo() {
  // --- State ---
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: '同学你好，欢迎来到子午线 AI 实战营！今天直播间讲的干货都听懂了吗？', type: 'NORMAL' }
  ]);
  const [activeEvent, setActiveEvent] = useState(null);
  const [inventory, setInventory] = useState(50);
  const [liveLatency, setLiveLatency] = useState(45); // ms
  const [isProcessing, setIsProcessing] = useState(false);
  const [monitoringLogs, setMonitoringLogs] = useState([]);
  
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // --- Real-time Monitoring Simulation ---
  useEffect(() => {
    const interval = setInterval(() => {
      // Fluctuate latency slightly to look real
      if (activeEvent?.id !== 'EVT_LIVE_LAG') {
        setLiveLatency(prev => Math.max(20, Math.min(100, prev + (Math.random() * 20 - 10))));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [activeEvent]);

  // --- Logic ---
  const addLog = (msg, type = 'info') => {
    setMonitoringLogs(prev => [{ time: new Date().toLocaleTimeString(), msg, type }, ...prev].slice(0, 5));
  };

  const triggerEvent = async (eventKey) => {
    if (isProcessing) return;
    setIsProcessing(true);
    
    const event = EVENTS[eventKey];
    setActiveEvent(event);

    // 1. Simulate Signal Detection
    addLog(`⚠️ SIGNAL DETECTED: [${event.trigger}]`, 'warning');
    
    // 2. Routing Decision
    await new Promise(r => setTimeout(r, 600));
    addLog(`⚡️ ROUTING INTERRUPT: Switching to [${event.scriptName}]`, 'critical');

    // 3. AI Execution
    await new Promise(r => setTimeout(r, 600));
    setChatHistory(prev => [...prev, { 
      role: 'ai', 
      text: event.message, 
      type: event.type,
      scriptName: event.scriptName 
    }]);

    setIsProcessing(false);
    
    // Auto-reset active event visual after a few seconds
    setTimeout(() => setActiveEvent(null), 3000);
  };

  const handleUserMessage = async () => {
    if (isProcessing) return;
    setIsProcessing(true);
    setChatHistory(prev => [...prev, { role: 'user', text: '还在吗？我在犹豫要不要买...' }]);
    
    await new Promise(r => setTimeout(r, 800));
    
    // If inventory is critical, standard reply is overridden by Urgency logic
    if (inventory <= 3) {
      const event = EVENTS.INVENTORY_LOW;
      addLog(`🛡️ CONTEXT CHECK: Inventory Critical (${inventory})`, 'warning');
      setChatHistory(prev => [...prev, { role: 'ai', text: event.message, type: event.type, scriptName: event.scriptName }]);
    } else {
      setChatHistory(prev => [...prev, { role: 'ai', text: EVENTS.NORMAL.message, type: 'NORMAL', scriptName: EVENTS.NORMAL.scriptName }]);
    }
    
    setIsProcessing(false);
  };

  const adjustInventory = (val) => {
    setInventory(val);
    if (val <= 3) {
      triggerEvent('INVENTORY_LOW');
    } else {
      addLog(`📦 Inventory Updated: ${val}`, 'info');
    }
  };

  const simulateLag = () => {
    setLiveLatency(8000); // Spike to 8s
    triggerEvent('LIVE_LAG');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Siren className="text-red-600" />
              事件策略路由 <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full uppercase tracking-wide">P0 Core / Real-time</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于业务信号的实时应激反应 • 秒级抓住转化窗口期
            </p>
          </div>
          <div className="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-600">
            <Radio className="w-3 h-3 animate-pulse text-green-500" />
            System Monitoring Active
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[800px] lg:h-[700px]">
          
          {/* Left: Event Control Center */}
          <div className="lg:col-span-3 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50/50">
                <h2 className="font-bold text-slate-700 flex items-center gap-2">
                  <Activity className="w-4 h-4" /> 业务信号发生器
                </h2>
              </div>
              
              <div className="p-4 space-y-6 flex-1 overflow-y-auto">
                
                {/* 1. Inventory Control */}
                <div className="space-y-3 bg-slate-50 p-3 rounded-xl border border-slate-100">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold uppercase text-slate-500 flex items-center gap-1">
                      <ShoppingCart className="w-3 h-3" /> 库存监控
                    </span>
                    <span className={`text-xs font-mono font-bold ${inventory <= 3 ? 'text-red-600 animate-pulse' : 'text-slate-700'}`}>
                      Count: {inventory}
                    </span>
                  </div>
                  <input 
                    type="range" min="1" max="50" value={inventory} 
                    onChange={(e) => adjustInventory(Number(e.target.value))}
                    className={`w-full h-2 rounded-lg appearance-none cursor-pointer ${inventory <= 3 ? 'bg-red-200 accent-red-600' : 'bg-slate-200 accent-slate-600'}`}
                  />
                  <div className="flex gap-2">
                    <button onClick={() => adjustInventory(50)} className="flex-1 py-1 text-[10px] bg-white border border-slate-200 rounded hover:bg-slate-50">Reset (50)</button>
                    <button onClick={() => adjustInventory(1)} className="flex-1 py-1 text-[10px] bg-red-100 text-red-700 border border-red-200 rounded hover:bg-red-200 font-bold">Panic (1)</button>
                  </div>
                </div>

                {/* 2. Live Stream Status */}
                <div className="space-y-3 bg-slate-50 p-3 rounded-xl border border-slate-100">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold uppercase text-slate-500 flex items-center gap-1">
                      <Wifi className="w-3 h-3" /> 直播推流监控
                    </span>
                    <span className={`text-xs font-mono font-bold ${liveLatency > 1000 ? 'text-orange-600' : 'text-green-600'}`}>
                      {liveLatency}ms
                    </span>
                  </div>
                  <div className="h-16 bg-slate-900 rounded-lg relative overflow-hidden flex items-end px-1 gap-0.5">
                    {/* Fake visualizer */}
                    {[...Array(10)].map((_, i) => (
                      <div key={i} className={`w-full transition-all duration-300 ${liveLatency > 1000 ? 'bg-orange-500' : 'bg-green-500'}`} style={{height: `${Math.random() * 100}%`}}></div>
                    ))}
                  </div>
                  <button 
                    onClick={simulateLag}
                    disabled={isProcessing}
                    className="w-full py-2 bg-orange-100 text-orange-700 border border-orange-200 rounded-lg text-xs font-bold hover:bg-orange-200 flex items-center justify-center gap-2"
                  >
                    <AlertTriangle className="w-3 h-3" /> 模拟卡顿/断流
                  </button>
                </div>

                {/* 3. Transaction Status */}
                <div className="space-y-3 bg-slate-50 p-3 rounded-xl border border-slate-100">
                   <div className="text-xs font-bold uppercase text-slate-500 flex items-center gap-1 mb-2">
                      <CreditCard className="w-3 h-3" /> 支付网关回调
                   </div>
                   <button 
                    onClick={() => triggerEvent('PAYMENT_FAIL')}
                    disabled={isProcessing}
                    className="w-full py-2 bg-blue-100 text-blue-700 border border-blue-200 rounded-lg text-xs font-bold hover:bg-blue-200 flex items-center justify-center gap-2"
                  >
                    <XCircle className="w-3 h-3" /> 模拟分期支付失败
                  </button>
                </div>

              </div>
            </div>
          </div>

          {/* Middle: Routing Logic Visualizer */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            <div className="bg-slate-900 text-slate-300 rounded-xl p-6 shadow-xl border border-slate-800 h-full flex flex-col relative overflow-hidden">
               {/* Background Grid */}
               <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none"></div>

               <div className="flex justify-between items-center mb-8 relative z-10">
                 <h2 className="font-bold text-white flex items-center gap-2">
                   <Zap className="w-5 h-5 text-yellow-400" />
                   事件策略路由引擎
                 </h2>
                 <div className="flex items-center gap-2">
                   <div className={`w-2 h-2 rounded-full ${activeEvent ? 'bg-red-500 animate-ping' : 'bg-green-500'}`}></div>
                   <span className="text-xs font-mono">{activeEvent ? 'INTERRUPT ACTIVE' : 'LISTENING...'}</span>
                 </div>
               </div>

               {/* Pipeline Visualization */}
               <div className="flex-1 flex flex-col items-center justify-center relative z-10 space-y-6">
                 
                 {/* Input Pipe */}
                 <div className="w-48 h-12 border-2 border-slate-600 rounded-lg flex items-center justify-center text-xs font-bold bg-slate-800">
                   Event Listeners
                 </div>
                 
                 <TrendingDown className="w-6 h-6 text-slate-500" />

                 {/* The Router Logic */}
                 <div className={`
                   w-64 p-4 rounded-xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-2
                   ${activeEvent ? 'border-red-500 bg-red-500/10 shadow-[0_0_30px_rgba(239,68,68,0.3)]' : 'border-slate-600 bg-slate-800'}
                 `}>
                   <div className="text-sm font-bold text-white">DECISION NODE</div>
                   {activeEvent ? (
                     <>
                       <div className="text-xs text-red-400 animate-pulse font-mono font-bold">MATCH: {activeEvent.trigger}</div>
                       <div className="text-[10px] text-slate-400">Priority: P0 (Critical)</div>
                     </>
                   ) : (
                     <div className="text-xs text-slate-500 font-mono">Status: Normal Flow</div>
                   )}
                 </div>

                 <TrendingDown className={`w-6 h-6 transition-all ${activeEvent ? 'text-red-500 scale-125' : 'text-slate-500'}`} />

                 {/* Output Pipe */}
                 <div className={`
                   w-64 p-3 rounded-lg border flex items-center justify-center gap-2 transition-all duration-500
                   ${activeEvent ? `${activeEvent.color} text-white border-transparent scale-105 font-bold` : 'border-slate-600 bg-slate-800 text-slate-400 text-xs'}
                 `}>
                    {activeEvent ? (
                      <>
                        {activeEvent.icon}
                        加载剧本: {activeEvent.scriptName}
                      </>
                    ) : (
                      "加载剧本: 常规对话逻辑"
                    )}
                 </div>

               </div>

               {/* Console Logs */}
               <div className="mt-8 bg-black/50 rounded-lg p-3 font-mono text-[10px] h-32 overflow-y-auto border border-white/5 relative z-10">
                 <div className="sticky top-0 bg-black/0 text-slate-500 font-bold mb-1 border-b border-white/10 pb-1">KERNEL LOGS</div>
                 {monitoringLogs.map((log, i) => (
                   <div key={i} className={`mb-1 ${log.type === 'critical' ? 'text-red-400 font-bold' : log.type === 'warning' ? 'text-yellow-400' : 'text-slate-400'}`}>
                     <span className="opacity-50">[{log.time}]</span> {log.msg}
                   </div>
                 ))}
               </div>

            </div>
          </div>

          {/* Right: AI Execution Preview */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col relative overflow-hidden">
               <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center z-10">
                 <h2 className="font-bold text-slate-700 flex items-center gap-2">
                   <MessageSquare className="w-4 h-4" /> 企微对话窗口
                 </h2>
                 {activeEvent && (
                   <span className="text-[10px] bg-red-100 text-red-600 px-2 py-1 rounded font-bold animate-pulse">
                     ⚠️ 紧急干预中
                   </span>
                 )}
               </div>

               {/* Chat Area */}
               <div className="flex-1 bg-slate-50 p-3 overflow-y-auto space-y-3 z-10">
                 {chatHistory.map((msg, idx) => (
                   <div key={idx} className={`flex flex-col ${msg.role === 'ai' ? 'items-start' : 'items-end'}`}>
                     {msg.role === 'ai' && msg.type !== 'NORMAL' && (
                       <div className={`mb-1 ml-1 text-[9px] px-1.5 py-0.5 rounded font-bold text-white w-fit flex items-center gap-1 ${EVENTS[Object.keys(EVENTS).find(k => EVENTS[k].scriptName === msg.scriptName)]?.color || 'bg-slate-400'}`}>
                         <Zap className="w-3 h-3" /> {msg.scriptName}
                       </div>
                     )}
                     <div className={`
                       max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed shadow-sm
                       ${msg.role === 'ai' 
                         ? `bg-white border border-slate-200 text-slate-700 rounded-tl-none ${msg.type !== 'NORMAL' ? 'border-l-4 border-l-red-500' : ''}` 
                         : 'bg-indigo-600 text-white rounded-tr-none'}
                     `}>
                       {msg.text}
                     </div>
                   </div>
                 ))}
                 {isProcessing && (
                   <div className="flex items-start">
                     <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none shadow-sm w-12 flex justify-center">
                       <div className="flex gap-1">
                         <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                         <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                         <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                       </div>
                     </div>
                   </div>
                 )}
                 <div ref={chatEndRef} />
               </div>
               
               {/* User Simulator */}
               <div className="p-3 bg-white border-t border-slate-100 z-10">
                 <button 
                   onClick={handleUserMessage}
                   disabled={isProcessing}
                   className="w-full py-3 rounded-xl border border-dashed border-slate-300 text-slate-500 text-xs font-bold hover:bg-slate-50 hover:text-slate-700 hover:border-slate-400 transition-all"
                 >
                   模拟用户发送: "还在吗？我在犹豫..."
                 </button>
               </div>

            </div>
          </div>

        </div>
      </div>
    </div>
  );
}