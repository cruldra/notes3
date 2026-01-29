import React, { useState, useEffect, useRef } from 'react';
import { 
  GitBranch, 
  User, 
  MessageSquare, 
  PlayCircle, 
  CheckCircle, 
  CreditCard, 
  Repeat, 
  ArrowRight, 
  Settings,
  Zap,
  Activity,
  Terminal,
  Database
} from 'lucide-react';

// --- 核心配置：阶段与策略剧本 ---
// 模拟 Coze 知识库或配置表中的数据
const STAGE_CONFIG = {
  NEW_LEAD: {
    id: 'NEW_LEAD',
    label: '新线索 (New Lead)',
    color: 'bg-gray-100 text-gray-600 border-gray-300',
    icon: <User className="w-5 h-5" />,
    scriptName: '🧊 破冰信任剧本',
    aiPrompt: '用户刚加微信，未建立信任。策略：自我介绍，发送问卷，语气热情但不骚扰。',
    exampleResponse: '你好呀！我是你的专属助教子君。👋 很高兴认识你！为了更好地安排学习计划，能麻烦花 1 分钟填个小问卷吗？'
  },
  ATTENDED: {
    id: 'ATTENDED',
    label: '已到课 (Attended)',
    color: 'bg-blue-100 text-blue-600 border-blue-300',
    icon: <PlayCircle className="w-5 h-5" />,
    scriptName: '📚 课程价值剧本',
    aiPrompt: '用户正在看直播或刚看完。策略：回顾课程亮点，激发讨论，引导完成作业。',
    exampleResponse: '刚才老师讲的“AI 变现三部曲”你听到了吗？那个部分真是太精彩了！对你的副业方向有启发吗？🤔'
  },
  CHECKED_IN: {
    id: 'CHECKED_IN',
    label: '已打卡 (Checked-in)',
    color: 'bg-orange-100 text-orange-600 border-orange-300',
    icon: <CheckCircle className="w-5 h-5" />,
    scriptName: '🔥 高意向逼单剧本',
    aiPrompt: '用户已完成作业，意向极高。策略：肯定执行力，抛出限时优惠，进行价值锚定。',
    exampleResponse: '看到你刚刚提交了作业，执行力太强了！🌟 跟你同步个消息，现在的早鸟优惠名额只剩最后 3 个了，建议现在锁定。'
  },
  PAID: {
    id: 'PAID',
    label: '已成交 (Transaction)',
    color: 'bg-green-100 text-green-600 border-green-300',
    icon: <CreditCard className="w-5 h-5" />,
    scriptName: '🤝 服务与关怀剧本',
    aiPrompt: '用户已付费。策略：发送入学通知，提供情绪价值，停止营销动作。',
    exampleResponse: '恭喜你正式入学！🎉 这是你的班主任微信 [QRCode]，快去添加领取入学大礼包吧！未来的路我们一起走！'
  },
  REPURCHASE: {
    id: 'REPURCHASE',
    label: '复购期 (Repurchase)',
    color: 'bg-purple-100 text-purple-600 border-purple-300',
    icon: <Repeat className="w-5 h-5" />,
    scriptName: '💎 VIP 升单剧本',
    aiPrompt: '老学员，已结课。策略：推荐进阶高阶课，强调老学员专属权益。',
    exampleResponse: '子君发现你最近的学习势头很猛！针对老学员，我们要开一个“高阶实战营”，只有内部名额，感兴趣看看吗？'
  }
};

export default function StageRoutingDemo() {
  // --- State ---
  const [currentStage, setCurrentStage] = useState('NEW_LEAD');
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: STAGE_CONFIG.NEW_LEAD.exampleResponse, stage: 'NEW_LEAD' }
  ]);
  const [systemLogs, setSystemLogs] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeEvent, setActiveEvent] = useState(null);

  const logsContainerRef = useRef(null);
  const chatContainerRef = useRef(null);

  // --- Auto-scroll Logs ---
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [systemLogs]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // --- Actions ---

  const addLog = (msg, type = 'info') => {
    const uniqueId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setSystemLogs(prev => [...prev, { id: uniqueId, time: new Date().toLocaleTimeString(), msg, type }]);
  };

  const handleUserEvent = async (event, targetStage) => {
    if (isAnimating) return;
    setIsAnimating(true);
    setActiveEvent(event);

    // 1. Log Event
    addLog(`收到用户事件: [${event}]`, 'event');
    
    // Simulate processing delay
    await new Promise(r => setTimeout(r, 600));

    // 2. State Change Logic
    if (currentStage === targetStage) {
      addLog(`状态未变更，保持在 [${STAGE_CONFIG[currentStage].label}]`, 'warning');
      setIsAnimating(false);
      setActiveEvent(null);
      return;
    }

    addLog(`检测到状态变更: ${STAGE_CONFIG[currentStage].id} -> ${targetStage}`, 'process');
    setCurrentStage(targetStage);

    // 3. Routing Logic (The Core)
    await new Promise(r => setTimeout(r, 600));
    const config = STAGE_CONFIG[targetStage];
    addLog(`策略路由命中: 加载剧本 [${config.scriptName}]`, 'success');
    
    // 4. Generate AI Response
    await new Promise(r => setTimeout(r, 800));
    setChatHistory(prev => [
      ...prev,
      { role: 'event', text: `用户触发：${event}` },
      { role: 'ai', text: config.exampleResponse, stage: targetStage }
    ]);

    setIsAnimating(false);
    setActiveEvent(null);
  };

  const clearDemo = () => {
    setCurrentStage('NEW_LEAD');
    setChatHistory([{ role: 'ai', text: STAGE_CONFIG.NEW_LEAD.exampleResponse, stage: 'NEW_LEAD' }]);
    setSystemLogs([]);
    addLog('系统重置完成', 'info');
  };

  // --- Render Components ---

  const StageCard = ({ stageKey }) => {
    const config = STAGE_CONFIG[stageKey];
    const isActive = currentStage === stageKey;
    
    return (
      <div className={`
        relative p-4 rounded-xl border-2 transition-all duration-500 flex flex-col gap-2
        ${isActive 
          ? `${config.color.split(' ')[0]} border-indigo-500 shadow-lg scale-105 z-10` 
          : 'bg-white border-gray-100 opacity-60 grayscale-[0.5]'}
      `}>
        {isActive && (
          <div className="absolute -top-3 left-4 bg-indigo-600 text-white text-[10px] px-2 py-0.5 rounded-full font-bold tracking-wider animate-bounce">
            CURRENT STAGE
          </div>
        )}
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 font-bold text-gray-800">
            {config.icon}
            <span className="text-sm">{config.label.split('(')[0]}</span>
          </div>
          {isActive && <Activity className="w-4 h-4 text-indigo-600 animate-pulse" />}
        </div>
        
        {isActive && (
          <div className="mt-2 text-xs bg-white/50 p-2 rounded border border-indigo-100 animate-in fade-in slide-in-from-left-2">
            <div className="flex items-center gap-1 text-indigo-700 font-semibold mb-1">
              <Settings className="w-3 h-3" />
              <span>当前执行策略：</span>
            </div>
            <div className="font-mono text-indigo-900">{config.scriptName}</div>
            <p className="mt-1 text-gray-500 leading-relaxed">{config.aiPrompt}</p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <GitBranch className="text-indigo-600" />
              阶段策略路由引擎 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Stage Strategy Routing</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于用户生命周期 (Lifecycle) 的自动化剧本切换演示 • <span className="text-indigo-600 font-medium">Coze 工作流核心逻辑</span>
            </p>
          </div>
          <button 
            onClick={clearDemo}
            className="px-4 py-2 bg-slate-100 text-slate-600 text-sm font-medium rounded-lg hover:bg-slate-200 transition-colors flex items-center gap-2"
          >
            <Repeat className="w-4 h-4" /> 重置演示
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[800px] lg:h-[700px]">
          
          {/* Left Column: Event Simulator */}
          <div className="lg:col-span-3 space-y-4">
            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
              <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4" /> 用户事件触发器
              </h2>
              
              <div className="space-y-3 flex-1">
                <p className="text-xs text-slate-500 mb-2">点击下方按钮模拟用户真实行为，观察右侧系统响应。</p>
                
                <button
                  onClick={() => handleUserEvent('进入直播间 (Live_Enter)', 'ATTENDED')}
                  disabled={isAnimating}
                  className={`w-full p-3 rounded-lg border text-left transition-all flex items-center gap-3 group
                    ${currentStage === 'NEW_LEAD' ? 'bg-blue-50 border-blue-200 hover:bg-blue-100' : 'bg-slate-50 border-slate-100 opacity-50'}
                  `}
                >
                  <div className="bg-blue-100 text-blue-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
                    <PlayCircle className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-700">进入直播间</div>
                    <div className="text-[10px] text-slate-500">Trigger: Attended</div>
                  </div>
                </button>

                <button
                  onClick={() => handleUserEvent('提交作业 (Submit_HW)', 'CHECKED_IN')}
                  disabled={isAnimating}
                  className={`w-full p-3 rounded-lg border text-left transition-all flex items-center gap-3 group
                    ${['ATTENDED', 'NEW_LEAD'].includes(currentStage) ? 'bg-orange-50 border-orange-200 hover:bg-orange-100' : 'bg-slate-50 border-slate-100 opacity-50'}
                  `}
                >
                  <div className="bg-orange-100 text-orange-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-700">完成打卡</div>
                    <div className="text-[10px] text-slate-500">Trigger: Checked_In</div>
                  </div>
                </button>

                <button
                  onClick={() => handleUserEvent('支付成功 (Payment_Success)', 'PAID')}
                  disabled={isAnimating}
                  className={`w-full p-3 rounded-lg border text-left transition-all flex items-center gap-3 group
                    ${['CHECKED_IN', 'ATTENDED'].includes(currentStage) ? 'bg-green-50 border-green-200 hover:bg-green-100' : 'bg-slate-50 border-slate-100 opacity-50'}
                  `}
                >
                  <div className="bg-green-100 text-green-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
                    <CreditCard className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-700">支付订单</div>
                    <div className="text-[10px] text-slate-500">Trigger: Paid</div>
                  </div>
                </button>

                <button
                  onClick={() => handleUserEvent('课程结束/召回 (Recall)', 'REPURCHASE')}
                  disabled={isAnimating}
                  className={`w-full p-3 rounded-lg border text-left transition-all flex items-center gap-3 group
                    ${currentStage === 'PAID' ? 'bg-purple-50 border-purple-200 hover:bg-purple-100' : 'bg-slate-50 border-slate-100 opacity-50'}
                  `}
                >
                  <div className="bg-purple-100 text-purple-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
                    <Repeat className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-700">老客召回</div>
                    <div className="text-[10px] text-slate-500">Trigger: Repurchase</div>
                  </div>
                </button>
              </div>
            </div>
          </div>

          {/* Middle Column: Strategy Visualization */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 flex-1 overflow-y-auto">
               <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                <Database className="w-4 h-4" /> 实时策略管道 (Pipeline)
              </h2>
              
              <div className="relative space-y-6 pl-4 before:absolute before:left-6 before:top-4 before:bottom-4 before:w-0.5 before:bg-slate-100">
                {Object.keys(STAGE_CONFIG).map((key) => (
                  <StageCard key={key} stageKey={key} />
                ))}
              </div>
            </div>

            {/* System Terminal */}
            <div className="bg-slate-900 rounded-xl p-4 h-48 flex flex-col shadow-inner">
               <div className="flex justify-between items-center mb-2 border-b border-slate-800 pb-2">
                 <h3 className="text-xs font-mono font-bold text-slate-400 flex items-center gap-2">
                   <Terminal className="w-3 h-3" /> ROUTING ENGINE LOGS
                 </h3>
                 {isAnimating && <span className="text-xs text-green-400 animate-pulse">Processing...</span>}
               </div>
               <div ref={logsContainerRef} className="flex-1 overflow-y-auto font-mono text-[10px] space-y-1.5 scrollbar-thin scrollbar-thumb-slate-700">
                 {systemLogs.length === 0 && <span className="text-slate-600 italic">系统待机中... 等待事件触发</span>}
                 {systemLogs.map((log) => (
                   <div key={log.id} className="flex gap-2">
                     <span className="text-slate-500 opacity-70">[{log.time}]</span>
                     <span className={`${
                       log.type === 'process' ? 'text-blue-400' :
                       log.type === 'success' ? 'text-green-400 font-bold' :
                       log.type === 'warning' ? 'text-yellow-400' :
                       log.type === 'event' ? 'text-purple-400' :
                       'text-slate-300'
                     }`}>
                       {log.type === 'success' && '➜ '}{log.msg}
                     </span>
                   </div>
                 ))}
               </div>
            </div>
          </div>

          {/* Right Column: AI Chat Preview */}
          <div className="lg:col-span-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
              <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center rounded-t-xl">
                 <h2 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-indigo-500" />
                  AI 销售助手对话框
                </h2>
                <span className="text-[10px] bg-slate-200 px-2 py-0.5 rounded text-slate-600">Preview Mode</span>
              </div>
              
              <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/30">
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex flex-col ${msg.role === 'ai' ? 'items-start' : 'items-center my-4 opacity-50'}`}>
                    
                    {msg.role === 'event' ? (
                      <span className="text-[10px] text-slate-400 bg-slate-100 px-3 py-1 rounded-full border border-slate-200">
                        --- {msg.text} ---
                      </span>
                    ) : (
                      <>
                        <div className="flex items-center gap-2 mb-1 ml-1">
                          <span className="text-[10px] font-bold text-slate-400 uppercase">
                            AI Copilot
                          </span>
                          {msg.stage && (
                            <span className={`text-[9px] px-1.5 py-0.5 rounded border ${
                              STAGE_CONFIG[msg.stage] 
                                ? STAGE_CONFIG[msg.stage].color.replace('text-', 'text-opacity-80 text-').replace('bg-', 'bg-opacity-20 bg-') 
                                : 'bg-gray-100 text-gray-500'
                            }`}>
                              基于: {STAGE_CONFIG[msg.stage]?.scriptName || '通用'}
                            </span>
                          )}
                        </div>
                        <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none shadow-sm text-sm text-slate-700 leading-relaxed max-w-[90%]">
                          {msg.text}
                        </div>
                      </>
                    )}
                  </div>
                ))}
                {isAnimating && (
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
              </div>

              {/* Fake Input */}
              <div className="p-4 border-t border-slate-100 bg-white rounded-b-xl">
                <div className="w-full bg-slate-100 h-10 rounded-lg flex items-center px-4 text-slate-400 text-sm cursor-not-allowed">
                  用户已由 AI 自动跟进，无需人工干预...
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}