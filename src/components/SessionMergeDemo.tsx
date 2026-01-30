import React, { useState, useEffect, useRef } from 'react';
import { 
  Users, 
  Link2, 
  Unlink, 
  MessageSquare, 
  ShieldCheck, 
  Brain, 
  GitMerge, 
  User, 
  ArrowRight,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

// --- Constants ---
const USER_A = {
  id: 'husband',
  name: '李先生 (Husband)',
  avatar: '👨🏻‍💻',
  role: 'Technical Decider',
  focus: '关注课程大纲、Python支持、实战项目',
  color: 'bg-blue-50 border-blue-200'
};

const USER_B = {
  id: 'wife',
  name: '王女士 (Wife)',
  avatar: '👩🏻‍💼',
  role: 'Budget Holder',
  focus: '关注价格、性价比、学习时长',
  color: 'bg-pink-50 border-pink-200'
};

const SCENARIOS = [
  {
    step: 1,
    action: '独立咨询 (Isolated)',
    description: '两人分别咨询，AI 无法互通信息。',
    chatA: { text: "你们这个课教 AI Agent 开发吗？支持 Python 吗？", reply: "支持的。我们的课程涵盖了从 Python 基础到 AI Agent 开发的全流程，非常适合技术进阶。" },
    chatB: { text: "3999 有点贵啊，能不能便宜点？", reply: "亲，3999 是我们的标准定价，包含了所有服务，暂时没有折扣哦。" }
  },
  {
    step: 2,
    action: '合并后咨询 (Merged)',
    description: '销售识别关系并合并，AI 综合双方信息进行回复。',
    chatA: { text: "那我这周末能开始学吗？", reply: "没问题！而且考虑到王女士那边还在犹豫预算，我们针对家庭学习有个双人立减优惠，您可以跟她商量下。" },
    chatB: { text: "还是觉得有点不划算...", reply: "王女士，虽然价格看着不低，但李先生非常看重其中的 'AI Agent' 实战项目，这对他的职场晋升很有帮助。作为家庭投资来说，回报率是很高的！而且我们刚申请到了一个家庭特惠..." }
  }
];

export default function SessionMergeDemo() {
  // --- State ---
  const [isMerged, setIsMerged] = useState(false);
  const [chatHistoryA, setChatHistoryA] = useState([]);
  const [chatHistoryB, setChatHistoryB] = useState([]);
  const [activeStep, setActiveStep] = useState(0); // 0 = Initial, 1 = Step 1 Done, 2 = Merged...
  const [systemLogs, setSystemLogs] = useState([]);

  // --- Actions ---
  const addLog = (msg) => {
    setSystemLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev]);
  };

  const handleStep1 = async () => {
    if (activeStep > 0) return;
    
    // User A speaks
    addLog(`收到消息 (李先生): 咨询技术细节`);
    setChatHistoryA(prev => [...prev, { role: 'user', text: SCENARIOS[0].chatA.text }]);
    await new Promise(r => setTimeout(r, 800));
    setChatHistoryA(prev => [...prev, { role: 'ai', text: SCENARIOS[0].chatA.reply }]);
    
    // User B speaks
    await new Promise(r => setTimeout(r, 500));
    addLog(`收到消息 (王女士): 咨询价格预算`);
    setChatHistoryB(prev => [...prev, { role: 'user', text: SCENARIOS[0].chatB.text }]);
    await new Promise(r => setTimeout(r, 800));
    setChatHistoryB(prev => [...prev, { role: 'ai', text: SCENARIOS[0].chatB.reply }]);

    setActiveStep(1);
    addLog(`⚠️ 警告: 检测到潜在关联 (IP/支付账号相似)，建议合并`);
  };

  const toggleMerge = () => {
    setIsMerged(!isMerged);
    if (!isMerged) {
      addLog(`🔗 执行合并: 创建决策单元 [Family_Li_Wang]`);
      addLog(`🧠 AI 上下文: 上下文已同步，共享 Intent 标签`);
    } else {
      addLog(`💔 解除合并: 拆分为独立线索`);
    }
  };

  const handleStep2 = async () => {
    if (activeStep < 1) return;

    // Logic branching based on merge status
    if (!isMerged) {
      // Still isolated (Dumb AI)
      setChatHistoryB(prev => [...prev, { role: 'user', text: "还是觉得有点不划算..." }]);
      await new Promise(r => setTimeout(r, 600));
      setChatHistoryB(prev => [...prev, { role: 'ai', text: "亲，我们的课程质量很高的，您再考虑一下？" }]); // Generic reply
      addLog(`回复 (王女士): 通用挽留话术 (未利用丈夫信息)`);
    } else {
      // Merged (Smart AI)
      setChatHistoryB(prev => [...prev, { role: 'user', text: SCENARIOS[1].chatB.text }]);
      await new Promise(r => setTimeout(r, 800));
      
      const smartReply = SCENARIOS[1].chatB.reply;
      setChatHistoryB(prev => [...prev, { role: 'ai', text: smartReply, isSmart: true }]);
      addLog(`🔥 智能回复 (王女士): 引用 [李先生] 的技术需求作为价值锚点`);
      
      await new Promise(r => setTimeout(r, 1000));
      setChatHistoryA(prev => [...prev, { role: 'user', text: SCENARIOS[1].chatA.text }]);
      await new Promise(r => setTimeout(r, 800));
      setChatHistoryA(prev => [...prev, { role: 'ai', text: SCENARIOS[1].chatA.reply, isSmart: true }]);
    }
    setActiveStep(2);
  };

  const reset = () => {
    setChatHistoryA([]);
    setChatHistoryB([]);
    setIsMerged(false);
    setActiveStep(0);
    setSystemLogs([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <GitMerge className="text-indigo-600" />
              多人协同会话合并 <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              将孤立的线索合并为“决策单元” • 让 AI 掌握全局信息
            </p>
          </div>
          <button 
            onClick={reset}
            className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1"
          >
            <Unlink className="w-4 h-4" /> 重置演示
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[800px] lg:h-[700px]">
          
          {/* Column 1: Husband's Phone */}
          <div className="lg:col-span-3 flex flex-col gap-4 h-full">
            <PhoneFrame user={USER_A} messages={chatHistoryA} isMerged={isMerged} />
          </div>

          {/* Column 2: The Bridge (Sales Dashboard) */}
          <div className="lg:col-span-6 flex flex-col gap-4 h-full">
            
            {/* Control Panel */}
            <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 z-10 relative overflow-hidden">
               {isMerged && (
                 <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-400 animate-pulse"></div>
               )}
               
               <div className="flex justify-between items-center mb-6">
                 <h2 className="font-bold text-slate-800 flex items-center gap-2">
                   <ShieldCheck className="w-5 h-5 text-indigo-600" />
                   销售决策工作台 (CRM)
                 </h2>
                 <div className={`px-3 py-1 rounded-full text-xs font-bold border ${isMerged ? 'bg-purple-50 text-purple-700 border-purple-200' : 'bg-slate-100 text-slate-500 border-slate-200'}`}>
                   状态: {isMerged ? 'LINKED (已合并)' : 'ISOLATED (独立)'}
                 </div>
               </div>

               {/* Connection Visualizer */}
               <div className="flex items-center justify-between px-8 py-6 bg-slate-50 rounded-xl border border-slate-100 mb-6 relative">
                  {/* Connection Line */}
                  <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60%] h-0.5 transition-all duration-700 ${isMerged ? 'bg-indigo-400' : 'bg-slate-300 border-t border-dashed border-slate-300'}`}></div>
                  
                  {/* User Nodes */}
                  <div className={`relative z-10 w-12 h-12 rounded-full border-2 flex items-center justify-center text-xl bg-white transition-all ${isMerged ? 'border-indigo-500 scale-110 shadow-indigo-200 shadow-lg' : 'border-slate-300 grayscale'}`}>
                    {USER_A.avatar}
                    {isMerged && <div className="absolute -bottom-6 w-20 text-[10px] text-center font-bold text-indigo-600">Technical</div>}
                  </div>

                  <div className="relative z-10 bg-white p-2 rounded-full">
                    <button 
                      onClick={toggleMerge}
                      className={`
                        w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md
                        ${isMerged 
                          ? 'bg-indigo-600 text-white hover:bg-indigo-700 rotate-0' 
                          : 'bg-slate-200 text-slate-400 hover:bg-slate-300 -rotate-45'}
                      `}
                    >
                      <Link2 className="w-6 h-6" />
                    </button>
                  </div>

                  <div className={`relative z-10 w-12 h-12 rounded-full border-2 flex items-center justify-center text-xl bg-white transition-all ${isMerged ? 'border-pink-500 scale-110 shadow-pink-200 shadow-lg' : 'border-slate-300 grayscale'}`}>
                    {USER_B.avatar}
                    {isMerged && <div className="absolute -bottom-6 w-20 text-[10px] text-center font-bold text-pink-600">Budget</div>}
                  </div>
               </div>

               {/* Step Controls */}
               <div className="grid grid-cols-2 gap-4">
                 <button
                   onClick={handleStep1}
                   disabled={activeStep > 0}
                   className={`p-3 rounded-lg border text-sm font-bold text-left transition-all ${activeStep > 0 ? 'bg-slate-50 border-slate-200 text-slate-400' : 'bg-white border-indigo-200 hover:border-indigo-400 text-indigo-700 shadow-sm'}`}
                 >
                   <span className="block text-xs uppercase opacity-70 mb-1">Step 1</span>
                   1. 模拟“各聊各的” (独立咨询)
                 </button>
                 <button
                   onClick={handleStep2}
                   disabled={activeStep !== 1}
                   className={`p-3 rounded-lg border text-sm font-bold text-left transition-all ${activeStep !== 1 ? 'bg-slate-50 border-slate-200 text-slate-400' : 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-md hover:shadow-lg'}`}
                 >
                   <span className="block text-xs uppercase opacity-70 mb-1">Step 2</span>
                   2. 模拟“再次提问” (触发 AI)
                 </button>
               </div>
               
               {!isMerged && activeStep === 1 && (
                 <div className="mt-3 text-center animate-bounce text-xs font-bold text-indigo-600">
                   👆 建议先点击中间的“连接图标”进行合并，再执行 Step 2
                 </div>
               )}
            </div>

            {/* System Logs (Context View) */}
            <div className="bg-slate-900 text-slate-300 rounded-xl p-4 flex-1 font-mono text-xs overflow-hidden flex flex-col shadow-inner">
               <div className="border-b border-slate-700 pb-2 mb-2 flex justify-between items-center">
                 <span className="font-bold text-slate-400 flex items-center gap-2">
                   <Brain className="w-3 h-3" /> AI CONTEXT MEMORY
                 </span>
                 {isMerged && <span className="text-green-400 text-[10px] border border-green-500/30 px-1.5 py-0.5 rounded">SHARED MEMORY ACTIVE</span>}
               </div>
               <div className="flex-1 overflow-y-auto space-y-2">
                 {systemLogs.length === 0 && <span className="opacity-30 italic">System ready...</span>}
                 {systemLogs.map((log, i) => (
                   <div key={i} className={`
                     ${log.includes('合并') ? 'text-green-400 font-bold' : ''}
                     ${log.includes('智能回复') ? 'text-purple-300' : ''}
                     ${log.includes('警告') ? 'text-yellow-400' : ''}
                   `}>
                     {log}
                   </div>
                 ))}
               </div>
            </div>

          </div>

          {/* Column 3: Wife's Phone */}
          <div className="lg:col-span-3 flex flex-col gap-4 h-full">
            <PhoneFrame user={USER_B} messages={chatHistoryB} isMerged={isMerged} />
          </div>

        </div>
      </div>
    </div>
  );
}

// --- Sub-Component: Phone Frame ---
function PhoneFrame({ user, messages, isMerged }) {
  const chatRef = useRef(null);
  
  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="bg-white h-full rounded-[2rem] border-8 border-slate-200 shadow-xl overflow-hidden flex flex-col relative">
      {/* Dynamic Header */}
      <div className={`p-4 ${user.color} border-b transition-colors duration-500`}>
        <div className="flex items-center gap-3">
          <div className="text-2xl">{user.avatar}</div>
          <div>
            <div className="font-bold text-sm text-slate-800">{user.name}</div>
            <div className="text-[10px] text-slate-500 leading-tight">{user.role}</div>
          </div>
        </div>
        {isMerged && (
          <div className="mt-2 bg-white/60 p-1.5 rounded text-[10px] text-indigo-700 font-bold flex items-center justify-center gap-1 animate-in fade-in slide-in-from-top-2">
            <Link2 className="w-3 h-3" /> Linked to Decision Unit
          </div>
        )}
      </div>

      {/* Chat Area */}
      <div className="flex-1 bg-slate-50 p-3 overflow-y-auto space-y-3">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`
              max-w-[85%] p-2.5 rounded-xl text-xs leading-relaxed shadow-sm
              ${msg.role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'}
              ${msg.isSmart ? 'ring-2 ring-purple-400 ring-offset-1' : ''}
            `}>
              {msg.text}
              {msg.isSmart && (
                <div className="mt-2 pt-2 border-t border-slate-100 text-[9px] text-purple-600 font-bold flex items-center gap-1">
                  <SparklesIcon /> AI Combined Insight
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={chatRef} />
      </div>

      {/* Input Area (Fake) */}
      <div className="p-3 bg-white border-t border-slate-100">
        <div className="h-8 bg-slate-100 rounded-full w-full flex items-center px-3 text-xs text-slate-400">
          Type a message...
        </div>
      </div>
    </div>
  );
}

function SparklesIcon() {
  return (
    <svg className="w-3 h-3 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 3.214L13 21l-2.286-6.857L5 12l5.714-3.214L13 3z" />
    </svg>
  );
}