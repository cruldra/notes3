import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  TrendingUp, 
  Zap, 
  Shield, 
  Target, 
  AlertCircle,
  RefreshCw,
  Gauge,
  ArrowUp,
  Lock
} from 'lucide-react';

// --- Configuration ---
const INTENT_TYPE = 'PRICE_SENSITIVITY'; // The scene we are tracking

const STRATEGIES = {
  LEVEL_1: {
    minConf: 0,
    maxConf: 75,
    name: '🛡️ 价值防御 (Standard Quote)',
    color: 'bg-blue-100 text-blue-700 border-blue-200',
    description: '标准报价，强调课程价值，不卑不亢。',
    reply: '亲，咱们 AI Agent 实战课定价 3999 元，包含 4 周直播授课和助教 1v1 答疑。目前是早鸟优惠期，下单还赠送全套实体教材哦。'
  },
  LEVEL_2: {
    minConf: 75,
    maxConf: 90,
    name: '🤝 价值锚定 (Value Anchor)',
    color: 'bg-orange-100 text-orange-700 border-orange-200',
    description: '感受到价格抗性，拆解服务价值，尝试挽留。',
    reply: '理解您的想法，毕竟 3999 不是小数目。但咱们是实战课，教的是真本事。很多学员学完第一个月接个私单就回本了，这其实是给自己未来的投资呀。'
  },
  LEVEL_3: {
    minConf: 90,
    maxConf: 100,
    name: '🔥 底价逼单 (Hard Close)',
    color: 'bg-red-100 text-red-700 border-red-200',
    description: '确认极高购买意向，抛出限时/隐藏优惠，临门一脚。',
    reply: '看您确实很有学习热情！这样吧，我手头刚好有个学员退出来的“早鸟名额”，可以立减 500 元，仅限今天有效，我帮您锁单？'
  }
};

// Now with specific impact scores to avoid "jumping" levels too early
const USER_INPUTS = [
  { text: "这课要多少钱啊？", label: "1. 普通询价 (Level 1)", delta: 5 }, 
  { text: "太贵了吧... 3999 超出预算了。", label: "2. 表达嫌贵 (Level 2)", delta: 15 },
  { text: "真的不能再便宜点了吗？别家才 2000。", label: "3. 极限施压 (Level 3)", delta: 20 },
  { text: "再考虑一下吧...", label: "4. 犹豫/重置 (Reset)", delta: -100 }
];

export default function ConfidenceDecayDemo() {
  // --- State ---
  const [messages, setMessages] = useState([]);
  const [confidence, setConfidence] = useState(60); // Initial confidence lower to ensure Level 1 start
  const [hitCount, setHitCount] = useState(0);
  const [currentStrategy, setCurrentStrategy] = useState(STRATEGIES.LEVEL_1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [decayActive, setDecayActive] = useState(false); // Visual effect trigger

  const chatRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- Logic ---
  const determineStrategy = (conf) => {
    if (conf >= 90) return STRATEGIES.LEVEL_3;
    if (conf >= 75) return STRATEGIES.LEVEL_2;
    return STRATEGIES.LEVEL_1;
  };

  const handleUserAction = async (input) => {
    if (isProcessing) return;
    setIsProcessing(true);
    setDecayActive(false);

    // 1. User Message
    setMessages(prev => [...prev, { role: 'user', text: input.text }]);

    // 2. Simulate AI Analysis Delay
    await new Promise(r => setTimeout(r, 600));

    // 3. Logic Engine
    let newConf = confidence;
    let newHitCount = hitCount;
    let visualDelta = 0;

    if (input.delta < 0) {
      // Reset Logic
      newConf = 60;
      newHitCount = 0;
      visualDelta = -100;
    } else {
      // Hit Logic
      // Only count as a "hit" (repetition) if it's a significant objection (delta > 5)
      // Standard query (delta=5) just maintains conversation flow
      if (input.delta > 10) {
        newHitCount += 1;
        setDecayActive(true);
      }
      
      visualDelta = input.delta;
      newConf = Math.min(newConf + input.delta, 98); // Cap at 98%
    }

    setConfidence(newConf);
    setHitCount(newHitCount);
    
    const strategy = determineStrategy(newConf);
    setCurrentStrategy(strategy);

    // 4. AI Reply
    await new Promise(r => setTimeout(r, 600));
    setMessages(prev => [...prev, { 
      role: 'ai', 
      text: strategy.reply, 
      strategyName: strategy.name,
      delta: visualDelta
    }]);

    setIsProcessing(false);
  };

  const reset = () => {
    setMessages([]);
    setConfidence(60);
    setHitCount(0);
    setCurrentStrategy(STRATEGIES.LEVEL_1);
    setDecayActive(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="text-indigo-600" />
              场景置信度衰减 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2 Optimization</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              通过用户重复行为动态校准意图 • 让 AI 越聊越懂用户痛点
            </p>
          </div>
          <button 
            onClick={reset}
            className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1"
          >
            <RefreshCw className="w-4 h-4" /> 重置状态
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[750px] lg:h-[650px]">
          
          {/* Left: Chat Interaction */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <span className="font-bold text-slate-700 flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" /> 价格攻防演示 (Price Battle)
                </span>
              </div>
              
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/30">
                {messages.length === 0 && (
                  <div className="text-center text-slate-400 mt-20 text-sm italic">
                    <p>场景设定：用户对价格犹豫不决。</p>
                    <p>请点击下方按钮，模拟用户从询价到嫌贵的心理变化。</p>
                  </div>
                )}
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'ai' ? 'justify-start' : 'justify-end'}`}>
                    <div className="flex flex-col max-w-[85%]">
                       {msg.role === 'ai' && (
                         <div className="flex items-center gap-2 mb-1 ml-1">
                           <span className="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded border border-slate-200 font-bold">
                             {msg.strategyName}
                           </span>
                           {msg.delta > 0 && (
                             <span className="text-[10px] text-green-500 font-bold flex items-center animate-pulse">
                               <ArrowUp className="w-3 h-3" /> Conf +{msg.delta}%
                             </span>
                           )}
                           {msg.delta < 0 && (
                             <span className="text-[10px] text-slate-400 font-bold">
                               Reset
                             </span>
                           )}
                         </div>
                       )}
                       <div className={`p-3 rounded-2xl text-sm shadow-sm leading-relaxed ${msg.role === 'ai' ? 'bg-white border border-slate-200 rounded-tl-none' : 'bg-indigo-600 text-white rounded-tr-none'}`}>
                         {msg.text}
                       </div>
                    </div>
                  </div>
                ))}
                <div ref={chatRef} />
              </div>

              {/* Controls */}
              <div className="p-3 bg-white border-t border-slate-100 grid grid-cols-1 gap-2">
                <p className="text-[10px] font-bold text-slate-400 uppercase ml-1">模拟用户输入 (User Input)</p>
                {USER_INPUTS.map((input, i) => (
                  <button 
                    key={i}
                    onClick={() => handleUserAction(input)}
                    disabled={isProcessing}
                    className="text-left px-3 py-2 rounded-lg border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50 transition-all text-xs font-medium text-slate-700 disabled:opacity-50 flex justify-between"
                  >
                    <span>{input.text}</span>
                    <span className={`text-[10px] opacity-50 font-mono ${input.delta > 10 ? 'text-red-500' : 'text-slate-500'}`}>
                      {input.delta > 0 ? `+${input.delta}` : 'Reset'}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Confidence Monitor */}
          <div className="lg:col-span-7 flex flex-col gap-6 h-full">
            
            {/* 1. Confidence Gauge */}
            <div className="bg-slate-900 text-white rounded-xl p-6 shadow-xl border border-slate-800 relative overflow-hidden">
              <div className="flex justify-between items-center mb-8 relative z-10">
                <h2 className="font-bold flex items-center gap-2">
                  <Gauge className="w-5 h-5 text-indigo-400" />
                  Intent Confidence Engine
                </h2>
                <div className="flex items-center gap-2 text-xs bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
                  <Target className="w-3 h-3 text-red-400" />
                  <span>Target Intent: <span className="font-mono font-bold text-red-400">{INTENT_TYPE}</span></span>
                </div>
              </div>

              {/* Progress Bar Visual */}
              <div className="relative z-10">
                <div className="flex justify-between text-xs font-bold text-slate-400 mb-2 uppercase">
                  <span>Level 1 (Inquiry)</span>
                  <span>Level 2 (Resist)</span>
                  <span>Level 3 (Close)</span>
                </div>
                <div className="h-8 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700 relative">
                  {/* Threshold Markers */}
                  <div className="absolute top-0 bottom-0 left-[75%] w-0.5 bg-yellow-500/50 z-20" title="Level 2 Threshold (75%)"></div>
                  <div className="absolute top-0 bottom-0 left-[90%] w-0.5 bg-red-500/50 z-20" title="Level 3 Threshold (90%)"></div>
                  
                  {/* Fill */}
                  <div 
                    className={`h-full transition-all duration-700 ease-out flex items-center justify-end pr-3
                      ${confidence >= 90 ? 'bg-gradient-to-r from-indigo-500 to-red-500' : 
                        confidence >= 75 ? 'bg-gradient-to-r from-indigo-500 to-orange-500' : 
                        'bg-gradient-to-r from-blue-500 to-indigo-500'}
                    `}
                    style={{ width: `${confidence}%` }}
                  >
                    <span className="text-xs font-black text-white drop-shadow-md">{confidence}%</span>
                  </div>
                </div>
                
                {/* Dynamic Feedback Text */}
                <div className="mt-4 flex justify-between items-center h-8">
                  <div className="text-xs text-slate-400">
                    Confidence Score: <span className="text-white font-mono text-lg">{confidence}</span>
                  </div>
                  {decayActive && (
                     <div className="text-sm font-bold text-green-400 animate-in slide-in-from-bottom-2 fade-in">
                       ⚡️ Repetition Detected {'>'} Boosting Confidence
                     </div>
                  )}
                </div>
              </div>

              {/* Background Decoration */}
              {confidence >= 90 && (
                 <div className="absolute -right-10 -bottom-20 w-64 h-64 bg-red-600/20 blur-3xl rounded-full pointer-events-none animate-pulse"></div>
              )}
            </div>

            {/* 2. Strategy Logic Map */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col relative">
               <h3 className="font-bold text-slate-800 mb-6 flex items-center gap-2">
                 <Zap className="w-5 h-5 text-indigo-600" />
                 策略路由映射 (Strategy Mapping)
               </h3>
               
               <div className="space-y-4 relative z-10">
                 {Object.entries(STRATEGIES).map(([key, strat]) => {
                   const isActive = currentStrategy.name === strat.name;
                   return (
                     <div key={key} className={`
                       p-4 rounded-xl border-2 transition-all duration-500 flex items-center gap-4 relative overflow-hidden
                       ${isActive ? `${strat.color} scale-105 shadow-md` : 'bg-slate-50 border-slate-100 opacity-50 grayscale'}
                     `}>
                       {isActive && <div className="absolute left-0 top-0 w-1 h-full bg-current"></div>}
                       
                       <div className="w-12 h-12 rounded-full bg-white/50 flex items-center justify-center text-2xl shadow-sm shrink-0">
                         {key === 'LEVEL_1' ? '🛡️' : key === 'LEVEL_2' ? '🤝' : '🔥'}
                       </div>
                       
                       <div className="flex-1">
                         <div className="flex justify-between items-center mb-1">
                           <h4 className="font-bold">{strat.name}</h4>
                           <span className="text-[10px] font-mono font-bold opacity-70 border px-1.5 rounded bg-white/50">
                             {strat.minConf}% - {strat.maxConf}%
                           </span>
                         </div>
                         <p className="text-xs opacity-90 leading-relaxed">
                           {strat.description}
                         </p>
                       </div>

                       {isActive && (
                         <div className="absolute right-2 top-2 animate-pulse">
                           <Shield className="w-4 h-4 opacity-50" />
                         </div>
                       )}
                     </div>
                   );
                 })}
               </div>

               {/* Lock Indicator for Level 3 */}
               {confidence < 90 && (
                 <div className="absolute bottom-8 right-8 text-slate-300 opacity-20 pointer-events-none">
                   <Lock className="w-32 h-32" />
                 </div>
               )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}