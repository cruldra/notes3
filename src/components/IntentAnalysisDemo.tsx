import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Brain, 
  Zap, 
  Terminal, 
  Activity, 
  AlertTriangle,
  Smile,
  Frown,
  Meh,
  Search,
  CheckCircle,
  XCircle,
  Tag
} from 'lucide-react';

// --- Scenarios Configuration ---
const SCENARIOS = [
  {
    id: 'price_complaint',
    text: "太贵了吧，隔壁家才卖 9.9，你们凭什么卖 399？",
    label: "场景1: 嫌贵/比价",
    analysis: {
      intent: "PRICE_OBJECTION",
      emotion: "NEGATIVE_AGGRESSIVE", // 负向-攻击性
      urgency: "HIGH",
      confidence: 0.95,
      reasoning: "用户进行了直接价格对比，并使用了'凭什么'等反问句，情绪激动。"
    },
    badReply: "亲，我们的课程物超所值哦，399元包含...",
    goodReply: "我非常理解您的感受，乍一听确实比9.9的贵不少。但咱们教的是能变现的真本事，隔壁可能只是个试听课。如果不满意，我们支持无理由退款，您看要不先试听一节？"
  },
  {
    id: 'skeptical_replay',
    text: "呵呵，难道还有回放？",
    label: "场景2: 怀疑/反问",
    analysis: {
      intent: "QUERY_REPLAY",
      emotion: "NEGATIVE_SKEPTICAL", // 负向-怀疑/嘲讽
      urgency: "MEDIUM",
      confidence: 0.92,
      reasoning: "检测到'呵呵'与'难道'，虽包含'回放'关键词，但实为反问句，表达对服务承诺的不信任。"
    },
    badReply: "有的，回放链接是：www.example.com/video",
    goodReply: "听得出来您之前可能遇到过承诺有回放但没兑现的情况。请放心，咱们这期是全程高清录播的！链接马上发您，永久有效，您可以随时检查。"
  },
  {
    id: 'tech_issue',
    text: "怎么一直支付失败啊？？急死我了，不想买了！",
    label: "场景3: 支付焦急",
    analysis: {
      intent: "TECH_ISSUE_PAYMENT",
      emotion: "ANXIOUS_HIGH", // 焦虑-极高
      urgency: "CRITICAL",
      confidence: 0.98,
      reasoning: "用户遇到支付阻断，且表达了'急死'和'不想买了'的放弃意向，需最高优先级介入。"
    },
    badReply: "请您尝试切换网络或重启手机。",
    goodReply: "别急别急！可能是系统卡顿了。我已经通知技术部在查了。我先把个人收款码发您，给您手动开通权限，绝不耽误您上课！"
  },
  {
    id: 'ambiguous',
    text: "再说吧",
    label: "场景4: 模糊/敷衍",
    analysis: {
      intent: "DELAY_DECISION",
      emotion: "NEUTRAL_PASSIVE", // 中性-消极
      urgency: "LOW",
      confidence: 0.65, // 低置信度
      reasoning: "文本过短，缺乏明确意图。可能是推脱，也可能是真的需要时间。"
    },
    badReply: "好的，那您考虑好了联系我。",
    goodReply: "没问题，买课毕竟是个大事。您主要是顾虑哪方面呢？是时间安排不开，还是觉得课程难度不合适？"
  }
];

export default function IntentAnalysisDemo() {
  // --- State ---
  const [messages, setMessages] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showJson, setShowJson] = useState(true);
  
  // --- Actions ---
  const handleSendMessage = async (scenario) => {
    // 1. User Message
    const userMsg = { role: 'user', text: scenario.text, timestamp: new Date().toLocaleTimeString() };
    setMessages(prev => [...prev, userMsg]);
    setCurrentAnalysis(null);
    setIsAnalyzing(true);

    // 2. Simulate AI Processing Delay (The "Brain" working)
    await new Promise(r => setTimeout(r, 1200));

    // 3. Analysis Result
    setIsAnalyzing(false);
    setCurrentAnalysis(scenario);

    // 4. Simulate Response (After analysis)
    await new Promise(r => setTimeout(r, 800));
    const aiMsg = { 
      role: 'ai', 
      text: scenario.goodReply, 
      badText: scenario.badReply, // Store for comparison
      tags: scenario.analysis,
      timestamp: new Date().toLocaleTimeString() 
    };
    setMessages(prev => [...prev, aiMsg]);
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentAnalysis(null);
  };

  // --- Helper to get color by emotion ---
  const getEmotionColor = (emotion) => {
    if (emotion.includes('NEGATIVE') || emotion.includes('ANXIOUS')) return 'text-red-600 bg-red-50 border-red-200';
    if (emotion.includes('POSITIVE')) return 'text-green-600 bg-green-50 border-green-200';
    return 'text-slate-600 bg-slate-50 border-slate-200';
  };

  const getUrgencyColor = (urgency) => {
    if (urgency === 'CRITICAL') return 'bg-red-600 text-white animate-pulse';
    if (urgency === 'HIGH') return 'bg-orange-500 text-white';
    if (urgency === 'MEDIUM') return 'bg-yellow-500 text-white';
    return 'bg-slate-400 text-white';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Brain className="text-indigo-600" />
              意图/情绪实时标签引擎 <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full uppercase tracking-wide">P0 Core</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              LLM 实时语义分析 • 让 AI 听懂“话外之音” • 避免关键词匹配的“智障回复”
            </p>
          </div>
          <button onClick={clearChat} className="text-sm text-slate-400 hover:text-slate-600">清空演示</button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[800px] lg:h-[650px]">
          
          {/* Left: Chat Simulator */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                <span className="font-bold text-slate-700 flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" /> 用户对话窗口
                </span>
                <span className="text-xs text-slate-400">Simulation Mode</span>
              </div>
              
              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/30">
                {messages.length === 0 && (
                  <div className="text-center text-slate-400 mt-10 text-sm">
                    请点击下方场景按钮开始测试...
                  </div>
                )}
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'ai' ? 'justify-start' : 'justify-end'}`}>
                    <div className={`max-w-[85%] rounded-2xl p-3 text-sm shadow-sm ${msg.role === 'ai' ? 'bg-white border border-slate-200 rounded-tl-none' : 'bg-indigo-600 text-white rounded-tr-none'}`}>
                      {msg.role === 'ai' && (
                        <div className="mb-2 pb-2 border-b border-slate-100 flex gap-2 flex-wrap">
                          <span className="text-[10px] font-bold bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">
                             {msg.tags.intent}
                          </span>
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-1 ${getEmotionColor(msg.tags.emotion)}`}>
                             {msg.tags.emotion.includes('NEGATIVE') ? <Frown className="w-3 h-3" /> : <Smile className="w-3 h-3" />}
                             {msg.tags.emotion}
                          </span>
                        </div>
                      )}
                      <div className="leading-relaxed">{msg.text}</div>
                      {msg.role === 'ai' && (
                        <div className="mt-2 pt-2 border-t border-slate-100 text-[10px] text-slate-400">
                          <span className="font-bold text-red-400 block mb-1">❌ 传统关键词回复 (Comparison):</span>
                          "{msg.badText}"
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isAnalyzing && (
                   <div className="flex justify-start">
                     <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                       <Brain className="w-4 h-4 text-indigo-500 animate-pulse" />
                       <span className="text-xs text-slate-500">正在分析情绪与意图...</span>
                     </div>
                   </div>
                )}
              </div>

              {/* Input Area (Scenario Buttons) */}
              <div className="p-4 bg-white border-t border-slate-100 space-y-2">
                <p className="text-xs font-bold text-slate-400 uppercase mb-2">选择测试场景 (Select Scenario)</p>
                <div className="grid grid-cols-1 gap-2">
                  {SCENARIOS.map(scenario => (
                    <button 
                      key={scenario.id}
                      onClick={() => handleSendMessage(scenario)}
                      disabled={isAnalyzing}
                      className="text-left px-3 py-2 rounded-lg border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50 transition-all text-xs font-medium text-slate-700 disabled:opacity-50 flex justify-between items-center group"
                    >
                      <span className="truncate flex-1">{scenario.label}: "{scenario.text.substring(0, 15)}..."</span>
                      <Zap className="w-3 h-3 text-slate-300 group-hover:text-indigo-500" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right: The AI Brain (Analysis) */}
          <div className="lg:col-span-7 flex flex-col gap-4 h-full">
            <div className="bg-slate-900 text-slate-300 rounded-xl p-6 shadow-xl border border-slate-800 h-full flex flex-col relative overflow-hidden">
              
              {/* Background Decoration */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -z-0 pointer-events-none"></div>

              <div className="flex justify-between items-center mb-6 z-10">
                <h2 className="font-bold text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-green-400" />
                  实时分析面板 (Real-time Analysis)
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Confidence Threshold: 0.75</span>
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                </div>
              </div>

              {!currentAnalysis && !isAnalyzing ? (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-600 z-10">
                  <Brain className="w-16 h-16 mb-4 opacity-20" />
                  <p>等待消息输入...</p>
                </div>
              ) : (
                <div className="space-y-8 z-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  
                  {/* 1. Core Tags Visualization */}
                  <div className="grid grid-cols-3 gap-4">
                    {/* Intent Card */}
                    <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 relative overflow-hidden group">
                      <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                      <div className="text-xs text-slate-500 uppercase font-bold mb-1">Intent (意图)</div>
                      <div className="text-lg font-bold text-white tracking-wide">
                        {isAnalyzing ? <span className="animate-pulse">Scanning...</span> : currentAnalysis?.analysis.intent}
                      </div>
                      <Search className="absolute bottom-2 right-2 w-8 h-8 text-blue-500/10 group-hover:text-blue-500/20 transition-colors" />
                    </div>

                    {/* Emotion Card */}
                    <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 relative overflow-hidden group">
                      <div className={`absolute top-0 left-0 w-1 h-full ${isAnalyzing ? 'bg-slate-500' : currentAnalysis?.analysis.emotion.includes('NEGATIVE') ? 'bg-red-500' : 'bg-green-500'}`}></div>
                      <div className="text-xs text-slate-500 uppercase font-bold mb-1">Emotion (情绪)</div>
                      <div className={`text-lg font-bold tracking-wide ${isAnalyzing ? 'text-white' : currentAnalysis?.analysis.emotion.includes('NEGATIVE') ? 'text-red-400' : 'text-green-400'}`}>
                        {isAnalyzing ? <span className="animate-pulse">Analyzing...</span> : currentAnalysis?.analysis.emotion}
                      </div>
                      {currentAnalysis?.analysis.emotion.includes('NEGATIVE') ? 
                        <AlertTriangle className="absolute bottom-2 right-2 w-8 h-8 text-red-500/10 group-hover:text-red-500/20 transition-colors" /> : 
                        <Smile className="absolute bottom-2 right-2 w-8 h-8 text-green-500/10 group-hover:text-green-500/20 transition-colors" />
                      }
                    </div>

                    {/* Urgency Card */}
                    <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 relative overflow-hidden group">
                      <div className="text-xs text-slate-500 uppercase font-bold mb-1">Urgency (紧急度)</div>
                      <div className="flex items-center gap-2">
                        {isAnalyzing ? (
                          <span className="text-white animate-pulse">Calculating...</span>
                        ) : (
                          <span className={`px-2 py-1 rounded text-xs font-bold ${getUrgencyColor(currentAnalysis?.analysis.urgency)}`}>
                            {currentAnalysis?.analysis.urgency}
                          </span>
                        )}
                      </div>
                      <Zap className="absolute bottom-2 right-2 w-8 h-8 text-yellow-500/10 group-hover:text-yellow-500/20 transition-colors" />
                    </div>
                  </div>

                  {/* 2. Reasoning & Confidence */}
                  <div className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex justify-between items-end mb-2">
                      <h3 className="text-sm font-bold text-slate-400">AI 推理过程 (Reasoning)</h3>
                      {!isAnalyzing && (
                        <div className="text-xs flex items-center gap-1">
                          <span className="text-slate-500">Confidence Score:</span>
                          <span className={`font-mono font-bold ${currentAnalysis?.analysis.confidence > 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                            {(currentAnalysis?.analysis.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed italic">
                      {isAnalyzing ? "正在调用 LLM 进行上下文语义拆解..." : `"${currentAnalysis?.analysis.reasoning}"`}
                    </p>
                    {/* Confidence Bar */}
                    {!isAnalyzing && (
                      <div className="w-full h-1 bg-slate-700 rounded-full mt-3 overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${currentAnalysis?.analysis.confidence > 0.8 ? 'bg-green-500' : 'bg-yellow-500'}`} 
                          style={{ width: `${currentAnalysis?.analysis.confidence * 100}%` }}
                        ></div>
                      </div>
                    )}
                  </div>

                  {/* 3. JSON Output Viewer */}
                  <div className="flex-1 flex flex-col min-h-[200px]">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-xs font-bold text-slate-500 uppercase flex items-center gap-2">
                        <Terminal className="w-3 h-3" /> System Output (JSON)
                      </h3>
                      <button onClick={() => setShowJson(!showJson)} className="text-[10px] text-blue-400 hover:text-blue-300">
                        {showJson ? 'Hide' : 'Show'}
                      </button>
                    </div>
                    {showJson && (
                      <div className="bg-black/50 rounded-lg p-4 font-mono text-xs text-green-400 overflow-auto border border-slate-700/50 flex-1 shadow-inner">
                        {isAnalyzing ? (
                          <div className="animate-pulse space-y-2">
                            <div className="h-2 bg-green-900/30 rounded w-1/2"></div>
                            <div className="h-2 bg-green-900/30 rounded w-3/4"></div>
                            <div className="h-2 bg-green-900/30 rounded w-2/3"></div>
                          </div>
                        ) : (
                          <pre>{JSON.stringify(currentAnalysis?.analysis, null, 2)}</pre>
                        )}
                      </div>
                    )}
                  </div>

                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}