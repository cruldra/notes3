import React, { useState, useEffect, useRef } from 'react';
import { 
  User, 
  Target, 
  Shield, 
  MessageSquare, 
  Database, 
  Zap, 
  Lock, 
  CheckCircle, 
  FileText,
  AlertOctagon,
  Sparkles,
  Search
} from 'lucide-react';

// --- Configuration ---
const FIELD_CONFIG = [
  { 
    id: 'city', 
    label: '所在城市', 
    sensitivity: 'LOW', 
    priority: 1, 
    keywords: ['深圳', '北京', '上海', '广州', '杭州', '成都'], 
    question: ['方便问一下您现在在哪个城市发展吗？', '看来您对这边很熟，您是在北京还是？'],
    value: null 
  },
  { 
    id: 'industry', 
    label: '行业/职业', 
    sensitivity: 'MEDIUM', 
    priority: 2, 
    keywords: ['电商', '运营', '产品', '开发', '销售', '老师', '设计'],
    question: ['为了给您推荐对标案例，冒昧问下您从事哪个行业呀？', '您是做产品还是运营相关工作的呢？'],
    value: null 
  },
  { 
    id: 'pain_point', 
    label: '核心痛点', 
    sensitivity: 'MEDIUM', 
    priority: 3, 
    keywords: ['副业', '太卷', '提效', '变现', '焦虑', '很多', '赚钱'],
    question: ['您这次想学 AI，主要是为了副业变现，还是工作提效呢？', '目前工作中遇到最大的 AI 落地难题是什么？'],
    value: null 
  },
  { 
    id: 'budget', 
    label: '预算范围', 
    sensitivity: 'HIGH', 
    priority: 4, 
    keywords: ['3000', '5000', '没钱', '预算', '多少钱'],
    question: ['我们有基础班和实战营，您预期的投入大概是多少呢？', '如果不方便透露，您可以看下这个价格区间哪个更合适？'],
    incentive: '回答后发送《2025 AI 变现白皮书.pdf》',
    value: null 
  }
];

const USER_RESPONSES = {
  city: "我在深圳。",
  industry: "我是做跨境电商运营的。",
  pain_point: "主要是想搞副业，主业太卷了。",
  budget: "预算 3000 左右吧。",
  refuse: "这个不方便说。",
  ignore: "你先回答我的问题。"
};

export default function PortraitCompletionDemo() {
  // --- State ---
  const [profile, setProfile] = useState(FIELD_CONFIG);
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: '您好！我是您的专属 AI 助教。关于课程有任何问题都可以问我哦！' }
  ]);
  const [activeQuestion, setActiveQuestion] = useState(null);
  const [circuitBreaker, setCircuitBreaker] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractionLog, setExtractionLog] = useState(null);
  
  const chatEndRef = useRef(null);

  const filledCount = profile.filter(f => f.value !== null && f.value !== 'REFUSED').length;
  const qualityScore = Math.round((filledCount / profile.length) * 100);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, extractionLog]);

  // --- Logic ---

  // Helper: Simulate LLM Extraction
  const extractInfoFromText = (text, currentProfile) => {
    const updates = {};
    const logs = [];
    
    currentProfile.forEach(field => {
      // Only extract if field is currently empty
      if (field.value === null && field.keywords) {
        const match = field.keywords.find(k => text.includes(k));
        if (match) {
          updates[field.id] = match; // Extract keyword only
          logs.push(`🔍 语义捕获: 识别到 "${match}" -> 填充 [${field.label}]`);
        }
      }
    });
    
    return { updates, logs };
  };

  // Check and trigger next question based on a SPECIFIC profile state (snapshot)
  // This avoids stale closure issues where it looks at old state
  const checkAndTriggerQuestion = async (snapshotProfile) => {
    if (circuitBreaker) return null;
    
    // Find next missing field from the snapshot
    const nextField = snapshotProfile.find(f => f.value === null);
    
    if (nextField) {
      await new Promise(r => setTimeout(r, 600));
      
      const questionText = nextField.question[0];
      const fullText = nextField.incentive 
        ? `${questionText} (🎁 小福利：回答后我把《行业白皮书》发您~)` 
        : questionText;

      setChatHistory(prev => [...prev, { 
        role: 'ai', 
        text: fullText, 
        isQuestion: true,
        fieldLabel: nextField.label
      }]);
      setActiveQuestion(nextField.id);
    } else {
      setActiveQuestion(null);
      // Check if completely filled
      const filled = snapshotProfile.filter(f => f.value !== null && f.value !== 'REFUSED').length;
      if (filled === snapshotProfile.length) {
         setChatHistory(prev => [...prev, { role: 'ai', text: '太棒了！您的需求我都了解了，这就为您生成专属学习计划...' }]);
      }
    }
  };

  const handleUserReply = async (type) => {
    if (isProcessing) return;
    setIsProcessing(true);
    setExtractionLog(null);

    // 1. Determine Input Text
    let userText = '';
    if (type === 'natural_chat') {
      userText = "我在深圳做电商运营，最近主业太卷了想搞点副业。";
    } else {
      userText = USER_RESPONSES[type] || type;
    }

    setChatHistory(prev => [...prev, { role: 'user', text: userText }]);

    // 2. Create a local copy of profile to mutate (avoiding stale state)
    let nextProfile = [...profile];
    
    // 3. Passive Extraction (Run against local copy)
    const { updates, logs } = extractInfoFromText(userText, nextProfile);
    let extractedCount = 0;

    if (logs.length > 0) {
      await new Promise(r => setTimeout(r, 400));
      setExtractionLog(logs);
      
      // Apply extractions to local copy
      nextProfile = nextProfile.map(f => {
        if (updates[f.id]) {
          extractedCount++;
          return { ...f, value: updates[f.id] }; 
        }
        return f;
      });
    }

    await new Promise(r => setTimeout(r, 800));

    // 4. Handle Active Question (if user was replying to one)
    if (activeQuestion) {
      if (type === 'refuse') {
        nextProfile = nextProfile.map(f => f.id === activeQuestion ? { ...f, value: 'REFUSED' } : f);
        setChatHistory(prev => [...prev, { role: 'ai', text: '明白明白，是我冒昧了 🙏。我们继续聊回课程吧...' }]);
        setCircuitBreaker(true);
        setActiveQuestion(null);
      } else if (type === 'ignore' && extractedCount === 0) {
        setChatHistory(prev => [...prev, { role: 'ai', text: '好的，关于您问的问题...' }]);
        setActiveQuestion(null);
      } else {
        // Was it answered by extraction OR explicitly?
        const fieldWasFilledByExtraction = !!updates[activeQuestion];
        const isExplicitAnswer = type === activeQuestion; // User clicked specific button

        if (fieldWasFilledByExtraction || isExplicitAnswer) {
           // If explicit answer but not captured by extraction keywords, fill it now
           if (isExplicitAnswer && !fieldWasFilledByExtraction) {
              nextProfile = nextProfile.map(f => f.id === activeQuestion ? { ...f, value: userText } : f);
           }

           // Check incentive
           const fieldConfig = nextProfile.find(f => f.id === activeQuestion);
           if (fieldConfig?.incentive) {
              setChatHistory(prev => [...prev, { role: 'ai', text: '感谢信任！🎁 这是为您准备的《2025 AI 变现白皮书》，请查收！' }]);
           } else {
              if (extractedCount > 1) {
                setChatHistory(prev => [...prev, { role: 'ai', text: `收到！原来您是在${updates['city']?.includes('深圳') ? '深圳' : '那边'}发展的${updates['industry']?.includes('电商') ? '电商' : ''}同行呀，幸会！` }]);
              } else {
                setChatHistory(prev => [...prev, { role: 'ai', text: '收到，了解了。' }]);
              }
           }
        } else {
           // User replied something else
           setChatHistory(prev => [...prev, { role: 'ai', text: '好的。' }]);
        }
        setActiveQuestion(null);
      }
    } else {
      // Normal chat flow (User initiated "Natural Chat")
      if (extractedCount > 0) {
         setChatHistory(prev => [...prev, { role: 'ai', text: '哇，这经历很丰富呀！这一行确实非常有前景。' }]);
      } else {
         setChatHistory(prev => [...prev, { role: 'ai', text: '没问题，这个问题是这样的...' }]);
      }
    }

    // 5. Update State & Trigger Next Loop with LATEST data
    setProfile(nextProfile);
    if (!circuitBreaker && type !== 'refuse') {
        setTimeout(() => checkAndTriggerQuestion(nextProfile), 1000);
    }

    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Database className="text-indigo-600" />
              画像质量评分与追问 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Feature 16.1</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              具备 <span className="font-bold text-indigo-600">静态信息抽取</span> 能力：听懂了就不问，没听懂才追问
            </p>
          </div>
          <div className="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-600">
            {circuitBreaker ? (
              <span className="flex items-center gap-1 text-red-500 font-bold">
                <Lock className="w-3 h-3" /> 追问熔断生效 (24h)
              </span>
            ) : (
              <span className="flex items-center gap-1 text-green-600">
                <Zap className="w-3 h-3" /> 追问引擎就绪
              </span>
            )}
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[700px]">
          
          {/* Left: User Profile Dashboard */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
              <h2 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                <User className="w-5 h-5 text-indigo-600" />
                客户画像完整度 (Data Quality)
              </h2>
              
              {/* Score Bar */}
              <div className="mb-8">
                <div className="flex justify-between items-end mb-2">
                  <span className="text-4xl font-black text-slate-800">{qualityScore}</span>
                  <span className="text-sm font-bold text-slate-400 mb-1">/ 100 分</span>
                </div>
                <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-1000 ${qualityScore < 60 ? 'bg-orange-500' : 'bg-green-500'}`} 
                    style={{ width: `${qualityScore}%` }}
                  ></div>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center gap-1">
                  {qualityScore < 60 ? <AlertOctagon className="w-3 h-3 text-orange-500" /> : <CheckCircle className="w-3 h-3 text-green-500" />}
                  {qualityScore < 60 ? '画像模糊，建议启动追问' : '画像清晰，可精准营销'}
                </div>
              </div>

              {/* Field List */}
              <div className="space-y-3 flex-1 overflow-y-auto">
                {profile.map((field) => (
                  <div key={field.id} className={`p-3 rounded-lg border flex items-center justify-between transition-all ${field.value ? 'bg-slate-50 border-slate-200' : 'bg-white border-dashed border-indigo-300'}`}>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-slate-700">{field.label}</span>
                        {field.value === 'REFUSED' && <span className="text-[10px] bg-red-100 text-red-600 px-1.5 rounded">已拒绝</span>}
                        {field.id === activeQuestion && <span className="text-[10px] bg-indigo-100 text-indigo-600 px-1.5 rounded animate-pulse">正在追问...</span>}
                      </div>
                      {field.value && field.value !== 'REFUSED' && (
                        <div className="text-xs text-indigo-600 mt-1 truncate max-w-[150px]" title={field.value}>
                          已填: <span className="font-bold">{field.value}</span>
                        </div>
                      )}
                    </div>
                    {field.value && field.value !== 'REFUSED' ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-slate-200"></div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Active Chat */}
          <div className="lg:col-span-8 flex flex-col gap-6 h-full">
            <div className="bg-slate-100 rounded-xl border border-slate-200 h-full flex flex-col overflow-hidden relative">
               
               {/* Extraction Log Overlay */}
               {extractionLog && (
                 <div className="absolute top-4 right-4 z-20 w-auto max-w-[300px] bg-green-50 border border-green-200 shadow-lg rounded-xl p-3 animate-in slide-in-from-top-2 fade-in">
                   <div className="flex items-center gap-2 mb-2 pb-2 border-b border-green-100">
                     <Sparkles className="w-4 h-4 text-green-600" />
                     <span className="text-xs font-bold text-green-800">AI 静态信息抽取中...</span>
                   </div>
                   <div className="space-y-1">
                     {extractionLog.map((log, i) => (
                       <div key={i} className="text-[10px] text-green-700 font-mono leading-tight">{log}</div>
                     ))}
                   </div>
                 </div>
               )}

               {/* Chat Area */}
               <div className="flex-1 overflow-y-auto p-6 space-y-6">
                 {chatHistory.map((msg, idx) => (
                   <div key={idx} className={`flex flex-col ${msg.role === 'ai' ? 'items-start' : 'items-end'}`}>
                     <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold text-slate-400">{msg.role === 'ai' ? 'AI Assistant' : 'User'}</span>
                        {msg.isQuestion && (
                          <span className="text-[10px] bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded border border-yellow-200 flex items-center gap-1">
                            <Target className="w-3 h-3" /> 追问: {msg.fieldLabel}
                          </span>
                        )}
                     </div>
                     <div className={`
                       max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm
                       ${msg.role === 'ai' 
                         ? 'bg-white text-slate-700 rounded-tl-none border border-slate-200' 
                         : 'bg-indigo-600 text-white rounded-tr-none'}
                     `}>
                       {msg.text}
                     </div>
                   </div>
                 ))}
                 {isProcessing && (
                   <div className="flex justify-start">
                     <div className="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm flex gap-1 items-center">
                       <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
                       <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce delay-75"></div>
                       <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce delay-150"></div>
                     </div>
                   </div>
                 )}
                 <div ref={chatEndRef} />
               </div>

               {/* Interaction Area */}
               <div className="p-4 bg-white border-t border-slate-200 z-10">
                 {activeQuestion ? (
                   <div className="space-y-3">
                     <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-500 uppercase">用户模拟回复选项</span>
                        <span className="text-xs text-indigo-600 font-medium">当前缺失字段：{profile.find(f=>f.id===activeQuestion)?.label}</span>
                     </div>
                     <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                       <button 
                         onClick={() => handleUserReply(activeQuestion)}
                         className="p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700 font-medium hover:bg-green-100 text-left truncate"
                       >
                         ✅ 配合: "{USER_RESPONSES[activeQuestion]}"
                       </button>
                       <button 
                         onClick={() => handleUserReply('refuse')}
                         className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 font-medium hover:bg-red-100 text-left"
                       >
                         ⛔️ 拒绝: "{USER_RESPONSES.refuse}"
                       </button>
                       <button 
                         onClick={() => handleUserReply('ignore')}
                         className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-600 font-medium hover:bg-slate-100 text-left"
                       >
                         🙈 忽略: "{USER_RESPONSES.ignore}"
                       </button>
                     </div>
                   </div>
                 ) : (
                   <div className="flex gap-2">
                     <button 
                       onClick={() => handleUserReply("chat")}
                       disabled={qualityScore === 100 || circuitBreaker}
                       className="flex-1 py-3 bg-indigo-50 text-indigo-600 border border-indigo-200 rounded-lg font-bold text-sm hover:bg-indigo-100"
                     >
                       💬 普通对话 (触发追问)
                     </button>
                     <button 
                       onClick={() => handleUserReply("natural_chat")}
                       disabled={qualityScore === 100 || circuitBreaker}
                       className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-bold text-sm shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-2"
                     >
                       <Sparkles className="w-4 h-4" /> 模拟自然对话: "我在深圳做电商..."
                     </button>
                   </div>
                 )}
                 {qualityScore === 100 && (
                    <div className="text-center text-xs text-green-600 font-bold mt-2">🎉 恭喜！客户画像已全部补全</div>
                 )}
               </div>

            </div>
          </div>

        </div>
      </div>
    </div>
  );
}