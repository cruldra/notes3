import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageCircle, 
  Sparkles, 
  Settings, 
  Smile, 
  Briefcase, 
  Zap, 
  BarChart2, 
  Edit3, 
  RefreshCw,
  User,
  Bot,
  Wand2
} from 'lucide-react';

// --- Configuration ---
const STYLES = {
  'CASUAL': {
    label: '活泼/亲切 (Casual)',
    icon: <Smile className="w-5 h-5 text-orange-500" />,
    color: 'bg-orange-100 text-orange-700 border-orange-300',
    description: '使用 Emoji、语气词（哈、呢、宝子），拉近距离。',
    promptMod: 'Using a lively, friendly tone with emojis. Call user "宝子" or "亲".'
  },
  'BUSINESS': {
    label: '商务/专业 (Business)',
    icon: <Briefcase className="w-5 h-5 text-blue-600" />,
    color: 'bg-blue-100 text-blue-700 border-blue-300',
    description: '用词精准、客观，无表情包，强调效率与专业度。',
    promptMod: 'Use professional, concise business language. No emojis.'
  },
  'ANIME': {
    label: '二次元 (Anime)',
    icon: <Sparkles className="w-5 h-5 text-purple-500" />,
    color: 'bg-purple-100 text-purple-700 border-purple-300',
    description: '使用颜文字 (QwQ)、可爱语气，适合年轻群体。',
    promptMod: 'Use cute "Anime" style with Kaomoji like (*^▽^*) and soft endings.'
  },
  'DEFAULT': {
    label: '标准/默认 (Standard)',
    icon: <User className="w-5 h-5 text-slate-500" />,
    color: 'bg-slate-100 text-slate-600 border-slate-300',
    description: '系统默认的中性语气，不功不过。',
    promptMod: 'Use standard polite customer service tone.'
  }
};

const SCENARIOS = [
  {
    id: 1,
    name: '测试: 活泼党',
    userText: "宝子！那个 399 的课还有名额嘛？😭 昨晚忘买了绝绝子...",
    expectedStyle: 'CASUAL',
    standardReply: "您好，399元的课程目前还有少量名额，请尽快下单。",
    styledReply: "宝子别哭！😭 帮你查了下还有最后几个坑位！幸好你来得及时，不然真就绝绝子了~ 快冲！🚀"
  },
  {
    id: 2,
    name: '测试: 商务党',
    userText: "请确认一下《AI 提效》课程的开票类目及税点，我们需要走对公报销流程。",
    expectedStyle: 'BUSINESS',
    standardReply: "您好，开票类目是技术服务费，税点是6%，支持对公转账。",
    styledReply: "收到。开票类目为【技术服务费】，税率为 6%。支持对公账户汇款，具体开票资料我稍后发送至您邮箱，请查收。"
  },
  {
    id: 3,
    name: '测试: 二次元',
    userText: "呜呜呜，错过直播了QAQ... 助教君有没有回放呀？求求了Orz",
    expectedStyle: 'ANIME',
    standardReply: "您好，直播有回放的，稍后发给您链接。",
    styledReply: "摸摸头不哭不哭 (*/ω＼*)！回放早就给助教君准备好啦✨~ 链接这就发射给你 biu biu biu ❤️！"
  }
];

export default function StyleInferenceDemo() {
  // --- State ---
  const [messages, setMessages] = useState([]);
  const [detectedStyle, setDetectedStyle] = useState('DEFAULT');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [adaptationEnabled, setAdaptationEnabled] = useState(true); // Toggle for feature
  const [metrics, setMetrics] = useState({ emojiDensity: 0, slangCount: 0, sentenceLength: 0 });
  const [manualOverride, setManualOverride] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- Logic ---
  const handleScenario = async (scenario) => {
    // 1. User Message
    const userMsg = { role: 'user', text: scenario.userText };
    setMessages(prev => [...prev, userMsg]);
    setIsAnalyzing(true);
    setManualOverride(false);

    // 2. Simulate Analysis (The "LLM" Step)
    await new Promise(r => setTimeout(r, 800));
    
    // Mock metrics calculation
    const isCasual = scenario.expectedStyle === 'CASUAL';
    const isAnime = scenario.expectedStyle === 'ANIME';
    const isBusiness = scenario.expectedStyle === 'BUSINESS';

    setMetrics({
      emojiDensity: isCasual ? 0.8 : isAnime ? 0.2 : 0,
      slangCount: isCasual ? 3 : isAnime ? 1 : 0,
      sentenceLength: isBusiness ? 45 : 15,
      kaomoji: isAnime // Special flag for anime
    });

    setDetectedStyle(scenario.expectedStyle);
    setIsAnalyzing(false);

    // 3. AI Response
    await new Promise(r => setTimeout(r, 600));
    const replyText = adaptationEnabled ? scenario.styledReply : scenario.standardReply;
    const aiMsg = { 
      role: 'ai', 
      text: replyText, 
      styleUsed: adaptationEnabled ? scenario.expectedStyle : 'DEFAULT' 
    };
    setMessages(prev => [...prev, aiMsg]);
  };

  const handleManualChange = (newStyle) => {
    setDetectedStyle(newStyle);
    setManualOverride(true);
  };

  const clearChat = () => {
    setMessages([]);
    setDetectedStyle('DEFAULT');
    setMetrics({ emojiDensity: 0, slangCount: 0, sentenceLength: 0 });
    setManualOverride(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Wand2 className="text-indigo-600" />
              风格偏好推测引擎 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">P1 Feature</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于 LLM 的文本特征分析 • 实现“千人千面”的风格化沟通
            </p>
          </div>
          
          <div className="flex items-center gap-4 bg-slate-100 p-2 rounded-xl">
            <span className="text-sm font-bold text-slate-600 pl-2">风格自适应开关:</span>
            <button 
              onClick={() => setAdaptationEnabled(!adaptationEnabled)}
              className={`
                relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none
                ${adaptationEnabled ? 'bg-indigo-600' : 'bg-slate-300'}
              `}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${adaptationEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[700px]">
          
          {/* Left: Chat Area */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <span className="font-bold text-slate-700 flex items-center gap-2">
                  <MessageCircle className="w-4 h-4" /> 模拟对话
                </span>
                <button onClick={clearChat} className="text-xs text-slate-400 hover:text-red-500 flex items-center gap-1">
                  <RefreshCw className="w-3 h-3" /> 清空
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/30">
                {messages.length === 0 && (
                  <div className="text-center text-slate-400 mt-20 text-sm italic">
                    点击下方按钮，测试不同用户的说话风格...
                  </div>
                )}
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'ai' ? 'justify-start' : 'justify-end'}`}>
                    <div className="flex flex-col max-w-[85%]">
                      {msg.role === 'ai' && (
                         <span className="text-[10px] text-slate-400 mb-1 ml-1 flex items-center gap-1">
                           <Bot className="w-3 h-3" /> 
                           {msg.styleUsed === 'DEFAULT' ? '标准回复' : `Adapted: ${STYLES[msg.styleUsed].label.split(' ')[0]}`}
                         </span>
                      )}
                      <div className={`
                        p-3 rounded-2xl text-sm shadow-sm leading-relaxed
                        ${msg.role === 'ai' 
                          ? `${msg.styleUsed === 'DEFAULT' ? 'bg-white border border-slate-200' : STYLES[msg.styleUsed].color} rounded-tl-none` 
                          : 'bg-slate-800 text-white rounded-tr-none'}
                      `}>
                        {msg.text}
                      </div>
                    </div>
                  </div>
                ))}
                {isAnalyzing && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                      <Zap className="w-4 h-4 text-indigo-500 animate-pulse" />
                      <span className="text-xs text-slate-500">AI 正在分析对方语气成分...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Scenario Inputs */}
              <div className="p-3 bg-white border-t border-slate-100 grid grid-cols-1 gap-2">
                <p className="text-[10px] font-bold text-slate-400 uppercase ml-1">输入模拟 (Select User Persona)</p>
                {SCENARIOS.map(s => (
                  <button 
                    key={s.id}
                    onClick={() => handleScenario(s)}
                    disabled={isAnalyzing}
                    className="text-left px-3 py-2 rounded-lg border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50 transition-all text-xs text-slate-600 flex justify-between items-center group"
                  >
                    <span className="font-bold w-20">{s.name}</span>
                    <span className="truncate flex-1 opacity-80">"{s.userText.substring(0, 20)}..."</span>
                    <ArrowRight className="w-3 h-3 text-slate-300 group-hover:text-indigo-500" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Analysis Dashboard */}
          <div className="lg:col-span-7 flex flex-col gap-6 h-full">
            
            {/* 1. Real-time Analysis Panel */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 relative overflow-hidden">
               <div className="flex justify-between items-center mb-6">
                 <h2 className="font-bold text-slate-800 flex items-center gap-2">
                   <BarChart2 className="w-5 h-5 text-indigo-600" />
                   特征提取分析 (Feature Extraction)
                 </h2>
                 {isAnalyzing && <span className="text-xs text-indigo-600 animate-pulse font-bold">● Analyzing...</span>}
               </div>

               <div className="grid grid-cols-3 gap-6">
                 {/* Metric 1: Emoji */}
                 <div className="bg-slate-50 rounded-lg p-4 text-center border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase font-bold mb-2">Emoji 密度</div>
                    <div className="h-16 flex items-end justify-center gap-1 mb-2">
                      <div className="w-3 bg-indigo-200 rounded-t-sm h-full relative overflow-hidden">
                         <div className="absolute bottom-0 w-full bg-indigo-500 transition-all duration-1000" style={{height: `${metrics.emojiDensity * 100}%`}}></div>
                      </div>
                    </div>
                    <div className="text-lg font-black text-slate-700">{(metrics.emojiDensity * 100).toFixed(0)}%</div>
                 </div>

                 {/* Metric 2: Slang */}
                 <div className="bg-slate-50 rounded-lg p-4 text-center border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase font-bold mb-2">口癖/热梗检测</div>
                    <div className={`text-3xl font-black my-3 ${metrics.slangCount > 0 ? 'text-orange-500' : 'text-slate-300'}`}>
                       {metrics.slangCount}
                    </div>
                    <div className="text-xs text-slate-400">Detected Words</div>
                 </div>

                 {/* Metric 3: Sentence */}
                 <div className="bg-slate-50 rounded-lg p-4 text-center border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase font-bold mb-2">平均句长</div>
                    <div className="text-lg font-black text-slate-700 mt-4">{metrics.sentenceLength}</div>
                    <div className="text-xs text-slate-400">Chars / Msg</div>
                    {metrics.sentenceLength > 30 && <span className="text-[10px] text-blue-500 font-bold">Long (Formal)</span>}
                 </div>
               </div>

               {metrics.kaomoji && (
                 <div className="absolute top-4 right-4 rotate-12 bg-purple-100 text-purple-600 px-2 py-1 rounded text-xs font-bold border border-purple-200 animate-bounce">
                   颜文字 Detected! (QwQ)
                 </div>
               )}
            </div>

            {/* 2. Style Tagging & Override */}
            <div className="bg-slate-900 text-white rounded-xl p-6 shadow-lg border border-slate-800 flex-1 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                 <h2 className="font-bold flex items-center gap-2">
                   <Zap className="w-5 h-5 text-yellow-400" />
                   当前推测风格 (Current Tone)
                 </h2>
                 {manualOverride && (
                   <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded border border-yellow-500/30">
                     Manual Override Active
                   </span>
                 )}
              </div>

              {/* Active Style Card */}
              <div className="flex items-start gap-4 mb-8">
                 <div className={`p-4 rounded-xl ${STYLES[detectedStyle].color.replace('bg-', 'bg-opacity-20 bg-').replace('border-', 'border-opacity-50 border-')} border-2 transition-all duration-500`}>
                   {STYLES[detectedStyle].icon}
                 </div>
                 <div>
                   <h3 className={`text-xl font-bold transition-all duration-300 ${isAnalyzing ? 'blur-sm' : ''}`}>
                     {STYLES[detectedStyle].label}
                   </h3>
                   <p className="text-sm text-slate-400 mt-1 leading-relaxed max-w-md">
                     {STYLES[detectedStyle].description}
                   </p>
                   {/* Prompt Injection Visualization */}
                   <div className="mt-3 bg-black/30 p-2 rounded border border-white/10 text-[10px] font-mono text-green-400">
                     <span className="text-slate-500">System Prompt Injection: </span>
                     {adaptationEnabled ? `"${STYLES[detectedStyle].promptMod}"` : '"(Feature Disabled)"'}
                   </div>
                 </div>
              </div>

              {/* Manual Override Controls */}
              <div className="mt-auto pt-6 border-t border-slate-700">
                <p className="text-xs font-bold text-slate-500 uppercase mb-3 flex items-center gap-2">
                  <Edit3 className="w-3 h-3" /> 人工修正 (Human Feedback)
                </p>
                <div className="grid grid-cols-4 gap-2">
                  {Object.entries(STYLES).map(([key, style]) => (
                    <button
                      key={key}
                      onClick={() => handleManualChange(key)}
                      className={`
                        py-2 rounded-lg text-xs font-medium transition-all border
                        ${detectedStyle === key 
                          ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-900/50' 
                          : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}
                      `}
                    >
                      {style.label.split(' ')[0]}
                    </button>
                  ))}
                </div>
              </div>

            </div>

          </div>

        </div>
      </div>
    </div>
  );
}

// Helper Icon
function ArrowRight({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  );
}