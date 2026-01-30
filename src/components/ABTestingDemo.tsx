import React, { useState, useEffect, useRef } from 'react';
import { 
  GitBranch, 
  Users, 
  BarChart2, 
  Play, 
  StopCircle, 
  CheckCircle, 
  Trophy, 
  Zap, 
  MessageSquare, 
  Tag, 
  TrendingUp, 
  AlertCircle
} from 'lucide-react';

// --- Constants ---
const EXPERIMENTS = [
  {
    id: 'SCRIPT_TONE',
    name: '话术风格测试 (Tone)',
    type: 'SCRIPT',
    icon: <MessageSquare className="w-5 h-5 text-indigo-500" />,
    variantA: { name: 'A: 商务专业风', content: '您好，我是您的专属顾问。关于 AI Agent 课程的详细大纲已发送，请查收。', trueRate: 0.05 },
    variantB: { name: 'B: 亲切活泼风', content: '宝子！你要的 AI 搞钱秘籍来啦 📚~ 整理了一晚上，记得看哦！(内含彩蛋)', trueRate: 0.12 }
  },
  {
    id: 'PRICE_OFFER',
    name: '价格优惠测试 (Offer)',
    type: 'PRICE',
    icon: <Tag className="w-5 h-5 text-green-500" />,
    variantA: { name: 'A: 直接打折', content: '限时特惠：原价 599，立减 100，仅需 499 元！', trueRate: 0.08 },
    variantB: { name: 'B: 赠送礼包', content: '原价 599 元，下单额外赠送价值 199 元的《提示词词典》实体书！', trueRate: 0.09 }
  }
];

export default function ABTestingDemo() {
  // --- State ---
  const [selectedExpId, setSelectedExpId] = useState('SCRIPT_TONE');
  const [status, setStatus] = useState('IDLE'); // IDLE, RUNNING, FINISHED
  const [stats, setStats] = useState({
    A: { visitors: 0, conversions: 0 },
    B: { visitors: 0, conversions: 0 }
  });
  const [trafficSplit, setTrafficSplit] = useState(50); // % for A
  const [logs, setLogs] = useState([]);
  const [winner, setWinner] = useState(null);

  const experiment = EXPERIMENTS.find(e => e.id === selectedExpId);

  // --- Simulation Engine ---
  useEffect(() => {
    if (status !== 'RUNNING') return;

    const interval = setInterval(() => {
      // 1. Simulate a Visitor
      const isGroupA = Math.random() * 100 < trafficSplit;
      const groupKey = isGroupA ? 'A' : 'B';
      const variant = isGroupA ? experiment.variantA : experiment.variantB;

      // 2. Simulate Conversion
      // Add some randomness to the true rate to make it realistic
      const didConvert = Math.random() < variant.trueRate;

      setStats(prev => ({
        ...prev,
        [groupKey]: {
          visitors: prev[groupKey].visitors + 1,
          conversions: prev[groupKey].conversions + (didConvert ? 1 : 0)
        }
      }));

      // 3. Log (Only for conversions to reduce noise)
      if (didConvert) {
        addLog(groupKey, `转化成功！用户响应了 [${variant.name.split(':')[0]}]`);
      }

      // 4. Auto-conclude if significant data collected (Simulated)
      const totalVisitors = stats.A.visitors + stats.B.visitors;
      if (totalVisitors > 200 && !winner) {
        checkWinner();
      }

    }, 100); // Fast simulation

    return () => clearInterval(interval);
  }, [status, trafficSplit, experiment, winner, stats]);

  // --- Actions ---
  const addLog = (group, msg) => {
    setLogs(prev => [{ id: Date.now(), group, msg, time: new Date().toLocaleTimeString() }, ...prev.slice(0, 7)]);
  };

  const startExperiment = () => {
    setStats({ A: { visitors: 0, conversions: 0 }, B: { visitors: 0, conversions: 0 } });
    setLogs([]);
    setWinner(null);
    setStatus('RUNNING');
  };

  const stopExperiment = () => {
    setStatus('FINISHED');
    checkWinner();
  };

  const checkWinner = () => {
    const rateA = stats.A.visitors > 0 ? stats.A.conversions / stats.A.visitors : 0;
    const rateB = stats.B.visitors > 0 ? stats.B.conversions / stats.B.visitors : 0;
    
    // Simple logic for demo
    if (rateA > rateB * 1.1) setWinner('A');
    else if (rateB > rateA * 1.1) setWinner('B');
    else setWinner('TIE');
  };

  const applyWinner = () => {
    alert(`已将流量 100% 切换至 [${winner === 'A' ? experiment.variantA.name : experiment.variantB.name}]！策略更新生效。`);
    setStatus('IDLE');
    setWinner(null);
    setStats({ A: { visitors: 0, conversions: 0 }, B: { visitors: 0, conversions: 0 } });
  };

  // --- Helpers ---
  const getRate = (group) => {
    const s = stats[group];
    if (s.visitors === 0) return '0.0%';
    return ((s.conversions / s.visitors) * 100).toFixed(1) + '%';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <GitBranch className="text-indigo-600" />
              A/B 效果对比实验室 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              数据驱动决策 • 实时对比不同策略的转化效果 (CTR/CVR)
            </p>
          </div>
          <div className="flex gap-3">
             {status === 'RUNNING' ? (
                <button 
                  onClick={stopExperiment}
                  className="px-6 py-2 bg-red-50 text-red-600 border border-red-200 rounded-lg font-bold hover:bg-red-100 flex items-center gap-2"
                >
                  <StopCircle className="w-4 h-4" /> 停止实验
                </button>
             ) : (
                <button 
                  onClick={startExperiment}
                  className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center gap-2 transition-all"
                >
                  <Play className="w-4 h-4" /> 开始测试
                </button>
             )}
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[700px]">
          
          {/* Left: Experiment Config */}
          <div className="lg:col-span-4 flex flex-col gap-4 h-full">
            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
              <h2 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                <Zap className="w-4 h-4" /> 实验配置台
              </h2>
              
              <div className="space-y-6 flex-1">
                {/* 1. Select Experiment */}
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-3">
                    选择测试场景
                  </label>
                  <div className="space-y-2">
                    {EXPERIMENTS.map(exp => (
                      <button
                        key={exp.id}
                        onClick={() => { setSelectedExpId(exp.id); setStatus('IDLE'); setStats({A:{visitors:0,conversions:0},B:{visitors:0,conversions:0}}); setWinner(null); }}
                        disabled={status === 'RUNNING'}
                        className={`
                          w-full p-3 rounded-lg border text-left flex items-center gap-3 transition-all
                          ${selectedExpId === exp.id 
                            ? 'bg-indigo-50 border-indigo-500 ring-1 ring-indigo-500' 
                            : 'bg-white border-slate-200 hover:border-indigo-300'}
                          ${status === 'RUNNING' ? 'opacity-50 cursor-not-allowed' : ''}
                        `}
                      >
                        <div className={`p-2 rounded-full ${selectedExpId === exp.id ? 'bg-white' : 'bg-slate-100'}`}>
                          {exp.icon}
                        </div>
                        <div>
                          <div className="font-bold text-sm text-slate-800">{exp.name}</div>
                          <div className="text-[10px] text-slate-500">Compare: {exp.type === 'SCRIPT' ? 'Content Tone' : 'Pricing Strategy'}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* 2. Traffic Split */}
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-3">
                    流量切分比例 (Traffic Split)
                  </label>
                  <div className="bg-slate-100 p-4 rounded-xl">
                    <div className="flex justify-between text-xs font-bold mb-2">
                      <span className="text-blue-600">Group A: {trafficSplit}%</span>
                      <span className="text-orange-600">Group B: {100 - trafficSplit}%</span>
                    </div>
                    <input 
                      type="range" min="10" max="90" step="10"
                      value={trafficSplit}
                      onChange={(e) => setTrafficSplit(Number(e.target.value))}
                      disabled={status === 'RUNNING'}
                      className="w-full accent-indigo-600 cursor-pointer"
                    />
                  </div>
                </div>

                {/* 3. Variant Details */}
                <div className="bg-slate-50 rounded-lg p-4 border border-slate-100 text-xs space-y-3">
                   <div className="flex gap-2">
                     <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-bold whitespace-nowrap">A 组策略</span>
                     <p className="text-slate-600 leading-tight">"{experiment.variantA.content}"</p>
                   </div>
                   <div className="border-t border-slate-200"></div>
                   <div className="flex gap-2">
                     <span className="bg-orange-100 text-orange-700 px-2 py-0.5 rounded font-bold whitespace-nowrap">B 组策略</span>
                     <p className="text-slate-600 leading-tight">"{experiment.variantB.content}"</p>
                   </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Live Monitor */}
          <div className="lg:col-span-8 flex flex-col gap-6 h-full">
            
            {/* Real-time Charts */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex-1 relative overflow-hidden">
               <div className="flex justify-between items-start mb-8">
                 <h2 className="font-bold text-slate-800 flex items-center gap-2">
                   <BarChart2 className="w-5 h-5 text-indigo-600" />
                   实时数据大屏
                 </h2>
                 {status === 'RUNNING' && (
                   <span className="flex items-center gap-2 text-xs font-mono text-green-600 bg-green-50 px-2 py-1 rounded animate-pulse">
                     <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                     LIVE TRAFFIC INCOMING
                   </span>
                 )}
               </div>

               <div className="grid grid-cols-2 gap-8 h-48">
                 {/* Group A Stats */}
                 <div className="relative bg-blue-50 rounded-xl border border-blue-100 p-5 flex flex-col justify-between overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                      <Users className="w-20 h-20 text-blue-600" />
                    </div>
                    <div>
                      <div className="text-blue-800 font-bold mb-1 flex items-center gap-2">
                        <span className="bg-blue-600 text-white w-6 h-6 flex items-center justify-center rounded text-xs">A</span>
                        {experiment.variantA.name}
                      </div>
                      <div className="text-xs text-blue-400">Sample Size: {stats.A.visitors}</div>
                    </div>
                    <div className="mt-4">
                      <div className="text-3xl font-black text-blue-700">{getRate('A')}</div>
                      <div className="text-xs font-bold text-blue-400 uppercase">Conversion Rate</div>
                    </div>
                    {/* Visual Bar */}
                    <div className="absolute bottom-0 left-0 w-full h-2 bg-blue-200">
                      <div className="h-full bg-blue-600 transition-all duration-500" style={{ width: getRate('A') }}></div>
                    </div>
                 </div>

                 {/* Group B Stats */}
                 <div className="relative bg-orange-50 rounded-xl border border-orange-100 p-5 flex flex-col justify-between overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                      <Users className="w-20 h-20 text-orange-600" />
                    </div>
                    <div>
                      <div className="text-orange-800 font-bold mb-1 flex items-center gap-2">
                        <span className="bg-orange-500 text-white w-6 h-6 flex items-center justify-center rounded text-xs">B</span>
                        {experiment.variantB.name}
                      </div>
                      <div className="text-xs text-orange-400">Sample Size: {stats.B.visitors}</div>
                    </div>
                    <div className="mt-4">
                      <div className="text-3xl font-black text-orange-600">{getRate('B')}</div>
                      <div className="text-xs font-bold text-orange-400 uppercase">Conversion Rate</div>
                    </div>
                    {/* Visual Bar */}
                    <div className="absolute bottom-0 left-0 w-full h-2 bg-orange-200">
                      <div className="h-full bg-orange-500 transition-all duration-500" style={{ width: getRate('B') }}></div>
                    </div>
                 </div>
               </div>

               {/* Winner Declaration Overlay */}
               {winner && (
                 <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center z-10 animate-in fade-in zoom-in duration-500">
                    <Trophy className="w-16 h-16 text-yellow-500 mb-4 drop-shadow-md animate-bounce" />
                    <h3 className="text-2xl font-black text-slate-800 mb-2">
                      胜出者: {winner === 'A' ? 'A 组' : winner === 'B' ? 'B 组' : '平局'}
                    </h3>
                    <p className="text-slate-500 mb-6 text-sm">
                      {winner === 'TIE' ? '数据差异不显著，建议继续观察' : `[${winner === 'A' ? experiment.variantA.name : experiment.variantB.name}] 的转化率显著更高。`}
                    </p>
                    {winner !== 'TIE' && (
                      <button 
                        onClick={applyWinner}
                        className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white rounded-full font-bold shadow-lg shadow-green-200 flex items-center gap-2 transition-transform active:scale-95"
                      >
                        <CheckCircle className="w-5 h-5" /> 立即推全该策略 (Apply to All)
                      </button>
                    )}
                 </div>
               )}
            </div>

            {/* Live Logs */}
            <div className="bg-slate-900 text-slate-300 rounded-xl p-4 h-40 overflow-hidden flex flex-col shadow-inner border border-slate-800">
               <div className="flex justify-between items-center mb-2 border-b border-slate-700 pb-2">
                 <h3 className="text-xs font-bold text-slate-400 uppercase flex items-center gap-2">
                   <TrendingUp className="w-3 h-3" /> 实时转化日志
                 </h3>
                 <span className="text-[10px] text-slate-500">Total Conversions: {stats.A.conversions + stats.B.conversions}</span>
               </div>
               <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1.5">
                 {logs.length === 0 && <span className="text-slate-600 italic">Waiting for traffic...</span>}
                 {logs.map((log) => (
                   <div key={log.id} className="flex gap-3 animate-in slide-in-from-left-2">
                     <span className="text-slate-500 opacity-50">[{log.time}]</span>
                     <span className={`font-bold ${log.group === 'A' ? 'text-blue-400' : 'text-orange-400'}`}>
                       [{log.group}组]
                     </span>
                     <span>{log.msg}</span>
                   </div>
                 ))}
               </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}