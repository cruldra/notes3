import React, { useState, useEffect } from 'react';
import { 
  Users, 
  RefreshCw, 
  ArrowRight, 
  Filter, 
  UserPlus, 
  MessageSquare, 
  Clock, 
  Settings,
  Archive,
  CheckCircle,
  AlertCircle,
  Play
} from 'lucide-react';

// --- Mock Data ---
const MOCK_LEADS = [
  { id: 101, name: '用户_9527', days: 45, status: 'NOT_ADDED', oldRep: '销售A (新手)', score: 'B' },
  { id: 102, name: '用户_8812', days: 12, status: 'NOT_ADDED', oldRep: '销售B', score: 'A' },
  { id: 103, name: '用户_3301', days: 60, status: 'NOT_ADDED', oldRep: '销售C', score: 'S' },
  { id: 104, name: '用户_1102', days: 32, status: 'NOT_ADDED', oldRep: '销售A (新手)', score: 'A' },
  { id: 105, name: '用户_5599', days: 5,  status: 'NOT_ADDED', oldRep: '销售B', score: 'B' },
  { id: 106, name: '用户_7744', days: 90, status: 'NOT_ADDED', oldRep: '销售C', score: 'C' },
];

const NEW_REPS = [
  { id: 'revival_1', name: '资深复活专员·李', avatar: '👨‍💼', count: 0 },
  { id: 'revival_2', name: '资深复活专员·王', avatar: '👩‍💼', count: 0 },
];

const SCRIPTS = {
  default: "你好，我是助教...",
  revival: "【系统分配】同学你好，之前可能工作忙没顾上通过。最近我们课程更新了 3.0 版本，特意为你保留了老粉免费升级名额，点此领取..."
};

export default function SilentReactivationDemo() {
  // --- State ---
  const [threshold, setThreshold] = useState(30); // Days
  const [leads, setLeads] = useState(MOCK_LEADS);
  const [processingState, setProcessingState] = useState('IDLE'); // IDLE, FILTERING, ASSIGNING, COMPLETED
  const [activatedLeads, setActivatedLeads] = useState([]);
  const [logs, setLogs] = useState([]);

  // --- Actions ---
  const addLog = (msg) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev]);
  };

  const resetDemo = () => {
    setLeads(MOCK_LEADS);
    setActivatedLeads([]);
    setProcessingState('IDLE');
    setLogs([]);
    NEW_REPS.forEach(r => r.count = 0);
  };

  const runBatchJob = async () => {
    if (processingState !== 'IDLE') return;
    
    // Step 1: Filter
    setProcessingState('FILTERING');
    addLog(`⏳ 开始执行每日批处理任务...`);
    addLog(`🔍 扫描条件：入库时间 > ${threshold} 天 & 未加微`);
    
    await new Promise(r => setTimeout(r, 1000));
    
    const targets = leads.filter(l => l.days > threshold);
    const ignored = leads.filter(l => l.days <= threshold);
    
    addLog(`✅ 扫描完成：发现 ${targets.length} 条沉默线索，${ignored.length} 条未达标忽略`);
    
    if (targets.length === 0) {
      setProcessingState('COMPLETED');
      return;
    }

    // Step 2: Reassign
    setProcessingState('ASSIGNING');
    await new Promise(r => setTimeout(r, 1000));
    
    const processed = targets.map((lead, idx) => {
      const newRep = NEW_REPS[idx % NEW_REPS.length];
      newRep.count += 1; // Mutating for demo simplicity
      return { 
        ...lead, 
        newRep: newRep.name, 
        script: SCRIPTS.revival,
        status: 'REACTIVATING' 
      };
    });

    addLog(`🔄 执行流转：${processed.length} 条线索已从原销售剥离`);
    addLog(`📤 任务分发：平均分配给 ${NEW_REPS.length} 位复活专员`);

    // Step 3: Complete
    await new Promise(r => setTimeout(r, 800));
    setLeads(ignored); // Keep only ignored in left pool
    setActivatedLeads(processed); // Move targets to right pool
    setProcessingState('COMPLETED');
    addLog(`🚀 激活触达：已自动发送“复活话术”短信/企微任务`);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <RefreshCw className="text-indigo-600" />
              沉默线索再激活 <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-full font-bold">需求 1.3</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              针对入库超过 N 天仍未加微的“沉睡”线索，自动换人清洗与二次触达。
            </p>
          </div>
          <div className="flex items-center gap-4 bg-slate-50 p-2 rounded-xl border border-slate-100">
            <div className="flex items-center gap-2 px-2">
              <Settings className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-bold text-slate-600">沉默阈值:</span>
              <input 
                type="range" min="7" max="90" value={threshold} 
                onChange={(e) => setThreshold(Number(e.target.value))}
                disabled={processingState !== 'IDLE'}
                className="w-32 accent-indigo-600 cursor-pointer"
              />
              <span className="text-sm font-mono font-bold text-indigo-600 w-16 text-center">{threshold} 天</span>
            </div>
            <div className="h-6 w-px bg-slate-200"></div>
            <button 
              onClick={processingState === 'COMPLETED' ? resetDemo : runBatchJob}
              disabled={processingState === 'FILTERING' || processingState === 'ASSIGNING'}
              className={`px-4 py-2 rounded-lg font-bold text-sm flex items-center gap-2 transition-all
                ${processingState === 'COMPLETED' 
                  ? 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50' 
                  : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md shadow-indigo-200'}
                ${(processingState === 'FILTERING' || processingState === 'ASSIGNING') ? 'opacity-70 cursor-wait' : ''}
              `}
            >
              {processingState === 'COMPLETED' ? <RefreshCw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {processingState === 'COMPLETED' ? '重置数据' : '执行批处理'}
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[600px]">
          
          {/* Left: The "Silent Pool" (Dead Sea) */}
          <div className="lg:col-span-4 flex flex-col gap-4">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                <h2 className="font-bold text-slate-700 flex items-center gap-2">
                  <Archive className="w-4 h-4" /> 沉睡线索池
                </h2>
                <span className="text-xs bg-slate-200 px-2 py-1 rounded-full text-slate-600">{leads.length} 人</span>
              </div>
              <div className="p-4 space-y-3 overflow-y-auto flex-1 bg-slate-50/50">
                {leads.map(lead => (
                  <div key={lead.id} className={`
                    bg-white p-3 rounded-lg border shadow-sm transition-all duration-500
                    ${lead.days > threshold 
                      ? processingState === 'FILTERING' 
                        ? 'border-indigo-400 ring-2 ring-indigo-100 scale-105 z-10' 
                        : 'border-slate-200 hover:border-red-300'
                      : 'border-slate-100 opacity-60'}
                  `}>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="font-bold text-slate-800 text-sm">{lead.name}</div>
                        <div className="text-xs text-slate-400 mt-0.5">原属: {lead.oldRep}</div>
                      </div>
                      <div className={`text-xs font-mono font-bold px-2 py-1 rounded ${lead.days > threshold ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'}`}>
                        入库 {lead.days} 天
                      </div>
                    </div>
                    {lead.days > threshold && processingState === 'IDLE' && (
                      <div className="text-[10px] text-red-400 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" /> 符合再激活条件
                      </div>
                    )}
                  </div>
                ))}
                {leads.length === 0 && (
                  <div className="text-center text-slate-400 text-sm py-10">
                    线索已全部处理
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Middle: Logic Pipeline */}
          <div className="lg:col-span-3 flex flex-col gap-4">
            {/* Logic Visualizer */}
            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col items-center justify-center gap-8 relative overflow-hidden">
              
              {/* Connecting Pipes */}
              <div className="absolute top-1/2 left-0 w-full h-2 bg-slate-100 -z-0"></div>
              
              {/* Step 1: Filter */}
              <div className={`
                relative z-10 w-32 p-3 rounded-xl border-2 text-center transition-all duration-500
                ${processingState === 'FILTERING' ? 'bg-indigo-50 border-indigo-500 scale-110 shadow-lg' : 'bg-white border-slate-200'}
              `}>
                <Filter className={`w-6 h-6 mx-auto mb-2 ${processingState === 'FILTERING' ? 'text-indigo-600' : 'text-slate-300'}`} />
                <div className="text-xs font-bold text-slate-700">筛选器</div>
                <div className="text-[10px] text-slate-400">&gt; {threshold} Days</div>
              </div>

              <ArrowRight className={`w-6 h-6 text-slate-300 ${processingState !== 'IDLE' ? 'animate-pulse text-indigo-400' : ''}`} />

              {/* Step 2: Assign */}
              <div className={`
                relative z-10 w-32 p-3 rounded-xl border-2 text-center transition-all duration-500
                ${processingState === 'ASSIGNING' ? 'bg-purple-50 border-purple-500 scale-110 shadow-lg' : 'bg-white border-slate-200'}
              `}>
                <RefreshCw className={`w-6 h-6 mx-auto mb-2 ${processingState === 'ASSIGNING' ? 'text-purple-600 animate-spin' : 'text-slate-300'}`} />
                <div className="text-xs font-bold text-slate-700">自动流转</div>
                <div className="text-[10px] text-slate-400">Round Robin</div>
              </div>

            </div>

            {/* System Logs */}
            <div className="bg-slate-900 rounded-xl p-3 h-48 overflow-y-auto font-mono text-[10px] text-slate-300 shadow-inner">
              <div className="border-b border-slate-700 pb-1 mb-2 font-bold text-slate-500">SYSTEM LOGS</div>
              {logs.map((log, i) => (
                <div key={i} className="mb-1">{log}</div>
              ))}
              {logs.length === 0 && <span className="opacity-50">Waiting for command...</span>}
            </div>
          </div>

          {/* Right: The "Revival Squad" (Active Tasks) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-indigo-50 flex justify-between items-center">
                <h2 className="font-bold text-indigo-900 flex items-center gap-2">
                  <UserPlus className="w-4 h-4" /> 激活任务列表
                </h2>
                <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full font-bold">
                  {activatedLeads.length} 任务已下发
                </span>
              </div>
              <div className="p-4 space-y-4 overflow-y-auto flex-1 bg-slate-50/30">
                {activatedLeads.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-50">
                    <Clock className="w-12 h-12 mb-2" />
                    <p>等待任务分配...</p>
                  </div>
                )}
                {activatedLeads.map((item, idx) => (
                  <div key={idx} className="bg-white p-4 rounded-xl border border-indigo-100 shadow-sm animate-in slide-in-from-left-4 fade-in duration-500" style={{animationDelay: `${idx * 100}ms`}}>
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 text-xs font-bold">
                          {item.newRep[0]}
                        </div>
                        <div>
                          <div className="text-sm font-bold text-slate-800">分配给: {item.newRep}</div>
                          <div className="text-[10px] text-slate-400">来自: {item.oldRep} (沉睡{item.days}天)</div>
                        </div>
                      </div>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                    
                    {/* The Revival Script */}
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-100 relative group cursor-pointer hover:bg-slate-100 transition-colors">
                      <div className="absolute -top-2 left-2 bg-indigo-600 text-white text-[9px] px-1.5 rounded uppercase font-bold tracking-wider">
                        Auto Script
                      </div>
                      <p className="text-xs text-slate-600 leading-relaxed pt-1">
                        <MessageSquare className="w-3 h-3 inline mr-1 text-slate-400" />
                        "{item.script}"
                      </p>
                    </div>
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