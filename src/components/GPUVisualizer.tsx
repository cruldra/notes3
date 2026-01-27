import React, { useState, useEffect, useRef } from 'react';
import { Database, Cpu, ArrowRight, Zap, AlertTriangle, Activity, Gauge, Battery, Layers } from 'lucide-react';

const GPUVisualizer = () => {
  // Parameters state
  const [vramSize, setVramSize] = useState(12); // GB
  const [bandwidth, setBandwidth] = useState(300); // GB/s
  const [frequency, setFrequency] = useState(1500); // MHz
  const [computeUnits, setComputeUnits] = useState(80); // Abstract "Cores"
  const [tdp, setTdp] = useState(250); // Watts (New Parameter)
  const [cacheSize, setCacheSize] = useState(32); // MB (New Parameter)
  
  // Scenario State
  const [scenario, setScenario] = useState('gaming');

  // Derived Metrics & Simulation State
  const [utilization, setUtilization] = useState(0);
  const [fps, setFps] = useState(0);
  const [bottleneck, setBottleneck] = useState('');
  const [isOOM, setIsOOM] = useState(false); 
  const [isThrottled, setIsThrottled] = useState(false); // Power Throttling state
  const [effectiveFreq, setEffectiveFreq] = useState(0);
  
  // Scenario Configs
  const scenarios = {
    gaming: { 
      name: "3A游戏大作 (4K)", 
      reqVram: 10, 
      reqBandwidth: 400, 
      reqOps: 50000, 
      desc: "高频渲染需要大火猛炒(功耗)，大缓存能有效减少去显存取数据的次数。"
    },
    ai_inference: { 
      name: "AI大模型推理 (LLM)", 
      reqVram: 16, 
      reqBandwidth: 800, 
      reqOps: 30000, 
      desc: "缓存对大模型推理帮助有限（数据复用率低），主要还是拼硬带宽。"
    },
    rendering: { 
      name: "3D渲染 / 视频导出", 
      reqVram: 8, 
      reqBandwidth: 200, 
      reqOps: 80000, 
      desc: "长时间高负载，如果功耗不够（如笔记本），频率会很快降下来。"
    }
  };

  // Calculate TOPS
  const tops = ((effectiveFreq * computeUnits * 2) / 1000).toFixed(1);

  // Simulation Loop
  useEffect(() => {
    const calculatePerformance = () => {
      const currentTask = scenarios[scenario];
      
      // --- Logic 1: Power Throttling (TDP) ---
      // Simple physics model: Higher frequency needs more power.
      // Let's say 10 MHz needs roughly 1 Watt for this chip size approximation.
      const maxFreqByPower = tdp * 8; // e.g., 200W -> supports max 1600MHz logic
      
      let actualFreq = frequency;
      if (frequency > maxFreqByPower) {
        actualFreq = maxFreqByPower;
        setIsThrottled(true);
      } else {
        setIsThrottled(false);
      }
      setEffectiveFreq(actualFreq);

      // --- Logic 2: Memory Constraint (VRAM) ---
      if (vramSize < currentTask.reqVram) {
        setIsOOM(true);
        setFps(1);
        setUtilization(100);
        setBottleneck('显存爆满 (OOM)');
        return;
      } else {
        setIsOOM(false);
      }

      // --- Logic 3: Cache Benefit ---
      // Cache effectively reduces the bandwidth requirement
      // Gaming benefits a lot from Cache (Infinity Cache logic), AI less so.
      let cacheMultiplier = 1;
      if (scenario === 'gaming') cacheMultiplier = 1 + (cacheSize / 128); // 96MB cache adds ~75% efficiency
      if (scenario === 'rendering') cacheMultiplier = 1 + (cacheSize / 256); 
      if (scenario === 'ai_inference') cacheMultiplier = 1 + (cacheSize / 1024); // Minimal help for LLM linear reading

      const effectiveBandwidth = bandwidth * cacheMultiplier;

      // --- Logic 4: Throughput Calculation ---
      const bandwidthFactor = effectiveBandwidth / currentTask.reqBandwidth;
      const computeFactor = (actualFreq * computeUnits) / currentTask.reqOps;
      
      const performanceScore = Math.min(bandwidthFactor, computeFactor);
      
      // Determine Bottleneck
      if (isThrottled) {
        setBottleneck('功耗撞墙 (TDP不足，被迫降频)');
      } else if (bandwidthFactor < computeFactor) {
        setBottleneck(cacheSize > 64 ? '显存带宽 (即使有缓存也不够了)' : '显存带宽 (试着加点缓存?)');
      } else {
        setBottleneck('核心算力 (厨师切得太慢)');
      }

      // FPS
      let calculatedFps = Math.floor(performanceScore * 60); 
      if (calculatedFps > 144) calculatedFps = 144;
      setFps(calculatedFps);
      
      // Utilization
      const util = Math.min(100, Math.floor((performanceScore / computeFactor) * 100));
      setUtilization(bandwidthFactor < computeFactor ? Math.floor((bandwidthFactor / computeFactor) * 90) : 99);
    };

    calculatePerformance();
  }, [vramSize, bandwidth, frequency, computeUnits, tdp, cacheSize, scenario]);


  return (
    <div className="flex flex-col gap-6 p-6 bg-slate-900 text-slate-100 min-h-screen font-sans rounded-xl">
      
      {/* Header */}
      <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
          显卡性能可视化实验室 v2.0
        </h1>
        <p className="text-slate-400 text-sm">
          现在包含 <span className="text-yellow-400 font-bold">功耗 (TDP)</span> 和 <span className="text-orange-400 font-bold">缓存 (Cache)</span> 两个隐藏参数。
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Controls Column */}
        <div className="lg:col-span-1 space-y-6">
          
          <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Activity size={16} /> 1. 选择任务场景
            </h3>
            <div className="space-y-2">
              {Object.entries(scenarios).map(([key, val]) => (
                <button
                  key={key}
                  onClick={() => setScenario(key)}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-all border ${
                    scenario === key 
                      ? 'bg-blue-600/20 border-blue-500 text-white' 
                      : 'bg-slate-700/30 border-slate-700 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  <div className="font-medium">{val.name}</div>
                  <div className="text-xs opacity-70 mt-1">{val.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 space-y-6">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <Gauge size={16} /> 2. 硬件参数调节
            </h3>

            {/* TDP (New) */}
            <div className="p-3 bg-slate-700/30 rounded-lg border border-yellow-500/20">
               <div className="flex justify-between mb-2">
                <label className="text-sm font-bold flex items-center gap-2 text-yellow-400">
                  <Battery size={16} /> 功耗墙 (TDP)
                </label>
                <span className="text-sm font-bold">{tdp} W</span>
              </div>
              <input 
                type="range" min="50" max="450" step="10" 
                value={tdp} 
                onChange={(e) => setTdp(Number(e.target.value))}
                className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-yellow-500"
              />
              <p className="text-xs text-slate-400 mt-1">如果功耗太低，即使你把频率拉高，实际也会被强制降频。</p>
            </div>

             {/* Cache (New) */}
             <div className="p-3 bg-slate-700/30 rounded-lg border border-orange-500/20">
               <div className="flex justify-between mb-2">
                <label className="text-sm font-bold flex items-center gap-2 text-orange-400">
                  <Layers size={16} /> L2/L3 缓存 (Cache)
                </label>
                <span className="text-sm font-bold">{cacheSize} MB</span>
              </div>
              <input 
                type="range" min="0" max="128" step="4" 
                value={cacheSize} 
                onChange={(e) => setCacheSize(Number(e.target.value))}
                className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-orange-500"
              />
              <p className="text-xs text-slate-400 mt-1">缓存越大，对显存带宽的依赖就越小（尤其是玩游戏）。</p>
            </div>

            <hr className="border-slate-700" />

            {/* Standard Params */}
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-blue-300">显存大小</label>
                <span className="text-xs">{vramSize} GB</span>
              </div>
              <input type="range" min="4" max="48" step="2" value={vramSize} onChange={(e) => setVramSize(Number(e.target.value))} className="w-full h-1 bg-slate-600 accent-blue-500 rounded appearance-none" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-green-300">显存带宽</label>
                <span className="text-xs">{bandwidth} GB/s</span>
              </div>
              <input type="range" min="100" max="1500" step="50" value={bandwidth} onChange={(e) => setBandwidth(Number(e.target.value))} className="w-full h-1 bg-slate-600 accent-green-500 rounded appearance-none" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-purple-300">目标频率</label>
                <span className="text-xs">{frequency} MHz</span>
              </div>
              <input type="range" min="800" max="3000" step="50" value={frequency} onChange={(e) => setFrequency(Number(e.target.value))} className="w-full h-1 bg-slate-600 accent-purple-500 rounded appearance-none" />
            </div>
            
             <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-red-300">核心规模</label>
                <span className="text-xs">{computeUnits} Units</span>
              </div>
              <input type="range" min="20" max="160" step="10" value={computeUnits} onChange={(e) => setComputeUnits(Number(e.target.value))} className="w-full h-1 bg-slate-600 accent-red-500 rounded appearance-none" />
            </div>

          </div>
        </div>

        {/* Visualization Column */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* The Visual Stage */}
          <div className="bg-black/40 rounded-xl p-8 border border-slate-700 relative overflow-hidden flex-1 min-h-[400px] flex items-center justify-between gap-4">
            
            {/* 1. Memory Store (VRAM) */}
            <div className="flex flex-col items-center w-1/4 relative z-10">
              <div className="text-blue-300 font-bold mb-2 flex items-center gap-2">
                <Database /> VRAM
              </div>
              <div className="w-20 h-40 bg-slate-800 border-2 border-slate-600 rounded-lg relative overflow-hidden">
                <div 
                  className={`absolute bottom-0 w-full transition-all duration-500 ${isOOM ? 'bg-red-600' : 'bg-blue-500/60'}`}
                  style={{ height: `${Math.min((scenarios[scenario].reqVram / vramSize) * 100, 100)}%` }}
                />
                 <div className="absolute inset-0 flex items-center justify-center text-xs font-mono font-bold drop-shadow-md z-10 text-white">
                   {isOOM ? 'FULL' : `${(Math.min((scenarios[scenario].reqVram / vramSize) * 100, 100)).toFixed(0)}%`}
                </div>
              </div>
            </div>

            {/* 2. Bandwidth & Cache */}
            <div className="flex-1 flex flex-col items-center relative h-32 justify-center z-0">
              
              {/* Cache Block (New Visual) */}
              <div className="absolute -top-12 flex flex-col items-center animate-bounce">
                 <div className="text-orange-400 text-xs font-bold flex items-center gap-1 mb-1">
                    <Layers size={12}/> 缓存 {cacheSize}MB
                 </div>
                 <div 
                    className="bg-orange-500/20 border border-orange-500 rounded px-2 py-1 text-[10px] text-orange-200 transition-all duration-300"
                    style={{ opacity: cacheSize > 0 ? 1 : 0, transform: `scale(${1 + cacheSize/200})` }}
                 >
                    {cacheSize > 64 ? "高速缓冲区 (命中率高)" : "小型缓冲区"}
                 </div>
              </div>

              {/* The Highway */}
              <div className="text-green-300 font-bold mb-1 flex items-center gap-2 text-xs">
                 <ArrowRight size={14} /> 带宽通道
              </div>
              <div 
                className="w-full bg-slate-800/50 border-y-2 border-slate-600 relative overflow-hidden flex items-center"
                style={{ height: `${Math.min(bandwidth / 10, 60)}px`, transition: 'height 0.3s' }}
              >
                {!isOOM && Array.from({ length: 8 }).map((_, i) => (
                   <div 
                    key={i}
                    className="absolute bg-green-500 rounded-sm opacity-80"
                    style={{
                      width: '20px',
                      height: '60%',
                      left: `-${20}%`,
                      animationName: 'moveData',
                      animationDuration: `${2000 / bandwidth}s`,
                      animationTimingFunction: 'linear',
                      animationIterationCount: 'infinite',
                      animationDelay: `${i * (2000/bandwidth) / 8}s`
                    }}
                   />
                ))}
              </div>
            </div>

            {/* 3. GPU Core & Power */}
            <div className="flex flex-col items-center w-1/4 relative z-10">
              <div className="text-purple-300 font-bold mb-2 flex items-center gap-2">
                <Cpu /> Core
              </div>
              
              {/* Chip Visual */}
              <div className={`w-32 h-32 bg-slate-800 border-2 rounded-lg relative flex items-center justify-center transition-colors duration-300 ${isThrottled ? 'border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.4)]' : 'border-purple-500 shadow-[0_0_30px_rgba(168,85,247,0.2)]'}`}>
                
                {/* Spinner */}
                <div 
                  className={`absolute inset-2 border-4 border-dashed rounded-full ${isThrottled ? 'border-red-500/50' : 'border-purple-500/30'}`}
                  style={{ 
                    animationName: 'spin',
                    animationDuration: `${isOOM ? 10 : 3000 / effectiveFreq}s`,
                    animationTimingFunction: 'linear',
                    animationIterationCount: 'infinite'
                  }}
                ></div>
                
                {/* Throttled Icon */}
                {isThrottled && (
                    <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/60 rounded-lg">
                        <div className="text-center">
                            <Zap className="text-yellow-400 mx-auto animate-pulse" />
                            <span className="text-[10px] text-yellow-400 font-bold">POWER LIMIT</span>
                        </div>
                    </div>
                )}

                {/* Core Grid */}
                {!isThrottled && (
                    <div className="grid grid-cols-4 gap-1 p-4">
                    {Array.from({ length: 16 }).map((_, i) => (
                        <div 
                        key={i} 
                        className={`w-1.5 h-1.5 rounded-full ${
                            utilization > 50 && !isOOM ? 'bg-purple-400 animate-pulse' : 'bg-slate-600'
                        }`}
                        style={{ animationDelay: `${i * 0.1}s` }}
                        ></div>
                    ))}
                    </div>
                )}
              </div>
              
              {/* Actual Frequency Display */}
              <div className="mt-2 text-xs font-mono">
                <span className={isThrottled ? "text-red-400 font-bold" : "text-slate-400"}>
                    {effectiveFreq} MHz
                </span>
                {isThrottled && <span className="text-slate-500 line-through ml-2 text-[10px]">{frequency}</span>}
              </div>

            </div>

          </div>

          {/* Results */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-black/30 p-4 rounded-lg border border-slate-700">
              <h4 className="text-slate-400 text-xs uppercase mb-2">当前算力 (TOPS)</h4>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-mono font-bold text-white">{tops}</span>
                <span className="text-xs text-slate-500 mb-1">TOPS (Int8/FP16)</span>
              </div>
            </div>

            <div className={`p-4 rounded-lg border ${
              isOOM ? 'bg-red-900/20 border-red-500/50' :
              isThrottled ? 'bg-yellow-900/20 border-yellow-500/50' :
              bottleneck.includes('带宽') ? 'bg-orange-900/20 border-orange-500/50' : 
              'bg-blue-900/20 border-blue-500/50'
            }`}>
              <h4 className="flex items-center gap-2 text-xs uppercase font-bold mb-2 opacity-80">
                <AlertTriangle size={14} /> 瓶颈分析
              </h4>
              <div className="font-bold text-md text-white">
                {bottleneck}
              </div>
            </div>
          </div>

        </div>
      </div>
      
      <style>{`
        @keyframes moveData {
          0% { left: -20%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default GPUVisualizer;