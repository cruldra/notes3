import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Eye, 
  Thermometer, 
  Wind, 
  Activity, 
  CheckCircle, 
  ShieldAlert, 
  Server, 
  FileText, 
  Clock, 
  MapPin,
  Play,
  RotateCcw,
  ChevronRight,
  BrainCircuit,
  Zap,
  TrendingUp,
  Siren
} from 'lucide-react';

// --- 类型定义 ---
type Step = {
  id: number;
  phase: string;
  title: string;
  time: string;
  description: string;
  activeSensors: string[]; // 用于高亮相关传感器
  riskLevel: string; // 风险等级
  systemLog: string[];
};

// --- 核心组件 ---
export default function FireAuthDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlay, setIsAutoPlay] = useState(false);

  // 模拟场景数据：B2 地库配电房阴燃火灾（真火早期预判与升级）
  const steps: Step[] = [
    {
      id: 0,
      phase: "L0 常态监控",
      title: "全域扫描中",
      time: "03:00:00 (T=-10s)",
      description: "凌晨 03:00，Sup-01 正在监控 B2 配电房。当前为无人值守时段，环境参数基线平稳。",
      activeSensors: [],
      riskLevel: "Normal",
      systemLog: [
        "> System Status: ONLINE (Night Mode)",
        "> Location: B2-E-01 (Main Power Room)",
        "> Baseline: Temp 25°C, CO 0ppm",
        "> Monitoring: 120 Sensors Active"
      ]
    },
    {
      id: 1,
      phase: "L1 弱信号捕捉",
      title: "微弱异常感知 (T+1min)",
      time: "03:01:00 (T+60s)",
      description: "烟感尚未触发(0%)。但高灵敏度气体传感器检测到 CO 缓慢上升至 15ppm，热成像发现配电柜局部升温至 45°C。系统识别到“异常变化率”。",
      activeSensors: ['gas', 'thermal'],
      riskLevel: "Watch",
      systemLog: [
        "! NOTICE: Abnormal Rate of Rise (RoR) detected",
        "> Gas Sensor: CO 0 -> 15ppm (Rising)",
        "> Thermal: Local Hotspot 45°C detected",
        "> Smoke Sensor: 0% (Silent)",
        "> Action: Enter 'Focus Mode' (1s/poll)"
      ]
    },
    {
      id: 2,
      phase: "L2 上下文与推理",
      title: "阴燃模式匹配 (T+2min)",
      time: "03:02:00 (T+120s)",
      description: "查无夜间检修工单，排除人为干扰。综合无明火、CO持续上升、局部热点特征，判定为“电气线路阴燃”。",
      activeSensors: ['gas', 'thermal', 'visual', 'context'],
      riskLevel: "P1 (Pending)",
      systemLog: [
        "> Context: 3:00 AM, No Work Order (Permit Check: Fail)",
        "> Visual: No Flame, Grey wispy smoke",
        "> Evidence Chain: No Flame + Heating + CO Rising",
        "> Inference: Match Pattern 'Electrical Smoldering' (Conf: 88%)"
      ]
    },
    {
      id: 3,
      phase: "L3 分级响应",
      title: "P1级 局部响应 (T+3.5min)",
      time: "03:03:30 (T+210s)",
      description: "确认为真火隐患，定级 P1 (重要)。策略：仅在配电房及中控室报警，派夜班保安携带灭火器核实，避免全楼恐慌。",
      activeSensors: ['decision'],
      riskLevel: "P1 (Important)",
      systemLog: [
        "> Risk Assessment: P1 (True Fire, Early Stage)",
        "> Strategy: Silent Mode (No Public Alarm)",
        "> Action: Local Alarm (Control Room + B2)",
        "> Dispatch: Guard 'Zhang' (w/ Extinguisher) -> B2-E-01"
      ]
    },
    {
      id: 4,
      phase: "L4 态势恶化",
      title: "动态升级判定 (T+4.5min)",
      time: "03:04:30 (T+270s)",
      description: "保安赶路中，传感器示数激增：CO > 50ppm，温度 > 70°C。系统预判“即将转为明火/轰燃”，自动触发升级机制。",
      activeSensors: ['gas', 'thermal', 'smoke'],
      riskLevel: "Escalating -> P0",
      systemLog: [
        "! CRITICAL: Threshold Breached",
        "> CO: 55ppm (>50), Temp: 72°C (>70)",
        "> Prediction: Flashover imminent in 60s",
        "> Decision: ESCALATE TO P0 (EMERGENCY)"
      ]
    },
    {
      id: 5,
      phase: "L4 自动阻断",
      title: "P0级 全楼联动 (T+5min)",
      time: "03:05:00 (T+300s)",
      description: "执行 P0 级预案：自动切断非消防电源（防短路扩大），启动排烟风机，开启全楼声光报警与疏散广播。",
      activeSensors: ['execution'],
      riskLevel: "P0 (Critical)",
      systemLog: [
        "> Action: CUT POWER (Non-Fire Load) [SUCCESS]",
        "> Action: Start Smoke Exhaust Fan [SUCCESS]",
        "> Action: Close Fire Doors (B2 Zone)",
        "> Action: FULL BUILDING ALARM & EVACUATION"
      ]
    }
  ];

  // 自动播放逻辑
  useEffect(() => {
    let interval: any;
    if (isAutoPlay && currentStep < steps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => prev + 1);
      }, 4000); // 延长每步时间以便阅读
    } else if (currentStep === steps.length - 1) {
      setIsAutoPlay(false);
    }
    return () => clearInterval(interval);
  }, [isAutoPlay, currentStep]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) setCurrentStep(prev => prev + 1);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsAutoPlay(false);
  };

  const currentData = steps[currentStep];

  // 辅助函数：根据步骤返回传感器模拟数值
  const getSensorData = (stepId: number) => {
    if (stepId === 0) return { co: 0, temp: 25, smoke: 0 };
    if (stepId === 1) return { co: 15, temp: 45, smoke: 0 }; // 弱信号
    if (stepId === 2) return { co: 25, temp: 52, smoke: 1.5 }; // 阴燃加剧
    if (stepId === 3) return { co: 35, temp: 60, smoke: 2.8 }; // P1
    if (stepId >= 4) return { co: 55, temp: 72, smoke: 5.5 }; // P0 恶化
    return { co: 0, temp: 25, smoke: 0 };
  };

  const sensorValues = getSensorData(currentStep);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-8 flex flex-col items-center">
      
      {/* 顶部标题栏 */}
      <header className="w-full max-w-6xl mb-8 flex flex-col md:flex-row justify-between items-center border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-red-500 to-orange-400 bg-clip-text text-transparent flex items-center gap-3">
            <Zap className="w-8 h-8 text-orange-500" />
            Sup-01 数智主管
          </h1>
          <p className="text-slate-400 mt-1">场景：B2 地库配电房阴燃火灾（真火早期预判与动态升级）</p>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <button 
            onClick={() => setIsAutoPlay(!isAutoPlay)}
            className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-all ${isAutoPlay ? 'bg-amber-600 hover:bg-amber-500' : 'bg-blue-600 hover:bg-blue-500'}`}
          >
            {isAutoPlay ? <span className="flex items-center gap-2">暂停演示</span> : <span className="flex items-center gap-2"><Play size={16}/> 自动演示</span>}
          </button>
          <button 
            onClick={handleReset}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium flex items-center gap-2 transition-all"
          >
            <RotateCcw size={16}/> 重置
          </button>
        </div>
      </header>

      {/* 主体内容区域 */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* 左侧：流程步骤条 */}
        <div className="lg:col-span-3 bg-slate-800/50 rounded-xl p-4 border border-slate-700 h-fit">
          <h3 className="text-slate-300 font-semibold mb-4 flex items-center gap-2">
            <Activity size={18} /> 动态推演流程
          </h3>
          <div className="space-y-0 relative">
            {/* 连接线 */}
            <div className="absolute left-3.5 top-2 bottom-4 w-0.5 bg-slate-700 z-0"></div>
            
            {steps.map((step, index) => (
              <div 
                key={step.id} 
                className={`relative z-10 pl-10 py-3 cursor-pointer transition-all ${index === currentStep ? 'opacity-100' : 'opacity-40'}`}
                onClick={() => setCurrentStep(index)}
              >
                <div className={`absolute left-1 top-4 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  index <= currentStep 
                    ? (step.riskLevel.includes('P0') ? 'bg-red-600 border-red-600' : 'bg-blue-500 border-blue-500')
                    : 'bg-slate-800 border-slate-600'
                }`}>
                  {index < currentStep && <CheckCircle size={12} className="text-white" />}
                  {index === currentStep && <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>}
                </div>
                <div className={`text-xs font-mono mb-0.5 ${step.riskLevel.includes('P0') ? 'text-red-400' : 'text-blue-400'}`}>{step.phase}</div>
                <div className="font-semibold text-sm">{step.title}</div>
                <div className="text-xs text-slate-500">{step.time}</div>
              </div>
            ))}
          </div>
        </div>

        {/* 中间：数字孪生/态势感知仪表盘 */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          
          {/* 状态总览卡片 */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 relative overflow-hidden">
            {/* 动态风险标签 */}
            {currentStep > 0 && (
              <div className={`absolute top-0 right-0 p-2 text-xs font-bold px-3 rounded-bl-xl border-b border-l animate-pulse ${
                currentStep >= 4 
                  ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                  : 'bg-amber-500/20 text-amber-400 border-amber-500/30'
              }`}>
                {currentData.riskLevel} 响应中
              </div>
            )}

            <div className="flex justify-between items-end mb-6">
              <div>
                 <div className="text-slate-400 text-sm mb-1 flex items-center gap-1"><MapPin size={14}/> 位置</div>
                 <div className="text-xl font-bold flex items-center gap-2">
                   B2-E-01 配电房
                   {currentStep >= 4 && <Siren className="text-red-500 animate-bounce" size={20}/>}
                 </div>
                 <div className="text-slate-500 text-xs">高压配电柜区 / 无人值守</div>
              </div>
              <div className="text-right">
                 <div className="text-slate-400 text-sm mb-1"><Clock size={14} className="inline mr-1"/>场景时间</div>
                 <div className="text-2xl font-mono text-cyan-400">
                   {currentData.time.split(' ')[0]}
                 </div>
              </div>
            </div>

            {/* 传感器矩阵：带有趋势指示 */}
            <div className="grid grid-cols-2 gap-3">
              {/* 气体传感器 (核心) */}
              <SensorCard 
                icon={<Activity size={20}/>} 
                label="气体检测 (CO)" 
                value={`${sensorValues.co} ppm`}
                status={sensorValues.co > 50 ? "critical" : sensorValues.co > 10 ? "warning" : "normal"}
                subText={sensorValues.co > 0 ? "持续上升 ↑" : "基线正常"}
                highlight={currentStep >= 1}
              />
              
              {/* 热成像 (核心) */}
              <SensorCard 
                icon={<Thermometer size={20}/>} 
                label="热成像 (Hotspot)" 
                value={`${sensorValues.temp}°C`}
                status={sensorValues.temp > 70 ? "critical" : sensorValues.temp > 40 ? "warning" : "normal"}
                subText={sensorValues.temp > 40 ? "局部温度梯度大" : "温度均匀"}
                highlight={currentStep >= 1}
              />

              {/* 视觉 (辅助) */}
              <SensorCard 
                icon={<Eye size={20}/>} 
                label="视觉复核" 
                value={currentStep >= 1 ? "灰白稀薄烟雾" : "画面清晰"}
                status={currentStep >= 1 ? "warning" : "normal"}
                subText={currentStep >= 1 ? "无明火" : "---"}
                highlight={false}
              />

              {/* 烟感 (滞后) */}
              <SensorCard 
                icon={<Wind size={20}/>} 
                label="传统烟感" 
                value={`${sensorValues.smoke}%/m`} 
                status={sensorValues.smoke > 3.0 ? "alert" : "normal"}
                subText={sensorValues.smoke < 3.0 ? "未达报警阈值" : "报警触发"}
                highlight={false}
              />
            </div>
          </div>

          {/* 研判结论区 */}
          <div className={`bg-slate-800 rounded-xl p-6 border transition-all duration-500 ${
            currentStep >= 4 
              ? 'border-red-500/50 shadow-[0_0_30px_rgba(239,68,68,0.2)]' 
              : currentStep === 3 
                ? 'border-amber-500/50'
                : 'border-slate-700 opacity-60'
          }`}>
             <h4 className="text-sm uppercase tracking-wider text-slate-400 mb-4 font-bold flex items-center gap-2">
               <Server size={16}/> 动态决策中心 (L3/L4)
             </h4>
             <div className="flex items-center gap-6">
               <div className="relative w-24 h-24 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                    <path className="text-slate-700" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="4" />
                    {/* 动态进度条颜色 */}
                    <path className={`transition-all duration-1000 ease-out ${currentStep >= 4 ? "text-red-500" : "text-amber-500"}`}
                          strokeDasharray={currentStep >= 3 ? "88, 100" : "0, 100"} 
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                          fill="none" stroke="currentColor" strokeWidth="4" />
                  </svg>
                  <div className="absolute text-center">
                    <div className="text-xl font-bold text-white">{currentStep >= 3 ? "88%" : "--"}</div>
                    <div className="text-[10px] text-slate-400">真火置信度</div>
                  </div>
               </div>
               
               <div className="flex-1 space-y-2">
                 <div className="flex justify-between border-b border-slate-700 pb-2">
                   <span className="text-slate-400">研判结果</span>
                   <span className={`font-bold ${currentStep >= 4 ? "text-red-500 animate-pulse" : currentStep === 3 ? "text-amber-400" : "text-slate-500"}`}>
                     {currentStep >= 4 ? "P0级 紧急真火 (Escalated)" : currentStep === 3 ? "P1级 阴燃隐患" : "分析中..."}
                   </span>
                 </div>
                 <div className="flex justify-between border-b border-slate-700 pb-2">
                   <span className="text-slate-400">响应策略</span>
                   <span className="font-bold text-white text-sm">
                     {currentStep >= 4 ? "全楼联动 + 自动阻断" : currentStep === 3 ? "局部报警 + 人工核实" : "--"}
                   </span>
                 </div>
                 <div className="flex justify-between items-center">
                   <span className="text-slate-400">关键动作</span>
                   <span className={`text-xs px-2 py-0.5 rounded ${currentStep >= 4 ? "bg-red-900 text-red-200" : "text-slate-500"}`}>
                     {currentStep >= 4 ? "切断电源 | 启动排烟" : currentStep === 3 ? "派单保安 | 携带灭火器" : "--"}
                   </span>
                 </div>
               </div>
             </div>
          </div>

        </div>

        {/* 右侧：系统日志/思维链 */}
        <div className="lg:col-span-3 bg-black rounded-xl border border-slate-700 p-4 font-mono text-xs flex flex-col h-[500px] lg:h-auto overflow-hidden shadow-inner">
           <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800">
             <span className="text-green-500 font-bold flex items-center gap-2"><FileText size={14}/> System Log</span>
             <span className="text-slate-500 animate-pulse">● Live</span>
           </div>
           
           <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2">
             {steps.slice(0, currentStep + 1).map((step) => (
               <div key={step.id} className="animate-fade-in-down">
                 <div className="text-blue-500 mb-1 opacity-70">[{step.time.split(' ')[0]}] {step.phase}</div>
                 {step.systemLog.map((log, i) => (
                   <div key={i} className={`pl-2 border-l-2 mb-1 leading-relaxed ${
                     log.includes('CRITICAL') || log.includes('FULL BUILDING') ? 'border-red-600 text-red-500 font-bold bg-red-900/10' :
                     log.includes('NOTICE') ? 'border-amber-500 text-amber-300' :
                     log.includes('Action') || log.includes('Decision') ? 'border-green-500 text-green-400' :
                     log.includes('Prediction') ? 'border-purple-500 text-purple-300' :
                     'border-slate-700 text-slate-300'
                   }`}>
                     {log}
                   </div>
                 ))}
               </div>
             ))}
             {/* 占位符确保自动滚动到底部 */}
             <div className="h-4"></div>
           </div>
        </div>

      </div>

      {/* 底部控制栏 */}
      <div className="w-full max-w-6xl mt-6 flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700">
         <div className="text-sm text-slate-400">
           当前阶段: <span className="text-white font-medium">{currentData.description}</span>
         </div>
         <button 
           onClick={handleNext}
           disabled={currentStep === steps.length - 1 || isAutoPlay}
           className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white px-6 py-2.5 rounded-lg font-bold shadow-lg shadow-blue-900/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
         >
           下一步 <ChevronRight size={18} />
         </button>
      </div>

    </div>
  );
}

// --- 子组件：传感器卡片 ---
function SensorCard({ icon, label, value, status, subText, highlight = false }: any) {
  const getStatusColor = (s: string) => {
    switch(s) {
      case 'critical': return 'text-red-100 bg-red-600 border-red-500 shadow-red-500/50 shadow-md animate-pulse';
      case 'alert': return 'text-red-400 bg-red-900/20 border-red-500/50';
      case 'warning': return 'text-amber-400 bg-amber-900/20 border-amber-500/50';
      default: return 'text-slate-300 bg-slate-700/30 border-slate-600/30';
    }
  };

  return (
    <div className={`p-3 rounded-lg border transition-all duration-300 ${getStatusColor(status)} ${highlight ? 'ring-2 ring-amber-400 scale-105 shadow-lg z-10' : ''}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="opacity-70">{icon}</span>
        <span className="text-xs font-bold uppercase opacity-80">{label}</span>
      </div>
      <div className="text-lg font-bold truncate flex items-center gap-1">
        {value}
        {status === 'critical' && <TrendingUp size={16} />}
      </div>
      <div className="text-[10px] opacity-80 mt-1">{subText}</div>
    </div>
  );
}