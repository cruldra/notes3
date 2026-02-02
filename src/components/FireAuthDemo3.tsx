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
  Siren,
  User,
  ClipboardCheck,
  Ban
} from 'lucide-react';

// --- 类型定义 ---
type Step = {
  id: number;
  phase: string;
  title: string;
  time: string;
  description: string;
  activeSensors: string[]; 
  riskLevel: string;
  systemLog: string[];
  mapState: {
    showSmoke: boolean;
    smokeColor: string; // 'white' | 'gray' | 'black'
    showWorkOrder: boolean;
    showGuard: boolean;
    sensorAlert: boolean;
  };
};

// --- 核心组件 ---
export default function FireAuthDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlay, setIsAutoPlay] = useState(false);

  // 模拟场景数据：装修作业干扰（喷漆/电焊）
  const steps: Step[] = [
    {
      id: 0,
      phase: "L0 常态监控",
      title: "区域巡检中",
      time: "15:32:50 (T=-10s)",
      description: "工作日下午，Sup-01 正在监控 3F 商铺区。3F-05 区域处于装修备案期，系统基线已自动调整。",
      activeSensors: [],
      riskLevel: "Normal",
      systemLog: [
        "> System Status: ONLINE (Day Mode)",
        "> Location: 3F-05 (Retail Zone - Under Renovation)",
        "> Baseline: Adjusted for 'Construction Mode'",
        "> Monitoring: Area Sensors Active"
      ],
      mapState: { showSmoke: false, smokeColor: 'white', showWorkOrder: false, showGuard: false, sensorAlert: false }
    },
    {
      id: 1,
      phase: "L1 异常感知",
      title: "多模态信号触发 (T=0s)",
      time: "15:33:00 (T+0s)",
      description: "烟感探测器触发报警，气体传感器显示 VOC (挥发性有机物) 瞬间飙升至 1200ppb，但 CO (燃烧产物) 浓度正常。",
      activeSensors: ['smoke', 'gas'],
      riskLevel: "Detecting",
      systemLog: [
        "! ALERT: Smoke Detector Triggered (3F-05)",
        "> Gas Analysis: VOC > 1200ppb (CRITICAL), CO Normal",
        "> Thermal: Temp 24°C (Normal)",
        "> Action: Trigger Multi-modal Verification"
      ],
      mapState: { showSmoke: true, smokeColor: 'white', showWorkOrder: false, showGuard: false, sensorAlert: true }
    },
    {
      id: 2,
      phase: "L2 上下文关联",
      title: "工单系统 RAG 检索 (T+0.3s)",
      time: "15:33:00 (T+300ms)",
      description: "系统自动查询工单数据库，发现该区域在当前时段有“墙面喷漆”作业备案。视觉复核确认为“白色雾气”。",
      activeSensors: ['context', 'visual'],
      riskLevel: "Analyzing",
      systemLog: [
        "> Visual Check: White Fog Detected (No Flame)",
        "> Query Work Order DB: Location=3F-05, Time=NOW",
        "> RESULT: Found WO-20260201-0033",
        "> Type: 'Wall Painting' (Source of VOC)"
      ],
      mapState: { showSmoke: true, smokeColor: 'white', showWorkOrder: true, showGuard: false, sensorAlert: true }
    },
    {
      id: 3,
      phase: "L3 因果推理",
      title: "合法作业误报判定 (T+0.5s)",
      time: "15:33:00 (T+500ms)",
      description: "推理链：白色雾气 + 高VOC + 无高温 + 有喷漆工单 = 合法作业误报。判定风险等级 P3 (低风险)。",
      activeSensors: ['decision'],
      riskLevel: "P3 (Low Risk)",
      systemLog: [
        "> Evidence Chain: High VOC + White Fog + No Heat",
        "> Context Match: Painting Job Active",
        "> Inference: False Alarm (Confidence: 98%)",
        "> Decision: SUPPRESS ALARM (Avoid Panic)"
      ],
      mapState: { showSmoke: true, smokeColor: 'green', showWorkOrder: true, showGuard: false, sensorAlert: false }
    },
    {
      id: 4,
      phase: "L4 执行与闭环",
      title: "派单核实 (T+5min)",
      time: "15:38:00 (T+5min)",
      description: "虽然抑制了声光报警，系统仍生成核查工单。保安到场确认是喷漆作业，现场安全。系统自动记录案例。",
      activeSensors: ['execution'],
      riskLevel: "Resolved",
      systemLog: [
        "> Action: Suppress Public Alarm (Silent Mode)",
        "> Dispatch: Guard to 3F-05 (Check in 5 min)",
        "> Feedback: 'Painting in progress, safe'",
        "> System: Case Closed. Logged as 'Painting False Alarm'."
      ],
      mapState: { showSmoke: true, smokeColor: 'green', showWorkOrder: true, showGuard: true, sensorAlert: false }
    }
  ];

  // 自动播放逻辑
  useEffect(() => {
    let interval: any;
    if (isAutoPlay && currentStep < steps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => prev + 1);
      }, 4000); 
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

  // 模拟传感器数值
  const getSensorData = (stepId: number) => {
    if (stepId === 0) return { voc: 50, co: 0, temp: 24, smoke: 0 }; // 正常
    // 异常触发：VOC极高（油漆味），CO正常（无燃烧），温度正常
    if (stepId >= 1) return { voc: 1250, co: 2, temp: 24.5, smoke: 4.5 }; 
    return { voc: 50, co: 0, temp: 24, smoke: 0 };
  };

  const sensorValues = getSensorData(currentStep);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-8 flex flex-col items-center">
      
      {/* 顶部标题栏 */}
      <header className="w-full max-w-6xl mb-6 flex flex-col md:flex-row justify-between items-center border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent flex items-center gap-3">
            <BrainCircuit className="w-8 h-8 text-indigo-400" />
            Sup-01 数智主管
          </h1>
          <p className="text-slate-400 mt-1">场景：装修作业干扰（喷漆误报）智能研判演示</p>
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
            <Activity size={18} /> 研判流程
          </h3>
          <div className="space-y-0 relative">
            <div className="absolute left-3.5 top-2 bottom-4 w-0.5 bg-slate-700 z-0"></div>
            {steps.map((step, index) => (
              <div 
                key={step.id} 
                className={`relative z-10 pl-10 py-3 cursor-pointer transition-all ${index === currentStep ? 'opacity-100' : 'opacity-40'}`}
                onClick={() => setCurrentStep(index)}
              >
                <div className={`absolute left-1 top-4 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  index <= currentStep 
                    ? 'bg-blue-500 border-blue-500'
                    : 'bg-slate-800 border-slate-600'
                }`}>
                  {index < currentStep && <CheckCircle size={12} className="text-white" />}
                  {index === currentStep && <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>}
                </div>
                <div className="text-xs text-blue-400 font-mono mb-0.5">{step.phase}</div>
                <div className="font-semibold text-sm">{step.title}</div>
                <div className="text-xs text-slate-500">{step.time}</div>
              </div>
            ))}
          </div>
        </div>

        {/* 中间：平面俯视态势图 + 仪表盘 */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          
          {/* 平面俯视态势图容器 */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 relative overflow-hidden h-[360px] flex items-center justify-center bg-grid-pattern">
            <div className="absolute top-4 left-4 z-20">
              <div className="text-xs text-slate-400 flex items-center gap-1"><MapPin size={12}/> 3F 平面图</div>
              <div className="text-lg font-bold text-white">3F-05 装修区域</div>
            </div>

            {/* SVG 平面图 */}
            <FloorPlanMap state={currentData.mapState} />

            {/* 决策结果悬浮层 */}
            {currentStep >= 3 && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-900/90 backdrop-blur border border-green-500/50 rounded-lg p-3 flex items-center gap-4 shadow-xl z-20 animate-fade-in-up w-3/4 justify-between">
                 <div className="flex items-center gap-3">
                   <div className="bg-green-500/20 p-2 rounded-full text-green-400">
                     <ShieldAlert size={20} />
                   </div>
                   <div>
                     <div className="text-green-400 font-bold text-sm">判定：合法作业误报 (P3)</div>
                     <div className="text-xs text-slate-400">置信度: 98% | 已抑制报警</div>
                   </div>
                 </div>
                 <div className="text-right border-l border-slate-700 pl-4">
                   <div className="text-[10px] text-slate-500 uppercase">Action</div>
                   <div className="text-xs font-mono text-white">Silent Mode</div>
                 </div>
              </div>
            )}
          </div>

          {/* 传感器数据卡片 */}
          <div className="grid grid-cols-4 gap-2">
            <SensorCard 
              icon={<Wind size={16}/>} label="烟感" 
              value={`${sensorValues.smoke}%/m`} 
              status={sensorValues.smoke > 3 ? "alert" : "normal"}
              alertText="触发"
            />
            <SensorCard 
              icon={<Activity size={16}/>} label="VOC (气味)" 
              value={`${sensorValues.voc}ppb`} 
              status={sensorValues.voc > 1000 ? "critical" : "normal"}
              alertText="极高"
            />
            <SensorCard 
              icon={<Activity size={16}/>} label="CO (燃烧)" 
              value={`${sensorValues.co}ppm`} 
              status="normal"
              alertText="正常"
            />
            <SensorCard 
              icon={<Thermometer size={16}/>} label="温度" 
              value={`${sensorValues.temp}°C`} 
              status="normal"
              alertText="正常"
            />
          </div>
        </div>

        {/* 右侧：系统日志 */}
        <div className="lg:col-span-3 bg-black rounded-xl border border-slate-700 p-4 font-mono text-xs flex flex-col h-[480px] lg:h-auto overflow-hidden shadow-inner">
           <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800">
             <span className="text-green-500 font-bold flex items-center gap-2"><FileText size={14}/> 智能研判日志</span>
             <span className="text-slate-500 animate-pulse">● Live</span>
           </div>
           
           <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2">
             {steps.slice(0, currentStep + 1).map((step) => (
               <div key={step.id} className="animate-fade-in-down">
                 <div className="text-blue-500 mb-1 opacity-70">[{step.time.split(' ')[0]}] {step.phase}</div>
                 {step.systemLog.map((log, i) => (
                   <div key={i} className={`pl-2 border-l-2 mb-1 leading-relaxed ${
                     log.includes('ALERT') ? 'border-red-500 text-red-400 font-bold' :
                     log.includes('Decision') || log.includes('RESULT') ? 'border-green-500 text-green-400 font-bold' :
                     log.includes('Query') || log.includes('Context') ? 'border-indigo-500 text-indigo-300' :
                     'border-slate-700 text-slate-300'
                   }`}>
                     {log}
                   </div>
                 ))}
               </div>
             ))}
             <div className="h-4"></div>
           </div>
        </div>

      </div>

      {/* 底部信息栏 */}
      <div className="w-full max-w-6xl mt-6 flex justify-between items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700">
         <div className="text-sm text-slate-400">
           当前状态: <span className="text-white font-medium">{currentData.description}</span>
         </div>
         <button 
           onClick={handleNext}
           disabled={currentStep === steps.length - 1 || isAutoPlay}
           className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white px-6 py-2.5 rounded-lg font-bold shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
         >
           下一步 <ChevronRight size={18} />
         </button>
      </div>

    </div>
  );
}

// --- 组件：平面俯视地图 (SVG) ---
function FloorPlanMap({ state }: { state: any }) {
  return (
    <div className="w-full h-full relative">
      <svg viewBox="0 0 400 300" className="w-full h-full">
        {/* 背景网格 */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* 墙体结构 */}
        <g stroke="#475569" strokeWidth="4" fill="none">
          {/* 主走廊 */}
          <path d="M 50 100 L 350 100" />
          <path d="M 50 200 L 350 200" />
          {/* 房间分隔 */}
          <path d="M 150 50 L 150 100" />
          <path d="M 250 50 L 250 100" />
          <path d="M 150 200 L 150 250" />
          <path d="M 250 200 L 250 250" />
          {/* 3F-05 房间轮廓 (重点区域) */}
          <rect x="160" y="30" width="80" height="60" stroke={state.sensorAlert ? "#EF4444" : "#64748B"} strokeWidth={state.sensorAlert ? "3" : "2"} fill="#1E293B" className="transition-all duration-500"/>
        </g>

        {/* 房间标签 */}
        <text x="100" y="80" fontSize="10" fill="#64748B" textAnchor="middle">3F-04</text>
        <text x="200" y="80" fontSize="12" fill={state.sensorAlert ? "#EF4444" : "#94A3B8"} fontWeight="bold" textAnchor="middle">3F-05 (装修)</text>
        <text x="300" y="80" fontSize="10" fill="#64748B" textAnchor="middle">3F-06</text>
        <text x="200" y="150" fontSize="10" fill="#475569" textAnchor="middle">主走廊</text>

        {/* 动态烟雾/雾气层 */}
        {state.showSmoke && (
          <g className="animate-pulse-slow mix-blend-screen">
             <circle cx="200" cy="60" r="25" fill={state.smokeColor === 'white' ? "rgba(255,255,255,0.4)" : "rgba(34,197,94,0.3)"} filter="blur(8px)" />
             <circle cx="210" cy="50" r="20" fill={state.smokeColor === 'white' ? "rgba(255,255,255,0.3)" : "rgba(34,197,94,0.2)"} filter="blur(6px)" />
             <circle cx="190" cy="55" r="15" fill={state.smokeColor === 'white' ? "rgba(200,200,255,0.3)" : "rgba(34,197,94,0.2)"} filter="blur(5px)" />
          </g>
        )}

        {/* 传感器图标位置 */}
        <g>
           {/* 烟感 */}
           <circle cx="180" cy="45" r="4" fill={state.sensorAlert ? "#EF4444" : "#3B82F6"} className="transition-colors duration-300"/>
           <circle cx="180" cy="45" r="8" fill="none" stroke={state.sensorAlert ? "#EF4444" : "#3B82F6"} opacity="0.5" className={state.sensorAlert ? "animate-ping" : ""} />
           {/* 气体 */}
           <circle cx="220" cy="45" r="4" fill={state.sensorAlert ? "#EF4444" : "#10B981"} className="transition-colors duration-300"/>
           {/* 摄像头 */}
           <path d="M 200 30 L 195 25 L 205 25 Z" fill="#F59E0B" />
        </g>
        
        {/* 装修工人图标 */}
        <g transform="translate(195, 60)">
           <circle cx="0" cy="0" r="3" fill="#F59E0B" />
           <rect x="-2" y="3" width="4" height="6" fill="#F59E0B" />
        </g>

        {/* 上下文工单连接线与卡片 */}
        {state.showWorkOrder && (
          <g>
            {/* 连接线 */}
            <path d="M 240 60 L 280 60 L 300 90" stroke="#6366F1" strokeWidth="1" strokeDasharray="4,4" className="animate-draw" />
            
            {/* 工单卡片 */}
            <foreignObject x="280" y="90" width="110" height="60">
              <div className="bg-indigo-900/90 border border-indigo-500 rounded p-2 text-[8px] text-indigo-100 shadow-lg animate-fade-in-up">
                <div className="font-bold flex items-center gap-1 mb-1"><ClipboardCheck size={8}/> 工单匹配</div>
                <div>ID: WO-0033</div>
                <div>内容: 墙面喷漆</div>
                <div className="text-green-300">状态: 备案中</div>
              </div>
            </foreignObject>
          </g>
        )}

        {/* 保安路径 */}
        {state.showGuard && (
          <g>
            <circle cx="340" cy="150" r="5" fill="#3B82F6" className="animate-pulse" />
            <text x="340" y="165" fontSize="8" fill="#3B82F6" textAnchor="middle">保安</text>
            <path d="M 340 150 L 280 150 L 240 100 L 200 80" stroke="#3B82F6" strokeWidth="2" strokeDasharray="4,4" fill="none" className="animate-dash" />
          </g>
        )}

      </svg>
      
      {/* 图例 */}
      <div className="absolute bottom-2 right-2 flex flex-col gap-1 text-[10px] text-slate-500 bg-slate-900/50 p-2 rounded border border-slate-800">
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-red-500"></div> 报警传感器</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-blue-500"></div> 正常传感器</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-indigo-500"></div> 业务数据流</div>
      </div>
    </div>
  );
}

// --- 组件：传感器数据卡片 ---
function SensorCard({ icon, label, value, status, alertText }: any) {
  const getColors = () => {
    if (status === 'critical') return 'bg-red-900/30 border-red-500/50 text-red-100';
    if (status === 'alert') return 'bg-amber-900/30 border-amber-500/50 text-amber-100';
    return 'bg-slate-700/30 border-slate-600/30 text-slate-300';
  };

  return (
    <div className={`p-3 rounded-lg border ${getColors()} flex flex-col justify-between h-24 transition-all duration-300`}>
      <div className="flex justify-between items-start">
        <div className="opacity-70">{icon}</div>
        {status !== 'normal' && (
           <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${status==='critical'?'bg-red-500':'bg-amber-500'} text-black`}>
             {alertText}
           </span>
        )}
      </div>
      <div>
        <div className="text-[10px] uppercase opacity-60 mb-0.5">{label}</div>
        <div className="text-lg font-bold font-mono leading-none">{value}</div>
      </div>
    </div>
  );
}