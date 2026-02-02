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
  Ban,
  Flame,
  Droplets,
  Disc
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
    showFire: boolean;
    fireIntensity: 'low' | 'high' | 'none';
    gasValveStatus: 'open' | 'closing' | 'closed';
    sprinklerStatus: 'ready' | 'disabled' | 'active'; // 油锅火灾需禁用喷淋
    showGuard: boolean;
    sensorAlert: boolean;
  };
};

// --- 核心组件 ---
export default function FireAuthDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlay, setIsAutoPlay] = useState(false);

  // 模拟场景数据：后厨油锅起火（懂灭火策略的系统）
  const steps: Step[] = [
    {
      id: 0,
      phase: "L0 常态监控",
      title: "晚餐高峰期",
      time: "19:00:00 (T=-10s)",
      description: "Sup-01 正在监控 1F 后厨区域。当前为晚餐高峰，炉灶全开，环境温度较高 (45°C)，燃气阀门开启中。",
      activeSensors: [],
      riskLevel: "Normal",
      systemLog: [
        "> System Status: ONLINE (Peak Dining Mode)",
        "> Location: 1F-K (Main Kitchen)",
        "> Baseline: High Temp Allowed (<60°C)",
        "> Gas Valve: OPEN (Flow: Normal)"
      ],
      mapState: { showFire: false, fireIntensity: 'none', gasValveStatus: 'open', sprinklerStatus: 'ready', showGuard: false, sensorAlert: false }
    },
    {
      id: 1,
      phase: "L1 异常感知",
      title: "瞬间起火 (T=0s)",
      time: "19:00:10 (T+0s)",
      description: "油温过高引发自燃。热成像检测到温度瞬间突破 200°C，视觉AI捕捉到明亮橙色火焰。CO浓度飙升。",
      activeSensors: ['thermal', 'visual', 'gas'],
      riskLevel: "Critical",
      systemLog: [
        "! ALERT: Rapid Temp Rise (>30°C/min)",
        "> Thermal: Max Temp 220°C (CRITICAL)",
        "> Visual: Orange Flame Detected (Conf: 99%)",
        "> Gas: CO > 150ppm (Combustion Confirmed)"
      ],
      mapState: { showFire: true, fireIntensity: 'high', gasValveStatus: 'open', sprinklerStatus: 'ready', showGuard: false, sensorAlert: true }
    },
    {
      id: 2,
      phase: "L2 策略研判",
      title: "场景识别与策略生成 (T+0.3s)",
      time: "19:00:10 (T+300ms)",
      description: "基于‘明火+超高温+后厨’特征，判定为‘油锅火灾’。触发生命线规则：禁止水喷淋，防止爆燃。",
      activeSensors: ['decision', 'context'],
      riskLevel: "P0 (Emergency)",
      systemLog: [
        "> Pattern Match: 'Oil Pan Fire' (Conf: 99%)",
        "> SAFETY LOCK: Class F Fire Detected",
        "> RULE TRIGGER: DISABLE SPRINKLER (Water Hazard)",
        "> Strategy: Cut Fuel + Suffocate Fire"
      ],
      mapState: { showFire: true, fireIntensity: 'high', gasValveStatus: 'open', sprinklerStatus: 'disabled', showGuard: false, sensorAlert: true }
    },
    {
      id: 3,
      phase: "L3 自动执行",
      title: "切断燃料 (T+0.5s)",
      time: "19:00:10 (T+500ms)",
      description: "系统自动指令燃气控制系统关闭主阀门，切断火源燃料。同时锁定喷淋系统，防止误动作。",
      activeSensors: ['execution'],
      riskLevel: "P0 (Handling)",
      systemLog: [
        "> Action: CLOSE GAS VALVE (Priority: P0)",
        "> Action: LOCK Sprinkler System [LOCKED]",
        "> Action: Local Alarm (Kitchen Only)",
        "> Status: Fuel Cut-off Successful"
      ],
      mapState: { showFire: true, fireIntensity: 'high', gasValveStatus: 'closed', sprinklerStatus: 'disabled', showGuard: false, sensorAlert: true }
    },
    {
      id: 4,
      phase: "L4 精准派单",
      title: "专业处置 (T+1min)",
      time: "19:01:10 (T+1min)",
      description: "保安接到特殊指令：‘带灭火毯，禁止用水’。保安迅速到场覆盖灭火，避免了重大损失。",
      activeSensors: ['execution'],
      riskLevel: "Controlling",
      systemLog: [
        "> Dispatch: Guard 'Wang' (Nearest)",
        "> Instruction: 'USE FIRE BLANKET ONLY! NO WATER!'",
        "> Status: Guard Arrived (50s)",
        "> Feedback: 'Fire covered, under control'"
      ],
      mapState: { showFire: true, fireIntensity: 'low', gasValveStatus: 'closed', sprinklerStatus: 'disabled', showGuard: true, sensorAlert: false }
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
    if (stepId === 0) return { temp: 45, co: 5, visual: '无明火', ror: '平稳' };
    if (stepId >= 1) return { temp: 230, co: 180, visual: '橙色火焰', ror: '>30°C/min' };
    return { temp: 45, co: 5, visual: '无明火', ror: '平稳' };
  };

  const sensorValues = getSensorData(currentStep);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-8 flex flex-col items-center">
      
      {/* 顶部标题栏 */}
      <header className="w-full max-w-6xl mb-6 flex flex-col md:flex-row justify-between items-center border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text text-transparent flex items-center gap-3">
            <Zap className="w-8 h-8 text-orange-500" />
            Sup-01 数智主管
          </h1>
          <p className="text-slate-400 mt-1">场景：后厨油锅起火（智能灭火策略）演示</p>
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
            <Activity size={18} /> 响应流程
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

        {/* 中间：后厨平面态势图 + 仪表盘 */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          
          {/* 平面俯视态势图容器 */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 relative overflow-hidden h-[360px] flex items-center justify-center bg-grid-pattern">
            <div className="absolute top-4 left-4 z-20">
              <div className="text-xs text-slate-400 flex items-center gap-1"><MapPin size={12}/> 1F 后厨平面图</div>
              <div className="text-lg font-bold text-white">热厨加工区</div>
            </div>

            {/* SVG 后厨平面图 */}
            <KitchenMap state={currentData.mapState} />

            {/* 策略浮窗 */}
            {currentStep >= 2 && (
              <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
                 <div className={`px-3 py-2 rounded border text-xs font-bold flex items-center gap-2 shadow-lg transition-all ${currentData.mapState.sprinklerStatus === 'disabled' ? 'bg-red-900/90 border-red-500 text-white' : 'bg-slate-800 border-slate-600 text-slate-400'}`}>
                   {currentData.mapState.sprinklerStatus === 'disabled' ? <Ban size={14} className="text-red-400"/> : <Droplets size={14}/>}
                   喷淋系统: {currentData.mapState.sprinklerStatus === 'disabled' ? '已强制禁用' : '就绪'}
                 </div>
                 <div className={`px-3 py-2 rounded border text-xs font-bold flex items-center gap-2 shadow-lg transition-all ${currentData.mapState.gasValveStatus === 'closed' ? 'bg-green-900/90 border-green-500 text-white' : 'bg-slate-800 border-slate-600 text-slate-400'}`}>
                   <Disc size={14} className={currentData.mapState.gasValveStatus === 'closed' ? "text-green-400" : ""}/>
                   燃气阀门: {currentData.mapState.gasValveStatus === 'closed' ? '已切断' : '开启'}
                 </div>
              </div>
            )}

            {/* 决策结果悬浮层 */}
            {currentStep >= 2 && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-900/90 backdrop-blur border border-red-500/50 rounded-lg p-3 flex items-center gap-4 shadow-xl z-20 animate-fade-in-up w-3/4 justify-between">
                 <div className="flex items-center gap-3">
                   <div className="bg-red-500/20 p-2 rounded-full text-red-400 animate-pulse">
                     <Flame size={20} />
                   </div>
                   <div>
                     <div className="text-red-400 font-bold text-sm">判定：油锅火灾 (P0)</div>
                     <div className="text-xs text-slate-400">策略: 禁水 | 断气 | 窒息灭火</div>
                   </div>
                 </div>
                 <div className="text-right border-l border-slate-700 pl-4">
                   <div className="text-[10px] text-slate-500 uppercase">Alert</div>
                   <div className="text-xs font-mono text-white">Local Only</div>
                 </div>
              </div>
            )}
          </div>

          {/* 传感器数据卡片 */}
          <div className="grid grid-cols-4 gap-2">
            <SensorCard 
              icon={<Thermometer size={16}/>} label="热成像" 
              value={`${sensorValues.temp}°C`} 
              status={sensorValues.temp > 200 ? "critical" : "normal"}
              alertText="极高"
            />
            <SensorCard 
              icon={<TrendingUp size={16}/>} label="温升速率" 
              value={sensorValues.ror} 
              status={sensorValues.temp > 200 ? "critical" : "normal"}
              alertText="爆燃"
            />
            <SensorCard 
              icon={<Eye size={16}/>} label="视觉" 
              value={sensorValues.visual} 
              status={sensorValues.temp > 200 ? "critical" : "normal"}
              alertText="明火"
            />
            <SensorCard 
              icon={<Activity size={16}/>} label="CO浓度" 
              value={`${sensorValues.co}ppm`} 
              status={sensorValues.co > 100 ? "critical" : "normal"}
              alertText="有毒"
            />
          </div>
        </div>

        {/* 右侧：系统日志 */}
        <div className="lg:col-span-3 bg-black rounded-xl border border-slate-700 p-4 font-mono text-xs flex flex-col h-[480px] lg:h-auto overflow-hidden shadow-inner">
           <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800">
             <span className="text-red-500 font-bold flex items-center gap-2"><FileText size={14}/> 决策核心日志</span>
             <span className="text-slate-500 animate-pulse">● Live</span>
           </div>
           
           <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2">
             {steps.slice(0, currentStep + 1).map((step) => (
               <div key={step.id} className="animate-fade-in-down">
                 <div className="text-blue-500 mb-1 opacity-70">[{step.time.split(' ')[0]}] {step.phase}</div>
                 {step.systemLog.map((log, i) => (
                   <div key={i} className={`pl-2 border-l-2 mb-1 leading-relaxed ${
                     log.includes('CRITICAL') || log.includes('ALERT') ? 'border-red-500 text-red-400 font-bold' :
                     log.includes('DISABLE') || log.includes('LOCK') ? 'border-amber-500 text-amber-400 font-bold' : // 策略性动作高亮
                     log.includes('Action') ? 'border-green-500 text-green-400' :
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
           className="flex items-center gap-2 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white px-6 py-2.5 rounded-lg font-bold shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
         >
           下一步 <ChevronRight size={18} />
         </button>
      </div>

    </div>
  );
}

// --- 组件：后厨平面地图 (SVG) ---
function KitchenMap({ state }: { state: any }) {
  return (
    <div className="w-full h-full relative">
      <svg viewBox="0 0 400 300" className="w-full h-full">
        {/* 背景 */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
          </pattern>
          <radialGradient id="fireGradient">
            <stop offset="0%" stopColor="#FFFF00" />
            <stop offset="50%" stopColor="#FF8C00" />
            <stop offset="100%" stopColor="#FF0000" stopOpacity="0" />
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* 墙体与设施 */}
        <g stroke="#475569" strokeWidth="3" fill="none">
          {/* 厨房轮廓 */}
          <rect x="50" y="50" width="300" height="200" />
          {/* 灶台区 */}
          <rect x="50" y="100" width="80" height="100" fill="#334155" />
          <text x="90" y="150" fontSize="10" fill="#94A3B8" textAnchor="middle" transform="rotate(-90 90 150)">热厨区</text>
          {/* 炉灶圆圈 */}
          <circle cx="90" cy="120" r="15" stroke="#94A3B8" strokeWidth="2" />
          <circle cx="90" cy="180" r="15" stroke="#94A3B8" strokeWidth="2" />
        </g>

        {/* 燃气管道系统 */}
        <g>
          <path d="M 50 250 L 90 250 L 90 200" stroke="#10B981" strokeWidth="4" opacity="0.6" />
          <text x="110" y="255" fontSize="10" fill="#10B981">燃气主管道</text>
          
          {/* 燃气阀门动画 */}
          <g transform="translate(70, 250)">
            <circle cx="0" cy="0" r="8" fill="#1E293B" stroke={state.gasValveStatus === 'closed' ? "#10B981" : "#64748B"} strokeWidth="2" />
            <rect x="-2" y="-6" width="4" height="12" fill={state.gasValveStatus === 'closed' ? "#10B981" : "#64748B"} className={`transition-transform duration-1000 ${state.gasValveStatus === 'closed' ? 'rotate-90' : ''}`} />
            {state.gasValveStatus === 'closed' && (
              <text x="-20" y="-15" fontSize="10" fill="#10B981" fontWeight="bold">CLOSED</text>
            )}
          </g>
        </g>

        {/* 喷淋系统 (天花板视角) */}
        <g transform="translate(90, 120)">
           <line x1="0" y1="0" x2="20" y2="0" stroke="#64748B" strokeWidth="1" strokeDasharray="2,2"/>
           <circle cx="0" cy="0" r="6" fill="none" stroke={state.sprinklerStatus === 'disabled' ? "#EF4444" : "#3B82F6"} strokeWidth="2" />
           {state.sprinklerStatus === 'disabled' && (
             <path d="M -4 -4 L 4 4 M 4 -4 L -4 4" stroke="#EF4444" strokeWidth="2" className="animate-pulse" />
           )}
           <text x="10" y="-10" fontSize="8" fill={state.sprinklerStatus === 'disabled' ? "#EF4444" : "#64748B"}>
             {state.sprinklerStatus === 'disabled' ? '喷淋禁用' : '喷淋就绪'}
           </text>
        </g>

        {/* 火焰动画 */}
        {state.showFire && (
          <g transform="translate(90, 120)">
            <path 
              d="M 0 0 Q -10 -20 0 -40 Q 10 -20 0 0" 
              fill="url(#fireGradient)" 
              className={`animate-pulse-fast ${state.fireIntensity === 'low' ? 'scale-50 opacity-50' : 'scale-125'}`}
            />
            {state.fireIntensity === 'high' && (
               <animateTransform attributeName="transform" type="scale" values="1;1.2;1" dur="0.2s" repeatCount="indefinite" />
            )}
          </g>
        )}

        {/* 保安路径 */}
        {state.showGuard && (
          <g>
            <circle cx="340" cy="200" r="5" fill="#3B82F6" className="animate-pulse" />
            <text x="340" y="215" fontSize="8" fill="#3B82F6" textAnchor="middle">保安(带灭火毯)</text>
            <path d="M 340 200 L 250 200 L 150 150 L 110 120" stroke="#3B82F6" strokeWidth="2" strokeDasharray="4,4" fill="none" className="animate-dash" />
          </g>
        )}

      </svg>
    </div>
  );
}

// --- 组件：传感器数据卡片 ---
function SensorCard({ icon, label, value, status, alertText }: any) {
  const getColors = () => {
    if (status === 'critical') return 'bg-red-900/40 border-red-500/80 text-red-100 shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse-slow';
    return 'bg-slate-700/30 border-slate-600/30 text-slate-300';
  };

  return (
    <div className={`p-3 rounded-lg border ${getColors()} flex flex-col justify-between h-24 transition-all duration-300`}>
      <div className="flex justify-between items-start">
        <div className="opacity-70">{icon}</div>
        {status !== 'normal' && (
           <span className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-red-500 text-black">
             {alertText}
           </span>
        )}
      </div>
      <div>
        <div className="text-[10px] uppercase opacity-60 mb-0.5">{label}</div>
        <div className="text-lg font-bold font-mono leading-none truncate">{value}</div>
      </div>
    </div>
  );
}