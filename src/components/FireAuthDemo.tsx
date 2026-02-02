import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Eye, 
  Thermometer, 
  Wind, 
  Activity, 
  CheckCircle, 
  XCircle, 
  ShieldAlert, 
  Server, 
  FileText, 
  Clock, 
  MapPin,
  Play,
  RotateCcw,
  ChevronRight,
  BrainCircuit
} from 'lucide-react';

// --- 类型定义 ---
type Step = {
  id: number;
  phase: string;
  title: string;
  time: string;
  description: string;
  activeSensors: string[];
  systemLog: string[];
};

// --- 核心组件 ---
export default function FireAuthDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlay, setIsAutoPlay] = useState(false);

  // 模拟场景数据：3F餐饮区油烟误报
  const steps: Step[] = [
    {
      id: 0,
      phase: "L0 常态监控",
      title: "系统待机中",
      time: "T = -10s",
      description: "Sup-01 数智主管正在监控全楼 12,000+ 个传感器点位。当前环境一切正常。",
      activeSensors: [],
      systemLog: [
        "> System Status: ONLINE",
        "> Heartbeat: Normal",
        "> Monitoring: 3F-C区-15号点位 (火锅店后厨)"
      ]
    },
    {
      id: 1,
      phase: "L1 异常触发",
      title: "烟感报警触发",
      time: "T = 0s",
      description: "3F-C区 15号点位烟感探测器数值突升，达到报警阈值。传统系统此时会直接触发全楼警报。",
      activeSensors: ['smoke'],
      systemLog: [
        "! ALERT: Smoke Sensor Triggered",
        "> Value: 3.2%/m (Threshold: 3.0%)",
        "> Location: 3F-C-15",
        "> Action: Waking up Sup-01 Agent..."
      ]
    },
    {
      id: 2,
      phase: "L1 多模态感知",
      title: "并行取证 (0-300ms)",
      time: "T = +0.3s",
      description: "Sup-01 瞬间调取视觉、热成像、气体传感器数据进行复核。发现有烟无火，温度正常，VOC较高。",
      activeSensors: ['smoke', 'visual', 'thermal', 'gas'],
      systemLog: [
        "> Visual Skill: Detected 'White Fog', No Flame (Conf: 0.95)",
        "> Thermal Skill: Max Temp 58°C (Stove), Ambient 26°C",
        "> Gas Skill: VOC High (350ppb), CO Normal (8ppm)",
        "> Data Alignment: Completed"
      ]
    },
    {
      id: 3,
      phase: "L2 理解与推理",
      title: "上下文与因果推理 (300-450ms)",
      time: "T = +0.45s",
      description: "查询工单、历史记录与物理铁律。识别为“晚餐高峰期+高频误报点位”，且不符合燃烧物理规律。",
      activeSensors: ['smoke', 'visual', 'thermal', 'gas', 'context'],
      systemLog: [
        "> Context: Dinner Peak (18:35), No Work Order",
        "> History: 12 false alarms in 30 days (Oil Fume)",
        "> Physics Check: Temp < 60°C AND CO < 30ppm = NO FIRE",
        "> Inference: Pattern Match 'Oil Fume' (96%)"
      ]
    },
    {
      id: 4,
      phase: "L3 智能决策",
      title: "决策生成 (450-500ms)",
      time: "T = +0.5s",
      description: "基于96%的误报置信度和P3低风险等级，决定抑制声光报警，并生成核实任务。",
      activeSensors: ['decision'],
      systemLog: [
        "> Risk Assessment: P3 (Low Risk)",
        "> Confidence: 96% (False Alarm)",
        "> DECISION: SUPPRESS ALARM (3 min)",
        "> Action: Generate Verification Task"
      ]
    },
    {
      id: 5,
      phase: "L4 执行与反馈",
      title: "任务闭环 (T+8min)",
      time: "T = +8min",
      description: "保安接到精准派单后到场，确认为厨师爆炒油烟。系统记录案例，完成闭环。",
      activeSensors: ['execution'],
      systemLog: [
        "> Dispatch: Guard 'Li' (Dist: 50m)",
        "> Status: Arrived (3 min)",
        "> Feedback: 'Chef cooking, heavy smoke, no fire'",
        "> System: Case Closed. Model updated."
      ]
    }
  ];

  // 自动播放逻辑
  useEffect(() => {
    let interval: any;
    if (isAutoPlay && currentStep < steps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => prev + 1);
      }, 3000); // 每3秒一步
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

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-8 flex flex-col items-center">
      
      {/* 顶部标题栏 */}
      <header className="w-full max-w-6xl mb-8 flex flex-col md:flex-row justify-between items-center border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent flex items-center gap-3">
            <BrainCircuit className="w-8 h-8 text-blue-400" />
            Sup-01 数智主管
          </h1>
          <p className="text-slate-400 mt-1">场景：3F 餐饮区油烟误报智能研判演示</p>
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
            {/* 连接线 */}
            <div className="absolute left-3.5 top-2 bottom-4 w-0.5 bg-slate-700 z-0"></div>
            
            {steps.map((step, index) => (
              <div 
                key={step.id} 
                className={`relative z-10 pl-10 py-3 cursor-pointer transition-all ${index === currentStep ? 'opacity-100' : 'opacity-40'}`}
                onClick={() => setCurrentStep(index)}
              >
                <div className={`absolute left-1 top-4 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  index <= currentStep ? 'bg-blue-500 border-blue-500' : 'bg-slate-800 border-slate-600'
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

        {/* 中间：数字孪生/态势感知仪表盘 */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          
          {/* 状态总览卡片 */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 relative overflow-hidden">
            {currentStep > 0 && currentStep < 4 && (
              <div className="absolute top-0 right-0 p-2 bg-amber-500/20 text-amber-400 text-xs font-bold px-3 rounded-bl-xl border-b border-l border-amber-500/30 animate-pulse">
                研判进行中...
              </div>
            )}
             {currentStep === 4 && (
              <div className="absolute top-0 right-0 p-2 bg-green-500/20 text-green-400 text-xs font-bold px-3 rounded-bl-xl border-b border-l border-green-500/30">
                决策已生成
              </div>
            )}

            <div className="flex justify-between items-end mb-6">
              <div>
                 <div className="text-slate-400 text-sm mb-1 flex items-center gap-1"><MapPin size={14}/> 位置</div>
                 <div className="text-xl font-bold">3F-C区 15号点位</div>
                 <div className="text-slate-500 text-xs">火锅店后厨 / 晚餐高峰期</div>
              </div>
              <div className="text-right">
                 <div className="text-slate-400 text-sm mb-1"><Clock size={14} className="inline mr-1"/>系统耗时</div>
                 <div className="text-2xl font-mono text-cyan-400">
                   {currentStep === 0 ? '0 ms' : currentStep === 1 ? '10 ms' : currentStep === 2 ? '300 ms' : currentStep === 3 ? '450 ms' : currentStep === 4 ? '500 ms' : '8 min'}
                 </div>
              </div>
            </div>

            {/* 传感器矩阵 */}
            <div className="grid grid-cols-2 gap-3">
              {/* 烟感 */}
              <SensorCard 
                icon={<Wind size={20}/>} 
                label="光电烟感" 
                value={currentStep >= 1 ? "3.2 %/m" : "0.5 %/m"} 
                status={currentStep >= 1 ? "alert" : "normal"}
                subText={currentStep >= 1 ? "阈值: 3.0%" : "正常"}
              />
              {/* 视觉 */}
              <SensorCard 
                icon={<Eye size={20}/>} 
                label="视觉复核" 
                value={currentStep >= 2 ? "白色雾气" : "待机"} 
                status={currentStep >= 2 ? "warning" : "normal"}
                subText={currentStep >= 2 ? "未见明火" : "---"}
              />
              {/* 热成像 */}
              <SensorCard 
                icon={<Thermometer size={20}/>} 
                label="热成像" 
                value={currentStep >= 2 ? "58.0°C" : "26.0°C"} 
                status={currentStep >= 2 && currentStep < 4 ? "normal" : "normal"} 
                // 注意：在Sup-01逻辑中，58度对于火灾来说是“正常/低温”，所以这里给normal色调，体现AI的判断
                subText={currentStep >= 2 ? "无异常温升" : "正常"}
                highlight={currentStep >= 3} // 在物理铁律检查时高亮
              />
              {/* 气体 */}
              <SensorCard 
                icon={<Activity size={20}/>} 
                label="气体分析" 
                value={currentStep >= 2 ? "VOC高 / CO低" : "正常"} 
                status={currentStep >= 2 ? "warning" : "normal"}
                subText={currentStep >= 2 ? "CO: 8ppm" : "CO: 2ppm"}
                highlight={currentStep >= 3}
              />
            </div>
          </div>

          {/* 研判结论区 (L3阶段显示) */}
          <div className={`bg-slate-800 rounded-xl p-6 border transition-all duration-500 ${
            currentStep >= 4 
              ? 'border-green-500/50 shadow-[0_0_20px_rgba(34,197,94,0.1)]' 
              : 'border-slate-700 opacity-50 grayscale'
          }`}>
             <h4 className="text-sm uppercase tracking-wider text-slate-400 mb-4 font-bold flex items-center gap-2">
               <Server size={16}/> 研判结论 (L3 决策层)
             </h4>
             <div className="flex items-center gap-6">
               <div className="relative w-24 h-24 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                    <path className="text-slate-700" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="4" />
                    <path className="text-green-500 transition-all duration-1000 ease-out" 
                          strokeDasharray={currentStep >= 4 ? "96, 100" : "0, 100"} 
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                          fill="none" stroke="currentColor" strokeWidth="4" />
                  </svg>
                  <div className="absolute text-center">
                    <div className="text-2xl font-bold text-white">{currentStep >= 4 ? "96%" : "--"}</div>
                    <div className="text-[10px] text-slate-400">误报置信度</div>
                  </div>
               </div>
               
               <div className="flex-1 space-y-2">
                 <div className="flex justify-between border-b border-slate-700 pb-2">
                   <span className="text-slate-400">研判结果</span>
                   <span className={`font-bold ${currentStep >= 4 ? "text-green-400" : "text-slate-500"}`}>
                     {currentStep >= 4 ? "油烟误报 (False Alarm)" : "计算中..."}
                   </span>
                 </div>
                 <div className="flex justify-between border-b border-slate-700 pb-2">
                   <span className="text-slate-400">风险等级</span>
                   <span className={`font-bold px-2 rounded ${currentStep >= 4 ? "bg-blue-900 text-blue-300" : "bg-slate-700 text-slate-500"}`}>
                     {currentStep >= 4 ? "P3 (低风险)" : "--"}
                   </span>
                 </div>
                 <div className="flex justify-between items-center">
                   <span className="text-slate-400">处置动作</span>
                   <span className={`text-sm ${currentStep >= 4 ? "text-white" : "text-slate-500"}`}>
                     {currentStep >= 4 ? "抑制报警 + 派单核实" : "--"}
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
                 <div className="text-blue-500 mb-1 opacity-70">[{step.time}] {step.phase}</div>
                 {step.systemLog.map((log, i) => (
                   <div key={i} className={`pl-2 border-l-2 mb-1 leading-relaxed ${
                     log.includes('ALERT') ? 'border-red-500 text-red-400' :
                     log.includes('DECISION') ? 'border-green-500 text-green-400 font-bold' :
                     log.includes('Physics') ? 'border-amber-500 text-amber-300' :
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
           当前场景: <span className="text-white font-medium">{currentData.description}</span>
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
      <div className="text-lg font-bold truncate">{value}</div>
      <div className="text-[10px] opacity-60 mt-1">{subText}</div>
    </div>
  );
}