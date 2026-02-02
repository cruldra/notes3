import React, { useState, useEffect, useRef } from 'react';
import { 
  Shield, 
  Eye, 
  Thermometer, 
  Wind, 
  FileText, 
  Activity, 
  Cpu, 
  Zap, 
  AlertTriangle, 
  CheckCircle, 
  Server,
  Database,
  Layers,
  Play
} from 'lucide-react';

const ArchitectureDemo = () => {
  const [activeScenario, setActiveScenario] = useState(null);
  const [processingStep, setProcessingStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);

  // Scroll logs to bottom
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
    setLogs(prev => [...prev, { time: timestamp, msg: message, type }]);
  };

  const scenarios = {
    falseAlarm: {
      id: 'falseAlarm',
      name: '场景一：3F餐饮区喷漆误报',
      desc: '烟感触发，但实际为装修喷漆作业',
      color: 'blue',
      data: {
        l1: {
          visual: { detect: '白色雾气', fire: false, smoke: true, confidence: 0.85 },
          thermal: { temp: '24.5°C', trend: '平稳' },
          gas: { voc: '3.2mg/m³ (High)', co: '0ppm', pm25: 'Low' },
          iot: { smoke_detector: 'Alarm (3.2%/m)' }
        },
        l2: {
          context: { workOrder: 'WO-20260201-0033 (墙面喷漆)', time: '工作时间' },
          fusion: '视觉白烟 + 低温 + 高VOC + 喷漆工单',
          reasoning: '符合[喷漆误报模式]',
          risk: 'P3 (低风险)',
          confidence: '96%'
        },
        l3: {
          strategy: '抑制报警',
          action: '暂缓声光报警 3分钟'
        },
        l4: {
          dispatch: '派单保安携带气体检测仪核实',
          device: '保持排烟系统待命，暂不启动'
        }
      }
    },
    realFire: {
      id: 'realFire',
      name: '场景二：B2配电房电气火灾',
      desc: '电缆短路引发阴燃，转为明火',
      color: 'red',
      data: {
        l1: {
          visual: { detect: '黑烟 + 局部火花', fire: true, smoke: true, confidence: 0.98 },
          thermal: { temp: '180°C (局部)', trend: '急剧上升 (+30°C/min)' },
          gas: { voc: 'High', co: '65ppm (Rising)', pm25: 'High' },
          iot: { smoke_detector: 'Alarm', breaker: 'Trip' }
        },
        l2: {
          context: { workOrder: '无', location: '重点防火区(配电房)' },
          fusion: '明火视觉 + 极高温 + CO飙升',
          reasoning: '符合[电气火灾模式] - 蔓延风险高',
          risk: 'P0 (灾难级)',
          confidence: '99%'
        },
        l3: {
          strategy: '全楼报警 + 立即联动',
          action: '启动应急预案 A'
        },
        l4: {
          dispatch: '通知119 + 全员疏散',
          device: '切断非消防电源 | 启动气体灭火 | 迫降电梯'
        }
      }
    }
  };

  const startSimulation = (scenarioKey) => {
    setActiveScenario(scenarioKey);
    setProcessingStep(0);
    setLogs([]);
    const scenario = scenarios[scenarioKey];

    addLog(`系统初始化: Sup-01 消防主管智能体被唤醒`, 'system');
    addLog(`事件触发: ${scenario.name}`, 'warning');

    // Simulate Step 1: Perception (L1)
    setTimeout(() => {
      setProcessingStep(1);
      addLog(`[L1 感知层] 正在采集多模态数据...`);
      addLog(`>> 视觉: ${scenario.data.l1.visual.detect}`, 'data');
      addLog(`>> 热成像: ${scenario.data.l1.thermal.temp}`, 'data');
      addLog(`>> 气体传感器: CO ${scenario.data.l1.gas.co}`, 'data');
    }, 800);

    // Simulate Step 2: Understanding (L2)
    setTimeout(() => {
      setProcessingStep(2);
      addLog(`[L2 理解层] 调用 Skills 进行证据融合与推理...`);
      if (scenario.data.l2.context.workOrder !== '无') {
        addLog(`>> 上下文检索: 发现关联工单 ${scenario.data.l2.context.workOrder}`, 'success');
      } else {
        addLog(`>> 上下文检索: 无相关作业报备`, 'warning');
      }
      addLog(`>> 因果推理: ${scenario.data.l2.reasoning}`);
      addLog(`>> 风险定级: ${scenario.data.l2.risk} (置信度 ${scenario.data.l2.confidence})`, scenarioKey === 'realFire' ? 'error' : 'success');
    }, 2500);

    // Simulate Step 3: Decision (L3)
    setTimeout(() => {
      setProcessingStep(3);
      addLog(`[L3 决策层] 基于世界模型生成处置策略...`);
      addLog(`>> 决策输出: ${scenario.data.l3.strategy}`, 'system');
    }, 4000);

    // Simulate Step 4: Execution (L4)
    setTimeout(() => {
      setProcessingStep(4);
      addLog(`[L4 执行层] 下发指令至边缘节点...`);
      addLog(`>> 任务派发: ${scenario.data.l4.dispatch}`);
      addLog(`>> 设备联动: ${scenario.data.l4.device}`);
      addLog(`流程结束: 闭环完成，耗时 ${(Math.random() * 0.5 + 0.5).toFixed(2)}秒`, 'system');
    }, 5500);
  };

  const renderScenarioButton = (key) => {
    const s = scenarios[key];
    const isActive = activeScenario === key;
    return (
      <button
        onClick={() => startSimulation(key)}
        className={`flex flex-col items-start p-4 rounded-xl border transition-all duration-300 w-full mb-3 ${
          isActive 
            ? `bg-${s.color}-50 border-${s.color}-500 shadow-md ring-1 ring-${s.color}-500` 
            : 'bg-white border-gray-200 hover:border-blue-300 hover:bg-gray-50'
        }`}
      >
        <div className="flex items-center w-full justify-between">
          <span className={`font-bold ${isActive ? `text-${s.color}-700` : 'text-gray-700'}`}>{s.name}</span>
          {isActive && <Activity className={`w-4 h-4 text-${s.color}-500 animate-pulse`} />}
        </div>
        <span className="text-xs text-gray-500 mt-1 text-left">{s.desc}</span>
      </button>
    );
  };

  const LayerCard = ({ level, title, icon: Icon, active, data, scenarioType }) => {
    const isError = scenarioType === 'realFire';
    const activeClass = active 
      ? (isError ? 'border-red-500 bg-red-50 shadow-red-100' : 'border-blue-500 bg-blue-50 shadow-blue-100')
      : 'border-gray-200 bg-white opacity-60';
    
    return (
      <div className={`relative border-l-4 rounded-r-lg p-4 mb-4 transition-all duration-500 shadow-sm ${activeClass}`}>
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center space-x-2">
            <div className={`p-1.5 rounded-full ${active ? (isError ? 'bg-red-100' : 'bg-blue-100') : 'bg-gray-100'}`}>
              <Icon size={18} className={active ? (isError ? 'text-red-600' : 'text-blue-600') : 'text-gray-400'} />
            </div>
            <span className="font-bold text-sm text-gray-700">{level} {title}</span>
          </div>
          {active && <span className="text-xs font-mono text-gray-500 animate-pulse">Processing...</span>}
        </div>
        
        {active && data && (
          <div className="mt-3 space-y-2 text-xs">
            {Object.entries(data).map(([key, value], idx) => {
               if (typeof value === 'object') return null; // handled separately if needed
               return (
                 <div key={idx} className="flex justify-between border-b border-black/5 pb-1 last:border-0">
                   <span className="text-gray-500 capitalize">{key}:</span>
                   <span className="font-medium text-gray-800 text-right">{value}</span>
                 </div>
               );
            })}
            
            {/* Special rendering for L1 nested data */}
            {level === 'L1' && (
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="bg-white/60 p-1 rounded border">
                  <div className="text-[10px] text-gray-400 flex items-center"><Eye size={10} className="mr-1"/> 视觉</div>
                  <div className="font-semibold text-gray-800">{data.visual?.detect}</div>
                </div>
                <div className="bg-white/60 p-1 rounded border">
                  <div className="text-[10px] text-gray-400 flex items-center"><Thermometer size={10} className="mr-1"/> 热成像</div>
                  <div className="font-semibold text-gray-800">{data.thermal?.temp}</div>
                </div>
              </div>
            )}
             {/* Special rendering for L2 nested data */}
             {level === 'L2' && (
              <div className="mt-2 bg-white/60 p-2 rounded border border-dashed border-gray-300">
                <div className="text-[10px] text-gray-400 mb-1">因果推理引擎 (Skills)</div>
                <div className="flex flex-wrap gap-1">
                  <span className="px-1.5 py-0.5 bg-indigo-100 text-indigo-700 rounded text-[10px]">证据融合</span>
                  <span className="px-1.5 py-0.5 bg-indigo-100 text-indigo-700 rounded text-[10px]">场景匹配</span>
                  {data.context?.workOrder !== '无' && <span className="px-1.5 py-0.5 bg-green-100 text-green-700 rounded text-[10px]">工单关联</span>}
                </div>
                <div className="mt-1 font-bold text-gray-800">{data.reasoning}</div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6 font-sans text-slate-800">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 flex items-center justify-center gap-3">
          <Shield className="w-8 h-8 text-blue-600" />
          Sup-01 消防主管系统架构演示
        </h1>
        <p className="text-slate-500 mt-2 text-sm">基于多模态数据智能火警研判的认知四层架构 (L1-L4)</p>
      </header>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Controls */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center">
              <Play size={16} className="mr-2" /> 场景模拟
            </h2>
            {renderScenarioButton('falseAlarm')}
            {renderScenarioButton('realFire')}
            
            <div className="mt-6 pt-6 border-t border-slate-100">
              <h3 className="text-xs font-semibold text-slate-400 mb-2">架构图例</h3>
              <div className="space-y-2 text-xs text-slate-600">
                <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-blue-500 mr-2"></div> 正常/误报流程</div>
                <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-red-500 mr-2"></div> 紧急/真火流程</div>
                <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-indigo-500 mr-2"></div> AI原子能力(Skills)</div>
              </div>
            </div>
          </div>

          {/* System Status Mockup */}
          <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
             <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center">
              <Server size={16} className="mr-2" /> 系统状态
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-2 bg-green-50 rounded-lg">
                <div className="text-xs text-green-600">云边协同</div>
                <div className="font-bold text-green-800">在线</div>
              </div>
              <div className="text-center p-2 bg-blue-50 rounded-lg">
                <div className="text-xs text-blue-600">研判时延</div>
                <div className="font-bold text-blue-800">~450ms</div>
              </div>
            </div>
          </div>
        </div>

        {/* Center Column: Architecture Flow */}
        <div className="lg:col-span-5">
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-100 h-full relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Layers size={120} />
            </div>
            
            <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
              <Cpu className="mr-2 text-indigo-600" /> 认知推理主链路
            </h2>

            <div className="relative">
              {/* Vertical connecting line */}
              <div className="absolute left-6 top-4 bottom-4 w-0.5 bg-gray-100 z-0"></div>

              <div className="relative z-10">
                <LayerCard 
                  level="L1" 
                  title="感知认知层 (Perception)" 
                  icon={Eye} 
                  active={processingStep >= 1} 
                  data={activeScenario ? scenarios[activeScenario].data.l1 : null}
                  scenarioType={activeScenario}
                />
                
                <LayerCard 
                  level="L2" 
                  title="理解认知层 (Understanding)" 
                  icon={Database} 
                  active={processingStep >= 2} 
                  data={activeScenario ? scenarios[activeScenario].data.l2 : null}
                  scenarioType={activeScenario}
                />

                <LayerCard 
                  level="L3" 
                  title="计算认知层 (Decision)" 
                  icon={Cpu} 
                  active={processingStep >= 3} 
                  data={activeScenario ? scenarios[activeScenario].data.l3 : null}
                  scenarioType={activeScenario}
                />

                <LayerCard 
                  level="L4" 
                  title="执行认知层 (Execution)" 
                  icon={Zap} 
                  active={processingStep >= 4} 
                  data={activeScenario ? scenarios[activeScenario].data.l4 : null}
                  scenarioType={activeScenario}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Audit Log & Trace */}
        <div className="lg:col-span-4">
          <div className="bg-gray-900 text-green-400 p-5 rounded-2xl shadow-lg font-mono text-xs h-[600px] flex flex-col">
            <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
              <span className="font-bold flex items-center"><FileText size={14} className="mr-2"/> Audit Log / Trace</span>
              <span className="bg-gray-800 px-2 py-1 rounded text-[10px] text-gray-400">Live Stream</span>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-gray-700">
              {logs.length === 0 && (
                <div className="text-gray-600 italic text-center mt-20">等待事件触发...</div>
              )}
              {logs.map((log, index) => (
                <div key={index} className={`flex animate-fade-in-up`}>
                  <span className="text-gray-500 mr-3">[{log.time}]</span>
                  <span className={`${
                    log.type === 'error' ? 'text-red-400 font-bold' : 
                    log.type === 'success' ? 'text-green-300' :
                    log.type === 'warning' ? 'text-yellow-400' :
                    log.type === 'data' ? 'text-blue-300' :
                    log.type === 'system' ? 'text-purple-300' : 'text-gray-300'
                  }`}>
                    {log.type === 'data' && '> '}
                    {log.msg}
                  </span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ArchitectureDemo;