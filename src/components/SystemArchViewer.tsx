import React, { useState, useEffect } from 'react';
import { 
  Server, 
  BrainCircuit, 
  ShieldCheck, 
  RefreshCw, 
  Database, 
  Layers, 
  Cpu, 
  Box, 
  Activity, 
  GitMerge, 
  Zap, 
  Lock,
  ArrowUp,
  ArrowDown,
  MousePointer2,
  ChevronRight, // 修复：添加缺失的图标引用
  CheckCircle
} from 'lucide-react';

// --- 数据定义 ---
// 修改：将 icon 属性改为组件引用，而不是直接存储 JSX 元素，避免渲染错误
const architectureLayers = [
  {
    id: 'layer5',
    title: 'L5: 持续进化 (Evolution)',
    icon: RefreshCw, 
    color: 'from-fuchsia-600 to-pink-600',
    description: '系统的“成长机制”。通过人工反馈和负样本挖掘，不断微调模型，越用越准。',
    features: [
      { name: 'RLHF (人类反馈)', desc: '保安反馈误报数据->自动入库->模型微调' },
      { name: '负样本挖掘', desc: '自动识别对抗样本（如夕阳反光、红衣）' },
      { name: '模型热更新', desc: 'OTA下发新权重，边缘节点无感升级' }
    ]
  },
  {
    id: 'layer4',
    title: 'L4: 认知编排 (Cognitive Engine)',
    icon: BrainCircuit,
    color: 'from-violet-600 to-indigo-600',
    description: '系统的“大脑”。基于Skills-First架构，动态组合原子能力，适应不同场景。',
    features: [
      { name: 'Skills-First 架构', desc: '200+原子能力拼装 (如:视觉验火+工单查询)' },
      { name: '动态编排引擎', desc: '根据事件类型(烟雾/高温)生成处理DAG图' },
      { name: '多模态融合', desc: '视觉+热+气+数据，四维证据交叉验证' }
    ]
  },
  {
    id: 'layer3',
    title: 'L3: 安全阀与治理 (Safety Guardrails)',
    icon: ShieldCheck,
    color: 'from-red-600 to-orange-600',
    description: '系统的“保险丝”。强制执行物理铁律和合规检查，防止AI幻觉和越权。',
    features: [
      { name: '物理铁律 (Iron Laws)', desc: '无热不火、无CO不阴燃 (强制否决AI幻觉)' },
      { name: '人在回路 (HITL)', desc: 'P0级高危决策强制要求人工确认' },
      { name: '合规性校验', desc: '确保联动逻辑符合消防规范 (如先断电后喷水)' }
    ]
  },
  {
    id: 'layer2',
    title: 'L2: 数字孪生与上下文 (Context)',
    icon: Database,
    color: 'from-blue-600 to-cyan-600',
    description: '系统的“世界观”。提供空间坐标和业务背景，让数据具备语义。',
    features: [
      { name: '空间语义映射', desc: '传感器ID -> 3D坐标 (x,y,z) + 防火分区' },
      { name: '业务上下文 RAG', desc: '实时关联工单系统、日程表、资产库' },
      { name: '时空索引', desc: '快速检索“半径30米内最近的灭火器”' }
    ]
  },
  {
    id: 'layer1',
    title: 'L1: 边缘基础设施 (Edge Infra)',
    icon: Server,
    color: 'from-slate-600 to-slate-800',
    description: '系统的“躯干”。高性能边缘计算节点，保障极致速度和断网可用性。',
    features: [
      { name: '边缘算力节点', desc: 'GPU工业级服务器，本地推理 <5ms' },
      { name: '断网自治模式', desc: '云端断连时，功能保留度 >80%' },
      { name: '全域感知接入', desc: 'Onvif视频流 + IoT传感器 (MQTT/Modbus)' }
    ]
  }
];

export default function SystemArchViewer() {
  const [activeLayer, setActiveLayer] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  // 模拟数据流动画
  useEffect(() => {
    const interval = setInterval(() => {
      setIsAnimating(true);
      setTimeout(() => setIsAnimating(false), 2000);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans flex flex-col items-center">
      
      {/* 顶部标题 */}
      <header className="max-w-5xl w-full mb-8 text-center">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent flex items-center justify-center gap-3">
          <Layers className="text-blue-500" /> Sup-01 系统技术架构视图
        </h1>
        <p className="text-slate-400 mt-2">基于“认知驱动”的五层工程化架构 | 点击层级查看技术细节</p>
      </header>

      <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* 左侧：核心架构堆栈 (交互区) */}
        <div className="lg:col-span-7 flex flex-col gap-4 relative">
          
          {/* 贯穿的数据流线 (装饰) */}
          <div className="absolute left-8 top-4 bottom-4 w-1 bg-slate-800 rounded-full hidden md:block"></div>
          {isAnimating && (
            <div className="absolute left-8 top-full w-1 h-20 bg-gradient-to-b from-cyan-500 to-transparent rounded-full animate-flow-up hidden md:block" style={{animationDuration: '2s'}}></div>
          )}

          {architectureLayers.map((layer, index) => (
            <div 
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              className={`
                relative group cursor-pointer 
                bg-slate-900 border transition-all duration-300 rounded-xl p-5
                flex items-center gap-5 hover:translate-x-2
                ${activeLayer === layer.id 
                  ? `border-l-4 border-t border-r border-b border-l-${layer.color.split(' ')[1].replace('to-', '')} border-slate-600 bg-slate-800 shadow-2xl` 
                  : 'border-slate-800 hover:border-slate-600 border-l-4 border-l-transparent'}
              `}
            >
              {/* 图标 (改为组件调用) */}
              <div className={`
                w-12 h-12 rounded-lg flex items-center justify-center bg-gradient-to-br ${layer.color} shadow-lg shrink-0
                ${activeLayer === layer.id ? 'scale-110' : 'opacity-80 group-hover:opacity-100'}
              `}>
                <layer.icon size={24} />
              </div>

              {/* 标题与简述 */}
              <div className="flex-1">
                <h3 className={`font-bold text-lg ${activeLayer === layer.id ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                  {layer.title}
                </h3>
                <p className="text-sm text-slate-500 mt-1">{layer.description}</p>
              </div>

              {/* 指示箭头 (已修复引用错误) */}
              <ChevronRight className={`text-slate-600 transition-transform ${activeLayer === layer.id ? 'rotate-90 text-white' : ''}`} />
              
              {/* 动态流光效果 (选中时) */}
              {activeLayer === layer.id && (
                <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-scan"></div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* 右侧：详情解析面板 */}
        <div className="lg:col-span-5">
          <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-6 h-full sticky top-8 backdrop-blur-sm">
            
            {!activeLayer ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 py-20">
                <MousePointer2 size={48} className="mb-4 opacity-50 animate-bounce" />
                <p>请点击左侧架构层级</p>
                <p className="text-sm">查看核心技术支柱详情</p>
              </div>
            ) : (
              <div className="animate-fade-in">
                {(() => {
                  const layer = architectureLayers.find(l => l.id === activeLayer);
                  if (!layer) return null;
                  const IconComponent = layer.icon;

                  return (
                    <>
                      <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold bg-gradient-to-r ${layer.color} mb-4`}>
                        TECH PILLAR
                      </div>
                      <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
                        <IconComponent size={28} /> {layer.title.split(':')[1]}
                      </h2>
                      <p className="text-slate-400 mb-6 text-sm leading-relaxed border-b border-slate-800 pb-4">
                        {layer.description}
                      </p>

                      <div className="space-y-4">
                        <h4 className="text-sm font-bold text-slate-300 uppercase tracking-wider">核心能力 (Core Capabilities)</h4>
                        {layer.features.map((feature, idx) => (
                          <div key={idx} className="bg-slate-800 p-4 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors">
                            <div className="flex items-start gap-3">
                              <div className="mt-1 text-cyan-400">
                                <CheckCircleIcon size={16} />
                              </div>
                              <div>
                                <div className="font-bold text-slate-200 text-sm">{feature.name}</div>
                                <div className="text-slate-400 text-xs mt-1">{feature.desc}</div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* 场景模拟小贴士 */}
                      <div className="mt-8 bg-slate-800/50 p-4 rounded-lg border border-slate-700 text-xs text-slate-400">
                        <div className="font-bold text-slate-300 mb-1 flex items-center gap-2">
                          <Zap size={14} className="text-yellow-400"/> 
                          实际场景映射
                        </div>
                        {layer.id === 'layer1' && "当烟感报警触发时，边缘节点在本地毫秒级处理信号，无需等待云端响应。"}
                        {layer.id === 'layer2' && "系统立即反查 B2-E-01 的 3D 坐标，并检索当前时段是否有检修工单。"}
                        {layer.id === 'layer3' && "若视觉AI判定为火，但物理铁律检测到温度未升高，强制拦截报警。"}
                        {layer.id === 'layer4' && "自动编排 '视觉复核' + '气体分析' Skills 并行运行，生成研判结论。"}
                        {layer.id === 'layer5' && "若保安反馈是误报，该案例自动进入负样本库，用于今晚的模型微调。"}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}
          </div>
        </div>

      </div>
      
      {/* 底部备注 */}
      <div className="max-w-6xl w-full mt-12 pt-6 border-t border-slate-800 flex justify-between text-slate-500 text-xs">
        <div>Sup-01 Intelligent Agent Architecture V2.0</div>
        <div className="flex gap-4">
          <span className="flex items-center gap-1"><Box size={12}/> Skills-First</span>
          <span className="flex items-center gap-1"><Cpu size={12}/> Edge Computing</span>
          <span className="flex items-center gap-1"><Lock size={12}/> Physical Laws</span>
        </div>
      </div>

      <style>{`
        @keyframes flow-up {
          0% { top: 100%; opacity: 0; }
          50% { opacity: 1; }
          100% { top: 0%; opacity: 0; }
        }
        .animate-flow-up {
          animation: flow-up 2s infinite linear;
        }
        @keyframes scan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-scan {
          animation: scan 3s infinite linear;
        }
      `}</style>
    </div>
  );
}

// 辅助组件
function CheckCircleIcon({size}: {size: number}) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="3" 
      strokeLinecap="round" 
      strokeLinejoin="round"
    >
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  )
}