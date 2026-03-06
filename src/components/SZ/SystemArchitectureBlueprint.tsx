import React, { useState, useEffect } from 'react';
import { 
  Database, Server, Cpu, Layers, Smartphone, 
  Monitor, Presentation, ShieldCheck, Activity,
  BrainCircuit, Share2, Network, ArrowUpCircle,
  Settings, Key, Fingerprint, Lock, CheckCircle
} from 'lucide-react';

export default function SystemArchitectureBlueprint() {
  const [activeLayer, setActiveLayer] = useState('data'); // 默认选中数据中台层，因为这是卖点
  const [activeNode, setActiveNode] = useState(null);

  // 架构层级定义 (严格对应 PRD 4.1 和 8.1 章节)
  const architectureData = [
    {
      id: 'frontend',
      title: '用户触达层 (前台展示)',
      icon: <Layers className="text-blue-500" size={24} />,
      gradient: 'from-blue-50 to-slate-50 border-blue-200',
      textColor: 'text-blue-800',
      description: '多端协同，为学生、辅导员、校领导提供千人千面的专属交互入口，彻底告别过去繁琐的办事大厅。',
      techStack: ['React / Vue3', 'Flutter / UniApp', 'ECharts / DataV', 'WebSocket (实时流)'],
      nodes: [
        { id: 'f1', name: 'AI 学生社区 (H5/小程序)', icon: <Smartphone size={18}/>, type: 'app' },
        { id: 'f2', name: '辅导员预警工作台 (PC)', icon: <Monitor size={18}/>, type: 'app' },
        { id: 'f3', name: '校级决策全景大屏 (指挥中心)', icon: <Presentation size={18}/>, type: 'app' }
      ]
    },
    {
      id: 'service',
      title: '业务微服务层 (中台逻辑)',
      icon: <Server className="text-indigo-500" size={24} />,
      gradient: 'from-indigo-50 to-slate-50 border-indigo-200',
      textColor: 'text-indigo-800',
      description: '采用 Spring Cloud 微服务架构，高内聚低耦合。所有的业务逻辑在这里被解耦为独立的服务，支持万级并发请求。',
      techStack: ['Spring Cloud Alibaba', 'Node.js', 'Redis 缓存', 'RabbitMQ / Kafka'],
      nodes: [
        { id: 's1', name: '智能对话服务', icon: <Settings size={16}/> },
        { id: 's2', name: '思政教育与图谱服务', icon: <Settings size={16}/> },
        { id: 's3', name: '学生全景画像服务', icon: <Settings size={16}/> },
        { id: 's4', name: '四级风险预警调度引擎', icon: <Settings size={16}/> }
      ]
    },
    {
      id: 'ai',
      title: 'AI 核心能力层 (智慧大脑)',
      icon: <Cpu className="text-purple-500" size={24} />,
      gradient: 'from-purple-100 to-fuchsia-50 border-purple-300 shadow-inner',
      textColor: 'text-purple-900',
      description: '整个系统的“灵魂”。本地化部署/调用行业大模型，结合校园专有语料微调，确保数据不出校的前提下实现精准的语义理解与推理。',
      techStack: ['私有化部署 LLM (千问/GLM等)', 'Neo4j 图数据库', 'LangChain / RAG', 'PyTorch 深度学习框架'],
      nodes: [
        { id: 'a1', name: '校园垂直大语言模型 (LLM)', icon: <BrainCircuit size={16}/>, highlight: true },
        { id: 'a2', name: '思政知识图谱引擎 (KG)', icon: <Share2 size={16}/>, highlight: true },
        { id: 'a3', name: 'NLP 语义与情感分析', icon: <Activity size={16}/> },
        { id: 'a4', name: '个性化精准推荐算法', icon: <Network size={16}/> }
      ]
    },
    {
      id: 'data',
      title: '中央厨房数据中台 (核心资产)',
      icon: <Database className="text-red-500" size={24} />,
      gradient: 'from-red-50 to-orange-50 border-red-300 border-dashed',
      textColor: 'text-red-800',
      description: '【解决最大痛点】通过 ETL 工具，将各学院、各部门孤立的老旧系统数据汇聚一堂，清洗、脱敏后形成统一的高价值数字资产池。',
      techStack: ['Hadoop / Spark 批处理', 'Flink 实时流计算', 'MySQL + Elasticsearch', 'Kettle / DataX (ETL)'],
      nodes: [
        { id: 'd1', name: '异构数据接入网关 (ETL)', icon: <Database size={16}/> },
        { id: 'd2', name: '标准化清洗与脱敏处理', icon: <ShieldCheck size={16}/> },
        { id: 'd3', name: '统一学生数据湖仓', icon: <Database size={16}/> },
        { id: 'd4', name: '标准 API 开放服务', icon: <Network size={16}/> }
      ]
    },
    {
      id: 'source',
      title: '校园既有基础系统层 (孤岛源头)',
      icon: <Layers className="text-slate-500" size={24} />,
      gradient: 'bg-slate-100 border-slate-300',
      textColor: 'text-slate-700',
      description: '学校长年累积的基础业务系统。我们不需要推翻重做，而是通过无损对接，让沉睡的数据产生新价值。',
      techStack: ['Oracle / SQL Server', 'RESTful API', '视图共享 / 数据库直连', '硬件物联网对接'],
      nodes: [
        { id: 'src1', name: '教务/研工系统 (成绩/学籍)', type: 'legacy' },
        { id: 'src2', name: '心康测评系统 (量表/记录)', type: 'legacy' },
        { id: 'src3', name: '安防门禁系统 (行为轨迹)', type: 'legacy' },
        { id: 'src4', name: '后勤一卡通 (消费画像)', type: 'legacy' }
      ]
    }
  ];

  const handleNodeClick = (layer, node, e) => {
    e.stopPropagation();
    setActiveLayer(layer.id);
    setActiveNode(node);
  };

  const getActiveLayerData = () => {
    return architectureData.find(l => l.id === activeLayer);
  };

  return (
    <div className="flex h-screen w-full bg-[#f0f4f8] font-sans overflow-hidden">
      
      {/* 左侧：架构图可视化区 */}
      <div className="flex-[3] p-8 overflow-y-auto relative">
        <div className="max-w-4xl mx-auto">
          
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-black text-slate-800 tracking-tight flex items-center gap-3">
                <Database className="text-blue-600" />
                AI 智慧思政社区系统架构蓝图
              </h1>
              <p className="text-sm text-slate-500 mt-2 font-medium">遵循 PRD v1.0 规范构筑的高扩展、微服务、AI 原生架构</p>
            </div>
            
            <div className="flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow-sm border border-slate-200">
              <span className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse"></span>
              <span className="text-xs font-bold text-slate-600">系统运行状态：正常</span>
            </div>
          </div>

          <div className="relative space-y-6">
            
            {/* 贯穿层级的数据流动画 (模拟数据自底向上流转) */}
            <div className="absolute left-1/2 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-300 via-purple-300 to-red-300 opacity-20 -translate-x-1/2 z-0 rounded-full"></div>
            
            <div className="absolute left-[30%] top-10 bottom-10 w-px bg-slate-300/50 z-0">
               <div className="w-1.5 h-8 bg-blue-400 rounded-full absolute top-full -translate-x-1/2 animate-[flow_3s_linear_infinite]"></div>
            </div>
            <div className="absolute left-[70%] top-10 bottom-10 w-px bg-slate-300/50 z-0">
               <div className="w-1.5 h-8 bg-purple-400 rounded-full absolute top-full -translate-x-1/2 animate-[flow_4s_linear_infinite_0.5s]"></div>
            </div>

            <style>{`
              @keyframes flow {
                0% { top: 100%; opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { top: 0%; opacity: 0; }
              }
            `}</style>

            {/* 逐层渲染架构 */}
            {architectureData.map((layer) => (
              <div 
                key={layer.id}
                onClick={() => { setActiveLayer(layer.id); setActiveNode(null); }}
                className={`relative z-10 p-5 rounded-2xl border-2 transition-all cursor-pointer bg-gradient-to-r ${layer.gradient} ${
                  activeLayer === layer.id ? 'ring-4 ring-blue-500/20 scale-[1.02] shadow-xl' : 'shadow-sm hover:shadow-md'
                }`}
              >
                {/* 层级标题 */}
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-white p-2 rounded-lg shadow-sm">{layer.icon}</div>
                  <h2 className={`text-lg font-bold ${layer.textColor}`}>{layer.title}</h2>
                  {activeLayer === layer.id && <span className="ml-auto text-xs font-bold bg-white px-3 py-1 rounded-full shadow-sm text-slate-600 border border-slate-200 animate-in fade-in">当前选中层级</span>}
                </div>

                {/* 节点容器 */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {layer.nodes.map((node) => (
                    <div 
                      key={node.id}
                      onClick={(e) => handleNodeClick(layer, node, e)}
                      className={`
                        flex items-center justify-center text-center gap-2 p-3 rounded-xl text-sm font-semibold transition-all border
                        ${node.type === 'legacy' ? 'bg-slate-200/50 border-slate-300 text-slate-600 hover:bg-slate-200' : 
                          node.type === 'app' ? 'bg-white border-blue-100 text-blue-800 shadow-sm hover:shadow-md hover:border-blue-300' :
                          node.highlight ? 'bg-purple-600 border-purple-700 text-white shadow-md hover:bg-purple-700' :
                          'bg-white/80 backdrop-blur border-white/50 text-slate-700 shadow-sm hover:bg-white hover:shadow-md'
                        }
                        ${activeNode?.id === node.id ? 'ring-2 ring-offset-2 ring-blue-500 scale-105' : ''}
                      `}
                    >
                      {node.icon && <span className="flex-shrink-0">{node.icon}</span>}
                      <span className="line-clamp-2">{node.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* 安全与运维护栏 (贯穿垂直方向) */}
            <div className="absolute -left-6 top-10 bottom-10 w-12 border-l-2 border-y-2 border-dashed border-slate-300 rounded-l-3xl flex flex-col justify-center items-center py-10 gap-20 opacity-50">
               <div className="bg-slate-100 p-2 rounded-full -translate-x-1/2" title="统一身份认证"><Key size={20} className="text-slate-500"/></div>
               <div className="bg-slate-100 p-2 rounded-full -translate-x-1/2" title="数据安全加密"><Lock size={20} className="text-slate-500"/></div>
               <div className="bg-slate-100 p-2 rounded-full -translate-x-1/2" title="操作审计日志"><Fingerprint size={20} className="text-slate-500"/></div>
            </div>
            <div className="absolute -left-[4.5rem] top-1/2 -translate-y-1/2 -rotate-90 text-xs font-bold text-slate-400 uppercase tracking-widest whitespace-nowrap">
              全链路安全与合规体系
            </div>

          </div>
        </div>
      </div>

      {/* 右侧：交互式详情解说面板 (向客户汇报的核心话术区) */}
      <div className="flex-[1.5] bg-white border-l border-slate-200 flex flex-col shadow-2xl z-20">
        
        <div className="h-16 flex items-center px-6 border-b border-slate-100 bg-slate-50">
          <h2 className="font-bold text-slate-800 flex items-center gap-2">
            <Presentation size={20} className="text-blue-600"/> 架构与价值解析
          </h2>
        </div>

        <div className="p-6 flex-1 overflow-y-auto">
          {getActiveLayerData() && (
            <div className="animate-in fade-in slide-in-from-right-4 duration-300">
              
              {/* 层级概览 */}
              <div className={`p-4 rounded-2xl mb-6 bg-gradient-to-br ${getActiveLayerData().gradient}`}>
                <div className="flex items-center gap-3 mb-2">
                  <div className="bg-white p-2 rounded-lg shadow-sm">{getActiveLayerData().icon}</div>
                  <h3 className={`text-xl font-bold ${getActiveLayerData().textColor}`}>
                    {getActiveLayerData().title}
                  </h3>
                </div>
                <p className="text-sm text-slate-700 leading-relaxed font-medium mt-3">
                  {getActiveLayerData().description}
                </p>
              </div>

              {/* 核心技术栈说明 */}
              <div className="mb-8">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">关键技术选型 (匹配 PRD 8.2)</h4>
                <div className="flex flex-wrap gap-2">
                  {getActiveLayerData().techStack.map((tech, idx) => (
                    <span key={idx} className="px-3 py-1.5 bg-slate-100 text-slate-600 text-xs font-bold rounded-lg border border-slate-200">
                      {tech}
                    </span>
                  ))}
                </div>
              </div>

              {/* 节点详情 (如果有选中具体模块) */}
              <div className="border-t border-slate-100 pt-6">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
                  {activeNode ? '选中模块深度解析' : '该层级商业价值落地'}
                </h4>
                
                {activeNode ? (
                  <div className="bg-blue-50 border border-blue-100 p-5 rounded-2xl">
                    <div className="flex items-center gap-2 mb-3">
                      <CheckCircle className="text-blue-500" size={20}/>
                      <h5 className="font-bold text-blue-900 text-lg">{activeNode.name}</h5>
                    </div>
                    <p className="text-sm text-blue-800/80 leading-relaxed mb-4">
                      {/* 这里为了演示效果，简单的写了一个针对不同节点的通用描述逻辑。实际中可根据每个节点的特性写死文案 */}
                      该模块是 {getActiveLayerData().title} 的关键组件。
                      {activeNode.id.startsWith('a') && '它使得系统具备了类人的逻辑推理和语义理解能力，不再是死板的规则引擎。'}
                      {activeNode.id.startsWith('d') && '这是解决校园数据孤岛的核心动作，通过自动化脚本让数据跑起来。'}
                      {activeNode.id.startsWith('s') && '独立部署，互不影响。即便在选课、抢票等高并发期，预警系统依然稳如磐石。'}
                      {activeNode.id.startsWith('f') && '直接服务于最终用户体验，界面极简，隐藏了底层所有复杂的数据交互逻辑。'}
                      {activeNode.type === 'legacy' && '针对老旧系统，我们提供免API侵入的数据库视图同步技术，最大程度降低学校协调难度。'}
                    </p>
                    
                    {/* 伪代码/技术细节装逼块 */}
                    <div className="bg-slate-900 rounded-xl p-4 overflow-hidden shadow-inner">
                      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-slate-700">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                        <span className="text-[10px] text-slate-400 font-mono ml-2">system_monitor.log</span>
                      </div>
                      <pre className="text-[10px] text-green-400 font-mono">
                        {`[INFO] 正在初始化模块配置...
[SUCCESS] ${activeNode.name} 加载完成.
[NETWORK] 监控到 1,245 次内部 RPC 调用.
[STATUS] CPU: 12% | 延迟: 14ms | 健康度: 优`}
                      </pre>
                    </div>
                  </div>
                ) : (
                  // 未选中具体节点时的默认话术
                  <div className="space-y-4">
                    {activeLayer === 'data' && (
                      <div className="bg-red-50 p-4 rounded-xl border border-red-100 flex items-start gap-3">
                        <ShieldCheck className="text-red-500 flex-shrink-0 mt-0.5" size={18}/>
                        <div>
                          <div className="font-bold text-red-900 text-sm mb-1">避坑指南 (刘总专供)</div>
                          <div className="text-xs text-red-800/80 leading-relaxed">
                            这一层是整个项目最容易亏钱的地方。在合同中必须严格界定接入“校园既有系统”的数量和接口标准。建议标配只接入拥有标准 API 的系统。
                          </div>
                        </div>
                      </div>
                    )}
                    {activeLayer === 'ai' && (
                      <div className="bg-purple-50 p-4 rounded-xl border border-purple-100 flex items-start gap-3">
                        <BrainCircuit className="text-purple-500 flex-shrink-0 mt-0.5" size={18}/>
                        <div>
                          <div className="font-bold text-purple-900 text-sm mb-1">合规与隐私底线</div>
                          <div className="text-xs text-purple-800/80 leading-relaxed">
                            因为涉及学生心理测评等极其敏感的数据，我们推荐采用“行业开源模型 + 校园本地化微调部署”的方案，确保核心数据绝不上公有云，彻底打消校领导的安全顾虑。
                          </div>
                        </div>
                      </div>
                    )}
                    {activeLayer !== 'data' && activeLayer !== 'ai' && (
                      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex items-start gap-3">
                        <ArrowUpCircle className="text-slate-500 flex-shrink-0 mt-0.5" size={18}/>
                        <div>
                          <div className="font-bold text-slate-700 text-sm mb-1">点击左侧具体模块查看详细说明</div>
                          <div className="text-xs text-slate-500 leading-relaxed">
                            您可以向客户展示我们每一个技术组件是如何直接回应 PRD 中的业务痛点的。
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}