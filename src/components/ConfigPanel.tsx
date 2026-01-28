import React, { useState } from 'react';
import { 
  Settings, 
  GitBranch, 
  Database, 
  MessageSquare, 
  Phone, 
  Smartphone, 
  Save, 
  Play, 
  Plus, 
  Search, 
  MoreHorizontal,
  Layout,
  Clock,
  AlertCircle,
  Check,
  Edit2,
  Trash2,
  ToggleLeft,
  ToggleRight,
  Zap
} from 'lucide-react';

// --- 子组件: 策略节点 (Canvas Node) ---
const Node = ({ type, title, subtitle, x, y, active }) => {
  const styles = {
    trigger: "bg-blue-50 border-blue-200 text-blue-800",
    condition: "bg-yellow-50 border-yellow-200 text-yellow-800",
    action: "bg-white border-gray-200 text-gray-800",
    end: "bg-gray-100 border-gray-200 text-gray-500"
  };

  const icons = {
    trigger: <Zap size={16} className="text-blue-500" />,
    condition: <GitBranch size={16} className="text-yellow-500" />,
    action: <MessageSquare size={16} className="text-purple-500" />,
    end: <div className="w-3 h-3 bg-gray-400 rounded-full" />
  };

  return (
    <div 
      className={`absolute w-64 p-3 rounded-lg border-2 shadow-sm cursor-move transition-all hover:shadow-md hover:border-blue-400 ${styles[type]} ${active ? 'ring-2 ring-blue-500 ring-offset-2' : ''}`}
      style={{ left: x, top: y }}
    >
      <div className="flex items-center gap-2 mb-1">
        {icons[type]}
        <span className="font-bold text-sm">{title}</span>
      </div>
      <div className="text-xs opacity-80">{subtitle}</div>
      
      {/* Ports */}
      <div className="absolute -top-1.5 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-white border-2 border-gray-300 rounded-full"></div>
      <div className="absolute -bottom-1.5 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-white border-2 border-gray-300 rounded-full hover:bg-blue-500 hover:border-blue-500 transition-colors"></div>
    </div>
  );
};

// --- 子组件: 策略画布 (Strategy Canvas) ---
const StrategyCanvas = () => {
  return (
    <div className="relative w-full h-full bg-slate-50 overflow-hidden font-sans">
      {/* Grid Background */}
      <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(#94a3b8 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
      
      {/* Connecting Lines (SVG) */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {/* Line 1: Trigger -> Condition */}
        <path d="M 180 120 C 180 180, 180 180, 180 200" fill="none" stroke="#cbd5e1" strokeWidth="2" />
        {/* Line 2: Condition -> Action A (Left) */}
        <path d="M 180 280 C 180 320, 80 320, 80 360" fill="none" stroke="#cbd5e1" strokeWidth="2" />
        {/* Line 3: Condition -> Action B (Right) */}
        <path d="M 180 280 C 180 320, 480 320, 480 360" fill="none" stroke="#cbd5e1" strokeWidth="2" />
         {/* Line 4: Action A -> End */}
         <path d="M 80 440 C 80 480, 180 480, 180 520" fill="none" stroke="#cbd5e1" strokeWidth="2" />
         {/* Line 5: Action B -> End */}
         <path d="M 480 440 C 480 480, 180 480, 180 520" fill="none" stroke="#cbd5e1" strokeWidth="2" />
      </svg>

      {/* Nodes */}
      <div className="absolute inset-0 p-10">
        <Node type="trigger" title="线索入库 (Trigger)" subtitle="来源 = 抖音 OR 百度" x={50} y={40} />
        
        <Node type="condition" title="意向分判断 (Split)" subtitle="AI 评分 > 80 (S/A量)" x={50} y={200} />
        
        {/* Branch A */}
        <Node type="action" title="[高意向] 极速跟进" subtitle="执行：加微三连击 (0延迟)" x={-50} y={360} active={true} />
        
        {/* Branch B */}
        <Node type="action" title="[低意向] 培育池清洗" subtitle="执行：加入长期培育池 + 发送干货" x={350} y={360} />

        <Node type="end" title="流程结束" subtitle="等待下一次触发" x={50} y={520} />
      </div>

      {/* Toolbar */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 bg-white p-2 rounded-lg shadow-md border border-gray-200">
        <button className="p-2 hover:bg-gray-100 rounded text-gray-600" title="Add Trigger"><Zap size={18} /></button>
        <button className="p-2 hover:bg-gray-100 rounded text-gray-600" title="Add Condition"><GitBranch size={18} /></button>
        <button className="p-2 hover:bg-gray-100 rounded text-gray-600" title="Add Action"><MessageSquare size={18} /></button>
        <div className="h-px bg-gray-200 my-1"></div>
        <button className="p-2 hover:bg-gray-100 rounded text-gray-600" title="Zoom In"><Plus size={18} /></button>
      </div>

      {/* Status Bar */}
      <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur px-3 py-1.5 rounded-full border border-gray-200 text-xs text-green-600 flex items-center gap-1.5 shadow-sm">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        策略运行中 (Last Run: 2 mins ago)
      </div>
    </div>
  );
};

// --- 子组件: 知识库表格 (Knowledge Base) ---
const KnowledgeBase = () => {
  const [data, setData] = useState([
    { key: "COURSE_PRICE_STD", value: "2980", desc: "标准课单价（元）", update: "2025-10-24 10:00 by Admin" },
    { key: "COURSE_PRICE_VIP", value: "3580", desc: "VIP 1v1 课单价（元）", update: "2025-10-20 14:30 by Admin" },
    { key: "CAMP_START_DATE", value: "2025-10-30", desc: "第6期开营日期", update: "2025-10-24 09:00 by Ops" },
    { key: "LIVE_URL_DAY1", value: "https://live.douyin.com/123", desc: "Day1 直播间链接", update: "2025-10-24 11:00 by Ops" },
    { key: "REFUND_POLICY", value: "7天无理由退款，观看进度<5%", desc: "退款政策简述", update: "2025-09-01 10:00 by Legal" },
  ]);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
          <div>
            <h3 className="font-bold text-gray-800">原子事实知识库 (Atomic Facts)</h3>
            <p className="text-xs text-gray-500 mt-1">修改此处数值，所有 AI 销售的话术将自动同步更新。</p>
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-1.5 bg-white border border-gray-300 rounded text-sm text-gray-600 hover:bg-gray-50 flex items-center gap-1">
              <Plus size={14} /> 新增变量
            </button>
            <button className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 flex items-center gap-1">
              <Save size={14} /> 发布更新
            </button>
          </div>
        </div>
        
        <table className="w-full text-sm text-left">
          <thead className="bg-gray-50 text-gray-500 font-medium">
            <tr>
              <th className="px-6 py-3 w-[200px]">变量 Key</th>
              <th className="px-6 py-3 w-[300px]">当前值 (Value)</th>
              <th className="px-6 py-3">描述/用途</th>
              <th className="px-6 py-3 text-right">最后更新</th>
              <th className="px-6 py-3 w-[80px]"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.map((item, idx) => (
              <tr key={idx} className="hover:bg-blue-50/30 transition-colors group">
                <td className="px-6 py-4 font-mono text-xs text-blue-600 font-medium">{item.key}</td>
                <td className="px-6 py-4">
                  <input 
                    type="text" 
                    defaultValue={item.value} 
                    className="w-full px-2 py-1 bg-gray-50 border border-transparent hover:border-gray-300 focus:bg-white focus:border-blue-500 rounded transition-all outline-none text-gray-800 font-medium"
                  />
                </td>
                <td className="px-6 py-4 text-gray-500">{item.desc}</td>
                <td className="px-6 py-4 text-right text-xs text-gray-400">{item.update}</td>
                <td className="px-6 py-4 text-right">
                  <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-1 hover:bg-gray-200 rounded text-gray-500"><Trash2 size={14} /></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        <div className="px-6 py-3 bg-yellow-50 text-yellow-800 text-xs border-t border-yellow-100 flex items-center gap-2">
           <AlertCircle size={14} />
           <span>注意：价格类变量的修改需要【财务总监】审批后才会对 C 端用户生效。</span>
        </div>
      </div>
    </div>
  );
};

// --- 子组件: 三连击配置 (Triple Hit Config) ---
const TripleHitConfig = () => {
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-xl font-bold text-gray-800">多渠道加微三连击配置</h2>
          <p className="text-sm text-gray-500 mt-1">SOP 名称: <span className="font-mono text-gray-700 bg-gray-100 px-1 rounded">SOP_New_Lead_V3</span></p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">总开关</span>
          <ToggleRight className="text-green-500 cursor-pointer" size={32} />
        </div>
      </div>

      {/* Timeline Visual */}
      <div className="flex items-center justify-between px-10 py-4 bg-white rounded-lg border border-gray-200 shadow-sm relative overflow-hidden">
         <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-100 -z-10"></div>
         <div className="flex flex-col items-center gap-2 z-10">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-xs font-bold text-blue-600">0 min</span>
         </div>
         <div className="flex flex-col items-center gap-2 z-10">
            <div className="w-3 h-3 bg-gray-300 rounded-full"></div>
            <span className="text-xs text-gray-400">5 min</span>
         </div>
         <div className="flex flex-col items-center gap-2 z-10">
            <div className="w-3 h-3 bg-gray-300 rounded-full"></div>
            <span className="text-xs text-gray-400">30 min</span>
         </div>
      </div>

      {/* Step 1: WeCom */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-3 bg-blue-50/50 border-b border-gray-100 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-green-500 rounded text-white"><MessageSquare size={16} /></div>
            <h3 className="font-bold text-gray-800">第一击：企业微信加好友 (自动)</h3>
          </div>
          <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded font-medium">即时触发</span>
        </div>
        <div className="p-6 grid grid-cols-2 gap-6">
          <div className="col-span-1 space-y-4">
             <div>
               <label className="block text-xs font-bold text-gray-500 mb-1">执行账号规则</label>
               <select className="w-full bg-gray-50 border border-gray-200 rounded px-3 py-2 text-sm">
                 <option>轮询分配 (Round Robin)</option>
                 <option>按地理位置分配</option>
               </select>
             </div>
             <div>
               <label className="block text-xs font-bold text-gray-500 mb-1">每日上限</label>
               <input type="number" defaultValue={50} className="w-full bg-gray-50 border border-gray-200 rounded px-3 py-2 text-sm" />
               <p className="text-[10px] text-gray-400 mt-1">防止账号风控，建议 &lt; 80</p>
             </div>
          </div>
          <div className="col-span-1">
            <label className="block text-xs font-bold text-gray-500 mb-1">好友申请文案 (验证语)</label>
            <textarea 
              className="w-full h-32 bg-gray-50 border border-gray-200 rounded p-3 text-sm resize-none"
              defaultValue="{昵称}你好，我是子君AI的课程顾问，给您发送一下今天的直播课件和资料，请通过一下~"
            ></textarea>
            <div className="flex justify-between mt-1">
              <span className="text-[10px] text-gray-400">支持变量: {"{昵称}"}, {"{来源}"}</span>
              <span className="text-[10px] text-gray-400">42/50 字</span>
            </div>
          </div>
        </div>
      </div>

      {/* Step 2: SMS */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden opacity-90">
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-100 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-blue-500 rounded text-white"><Smartphone size={16} /></div>
            <h3 className="font-bold text-gray-800">第二击：短信触达</h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">延迟时间:</span>
            <select className="bg-white border border-gray-200 text-xs rounded px-2 py-1">
              <option>5 分钟</option>
              <option>10 分钟</option>
            </select>
          </div>
        </div>
        <div className="p-6">
           <label className="block text-xs font-bold text-gray-500 mb-1">短信模板 (已备案)</label>
           <div className="flex gap-4">
              <div className="flex-1 p-3 border-2 border-blue-500 bg-blue-50 rounded cursor-pointer relative">
                <div className="text-xs font-bold text-blue-700 mb-1">模板 A: 利益诱导类</div>
                <div className="text-xs text-gray-600">【子君AI】您预留的《AI副业实战手册》已生成，请通过一下微信，以便为您发送下载链接。回T退订</div>
                <div className="absolute top-2 right-2 text-blue-500"><Check size={16} /></div>
              </div>
              <div className="flex-1 p-3 border border-gray-200 rounded cursor-pointer hover:bg-gray-50">
                <div className="text-xs font-bold text-gray-700 mb-1">模板 B: 服务通知类</div>
                <div className="text-xs text-gray-600">【子君AI】您的直播课席位已保留。助教老师正在添加您的微信，请留意新的好友申请。回T退订</div>
              </div>
           </div>
        </div>
      </div>

      {/* Step 3: Call */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden opacity-90">
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-100 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-purple-500 rounded text-white"><Phone size={16} /></div>
            <h3 className="font-bold text-gray-800">第三击：AI 智能外呼</h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">延迟时间:</span>
            <select className="bg-white border border-gray-200 text-xs rounded px-2 py-1">
              <option>30 分钟</option>
              <option>1 小时</option>
            </select>
          </div>
        </div>
        <div className="p-6 grid grid-cols-2 gap-8">
           <div>
              <label className="block text-xs font-bold text-gray-500 mb-1">外呼策略配置</label>
              <div className="space-y-2 mt-2">
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" defaultChecked className="rounded text-purple-600" />
                  <span>启用 AI 拟人声音 (TTS)</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" defaultChecked className="rounded text-purple-600" />
                  <span>被挂断自动重拨 (最多1次)</span>
                </label>
              </div>
           </div>
           <div>
              <label className="block text-xs font-bold text-gray-500 mb-1">骚扰防护 (免打扰)</label>
              <div className="bg-red-50 p-3 rounded text-xs text-red-800 border border-red-100">
                <p className="font-bold mb-1">禁止外呼时段：</p>
                <p>每日 22:00 - 次日 09:00</p>
                <p>午休 12:00 - 13:30</p>
              </div>
           </div>
        </div>
      </div>

      <div className="flex justify-end pt-4">
        <button className="px-6 py-2 bg-gray-100 text-gray-600 rounded-lg mr-4 font-medium hover:bg-gray-200">取消</button>
        <button className="px-6 py-2 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 font-medium flex items-center gap-2">
          <Save size={18} /> 保存并发布配置
        </button>
      </div>

    </div>
  );
};

// --- 主组件 ---
const ConfigPanel = () => {
  const [activeTab, setActiveTab] = useState('canvas'); // 'canvas', 'knowledge', 'triplehit'

  const renderContent = () => {
    switch(activeTab) {
      case 'canvas': return <StrategyCanvas />;
      case 'knowledge': return <KnowledgeBase />;
      case 'triplehit': return <TripleHitConfig />;
      default: return <StrategyCanvas />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 font-sans flex flex-col">
      {/* 1. Header */}
      <div className="bg-[#1e293b] text-white px-6 py-3 flex justify-between items-center shadow-md shrink-0 h-16">
        <div className="flex items-center gap-3">
          <div className="bg-purple-600 p-1.5 rounded-lg">
            <Settings size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold leading-none">运营配置后台</h1>
            <p className="text-[10px] text-gray-400 mt-1 uppercase tracking-wider">System Operations Console</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm">
           <button className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
             查看操作日志
           </button>
           <div className="flex items-center gap-2">
             <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center font-bold">A</div>
             <span className="text-xs text-gray-300">Admin</span>
           </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* 2. Sidebar */}
        <div className="w-60 bg-white border-r border-gray-200 flex flex-col py-6 shrink-0">
          <div className="px-6 mb-2 text-xs font-bold text-gray-400 uppercase tracking-wider">Core Strategies</div>
          <nav className="space-y-1 px-3">
             <button 
               onClick={() => setActiveTab('canvas')}
               className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'canvas' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`}
             >
               <GitBranch size={18} />
               策略画布 (Canvas)
             </button>
             <button 
               onClick={() => setActiveTab('triplehit')}
               className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'triplehit' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`}
             >
               <Zap size={18} />
               三连击配置 (SOP)
             </button>
          </nav>

          <div className="mt-8 px-6 mb-2 text-xs font-bold text-gray-400 uppercase tracking-wider">Data & Assets</div>
          <nav className="space-y-1 px-3">
             <button 
               onClick={() => setActiveTab('knowledge')}
               className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors ${activeTab === 'knowledge' ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`}
             >
               <Database size={18} />
               知识库 (Facts)
             </button>
             <button className="w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium text-gray-600 hover:bg-gray-50 rounded-lg">
               <MessageSquare size={18} />
               话术模板库
             </button>
          </nav>
        </div>

        {/* 3. Content Area */}
        <div className="flex-1 overflow-y-auto bg-gray-50 relative">
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default ConfigPanel;