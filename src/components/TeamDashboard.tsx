import React, { useState } from 'react';
import { 
  TrendingUp, 
  Users, 
  Target, 
  AlertTriangle, 
  Bell, 
  ArrowRight, 
  MoreHorizontal, 
  LayoutDashboard,
  Zap,
  CheckCircle2,
  XCircle,
  BarChart3,
  Search,
  Filter,
  ArrowDown
} from 'lucide-react';

// --- 组件: 漏斗图 (Funnel Chart) ---
const FunnelChart = () => {
  // 模拟数据
  const stages = [
    { label: "加微入库", value: 128, percent: "100%", status: "normal" },
    { label: "进直播间", value: 96, percent: "75%", status: "normal" },
    { label: "申请卡片", value: 42, percent: "32%", status: "critical", drop: "-43% 流失异常" }, // 红色报警
    { label: "成交转化", value: 35, percent: "27%", status: "normal" }
  ];

  return (
    <div className="flex flex-col gap-2 py-4">
      {stages.map((stage, idx) => (
        <div key={idx} className="relative flex items-center group">
          {/* Label */}
          <div className="w-24 text-right text-xs font-medium text-gray-500 mr-4">{stage.label}</div>
          
          {/* Bar / Shape */}
          <div className="flex-1 h-12 relative">
             {/* Background Trend Line (Visual) */}
             <div 
               className={`h-full rounded-r-lg flex items-center px-4 transition-all duration-500 ${
                 stage.status === 'critical' 
                   ? 'bg-red-50 border-l-4 border-red-500 w-[40%]' 
                   : 'bg-blue-50 border-l-4 border-blue-500'
               }`}
               style={{ width: stage.percent }} // Dynamic Width
             >
                <span className={`font-bold text-lg ${stage.status === 'critical' ? 'text-red-700' : 'text-blue-700'}`}>
                  {stage.value}
                </span>
                <span className="ml-2 text-xs text-gray-400">人</span>
             </div>
          </div>

          {/* Right Info */}
          <div className="w-32 pl-4">
             <div className="text-sm font-bold text-gray-700">{stage.percent}</div>
             {stage.status === 'critical' && (
               <div className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded flex items-center gap-1 w-fit animate-pulse">
                 <AlertTriangle size={10} />
                 {stage.drop}
               </div>
             )}
          </div>
          
          {/* Arrow Connector */}
          {idx < stages.length - 1 && (
            <div className="absolute left-[7.5rem] -bottom-3 z-10 text-gray-300">
               <ArrowDown size={14} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// --- 组件: 销售排行榜行 (Sales Row) ---
const SalesRow = ({ rank, name, avatar, amount, followRate, isRisk, onIntervene }) => {
  return (
    <div className="flex items-center p-4 hover:bg-gray-50 border-b border-gray-100 transition-colors">
      <div className={`w-8 h-8 flex items-center justify-center rounded-full font-bold text-sm mr-4 ${
         rank === 1 ? 'bg-yellow-100 text-yellow-700' : 
         rank === 2 ? 'bg-gray-200 text-gray-700' : 
         rank === 3 ? 'bg-orange-100 text-orange-700' : 'text-gray-400'
      }`}>
        {rank}
      </div>
      
      <div className="flex items-center gap-3 flex-1">
        <img src={avatar} alt={name} className="w-10 h-10 rounded-full bg-gray-200" />
        <div>
          <div className="font-bold text-gray-800 text-sm">{name}</div>
          <div className="text-xs text-gray-400">成交 {amount}元</div>
        </div>
      </div>

      <div className="w-32 text-right mr-8">
        <div className="text-[10px] text-gray-400 mb-1">S量跟进率</div>
        <div className={`flex items-center justify-end gap-1 font-bold ${isRisk ? 'text-red-600' : 'text-green-600'}`}>
           {isRisk && <AlertTriangle size={14} />}
           {followRate}%
        </div>
        {isRisk && <div className="text-[10px] text-red-400 scale-90 origin-right">低于阈值 60%</div>}
      </div>

      <button className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded">
        <MoreHorizontal size={18} />
      </button>
    </div>
  );
};

// --- 主组件 ---
const TeamDashboard = () => {
  const [alerts, setAlerts] = useState([
    { id: 1, type: 'danger', user: '销售-小张', msg: '有 3 个 S 级客户超过 24h 未联系', time: '10分钟前', status: 'pending' },
    { id: 2, type: 'warning', user: '销售-李四', msg: 'SOP 任务堆积 15 条', time: '30分钟前', status: 'pending' },
    { id: 3, type: 'info', user: '系统', msg: '第6期训练营直播转化率低于预期', time: '1小时前', status: 'done' },
  ]);

  const handleIntervention = (id) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, status: 'done' } : a));
  };

  return (
    <div className="min-h-screen bg-slate-50 text-gray-800 font-sans flex flex-col">
      
      {/* 1. 顶部导航 (Header) */}
      <div className="bg-[#1e293b] text-white px-6 py-4 flex justify-between items-center shadow-md shrink-0">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <LayoutDashboard size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold leading-none">主管监控看板</h1>
            <p className="text-[10px] text-gray-400 mt-1 uppercase tracking-wider">Team Performance Cockpit</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-gray-400">
           <span className="flex items-center gap-2 px-3 py-1 bg-slate-800 rounded-full">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              数据实时更新中
           </span>
           <div className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center font-bold text-white">M</div>
        </div>
      </div>

      {/* 2. 核心指标卡片 (Key Metrics) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6">
        <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between hover:shadow-md transition-shadow cursor-pointer">
           <div>
             <p className="text-xs text-gray-500 font-medium mb-1">本期总学员数 (Total Students)</p>
             <h3 className="text-3xl font-black text-gray-800">128</h3>
             <p className="text-xs text-green-600 flex items-center mt-1"><TrendingUp size={12} className="mr-1"/> 较上期 +12%</p>
           </div>
           <div className="w-12 h-12 bg-blue-50 rounded-full flex items-center justify-center text-blue-600">
             <Users size={24} />
           </div>
        </div>

        <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between hover:shadow-md transition-shadow cursor-pointer">
           <div>
             <p className="text-xs text-gray-500 font-medium mb-1">实时成交数 (Live Deals)</p>
             <h3 className="text-3xl font-black text-gray-800">35</h3>
             <p className="text-xs text-green-600 flex items-center mt-1"><TrendingUp size={12} className="mr-1"/> 转化进度正常</p>
           </div>
           <div className="w-12 h-12 bg-purple-50 rounded-full flex items-center justify-center text-purple-600">
             <Zap size={24} />
           </div>
        </div>

        <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between hover:shadow-md transition-shadow cursor-pointer border-l-4 border-l-yellow-400">
           <div>
             <p className="text-xs text-gray-500 font-medium mb-1">预估转化率 (Forecast ROI)</p>
             <h3 className="text-3xl font-black text-gray-800">27.3%</h3>
             <p className="text-xs text-yellow-600 flex items-center mt-1"><AlertTriangle size={12} className="mr-1"/> 略低于目标 30%</p>
           </div>
           <div className="w-12 h-12 bg-yellow-50 rounded-full flex items-center justify-center text-yellow-600">
             <Target size={24} />
           </div>
        </div>
      </div>

      {/* 3. 主内容区 (Main Grid) */}
      <div className="flex-1 px-6 pb-8 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0 overflow-y-auto">
        
        {/* 左侧：漏斗 + 排行榜 (Left Column) */}
        <div className="lg:col-span-8 flex flex-col gap-6">
           
           {/* 实时漏斗 (Live Funnel) */}
           <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex justify-between items-center mb-4">
                 <h3 className="font-bold text-gray-800 flex items-center gap-2">
                   <Filter size={18} className="text-blue-500" /> 全局转化漏斗
                 </h3>
                 <button className="text-xs text-blue-600 hover:underline">查看详情 &gt;</button>
              </div>
              <FunnelChart />
           </div>

           {/* 团队排行榜 (Team Leaderboard) */}
           <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex-1 flex flex-col">
              <div className="p-6 border-b border-gray-100 flex justify-between items-center">
                 <h3 className="font-bold text-gray-800 flex items-center gap-2">
                   <BarChart3 size={18} className="text-blue-500" /> 销售团队业绩榜
                 </h3>
                 <div className="flex gap-2">
                    <button className="px-3 py-1 bg-gray-100 rounded text-xs text-gray-600">按成交额</button>
                    <button className="px-3 py-1 bg-white border rounded text-xs text-gray-400">按跟进率</button>
                 </div>
              </div>
              
              <div className="flex-1 overflow-y-auto min-h-[300px]">
                 <SalesRow 
                    rank={1} name="王金牌" avatar="https://api.dicebear.com/7.x/avataaars/svg?seed=Wang"
                    amount="58,900" followRate={95} isRisk={false}
                 />
                 <SalesRow 
                    rank={2} name="李优秀" avatar="https://api.dicebear.com/7.x/avataaars/svg?seed=Li"
                    amount="42,000" followRate={88} isRisk={false}
                 />
                 <SalesRow 
                    rank={3} name="赵进取" avatar="https://api.dicebear.com/7.x/avataaars/svg?seed=Zhao"
                    amount="21,500" followRate={72} isRisk={false}
                 />
                 {/* 异常销售 - 小张 */}
                 <div className="relative">
                   <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-500 animate-pulse"></div>
                   <SalesRow 
                      rank={4} name="小张" avatar="https://api.dicebear.com/7.x/avataaars/svg?seed=Zhang"
                      amount="8,200" followRate={42} isRisk={true}
                   />
                 </div>
                 <SalesRow 
                    rank={5} name="陈新人" avatar="https://api.dicebear.com/7.x/avataaars/svg?seed=Chen"
                    amount="2,980" followRate={90} isRisk={false}
                 />
              </div>
           </div>
        </div>

        {/* 右侧：异常预警 (Right Alerts Panel) */}
        <div className="lg:col-span-4 flex flex-col">
           <div className="bg-white rounded-xl shadow-lg border border-red-100 h-full flex flex-col overflow-hidden">
              <div className="bg-red-50 p-4 border-b border-red-100 flex justify-between items-center">
                 <h3 className="font-bold text-red-800 flex items-center gap-2">
                   <Bell size={18} className="animate-bounce" /> 异常监控预警
                 </h3>
                 <span className="bg-red-200 text-red-800 text-xs px-2 py-0.5 rounded-full font-bold">3</span>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
                 {alerts.map((alert) => (
                    <div key={alert.id} className={`bg-white p-4 rounded-lg shadow-sm border transition-all duration-300 ${alert.status === 'done' ? 'opacity-50 grayscale' : 'border-l-4 border-l-red-500'}`}>
                       <div className="flex justify-between items-start mb-2">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${
                             alert.type === 'danger' ? 'bg-red-100 text-red-600' :
                             alert.type === 'warning' ? 'bg-orange-100 text-orange-600' :
                             'bg-blue-100 text-blue-600'
                          }`}>
                            {alert.type === 'danger' ? '紧急阻断' : '风险提示'}
                          </span>
                          <span className="text-[10px] text-gray-400">{alert.time}</span>
                       </div>
                       
                       <h4 className="font-bold text-gray-800 text-sm mb-1">{alert.user}</h4>
                       <p className="text-xs text-gray-600 mb-3 leading-relaxed">
                         {alert.msg}
                       </p>

                       {alert.status === 'pending' ? (
                         <div className="flex gap-2">
                            <button 
                              onClick={() => handleIntervention(alert.id)}
                              className="flex-1 bg-red-600 hover:bg-red-700 text-white text-xs py-2 rounded shadow-sm font-medium transition-colors flex items-center justify-center gap-1"
                            >
                               <Zap size={12} /> 一键干预
                            </button>
                            <button className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-600 text-xs rounded font-medium">
                               详情
                            </button>
                         </div>
                       ) : (
                         <div className="flex items-center gap-1 text-xs text-green-600 bg-green-50 p-2 rounded">
                            <CheckCircle2 size={12} /> 已完成干预处理
                         </div>
                       )}
                    </div>
                 ))}
                 
                 {/* 历史记录 placeholder */}
                 <div className="text-center text-xs text-gray-400 py-4">
                    - 仅显示最近 24h 异常 -
                 </div>
              </div>
           </div>
        </div>

      </div>
    </div>
  );
};

export default TeamDashboard;