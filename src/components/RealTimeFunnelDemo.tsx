import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart, 
  Activity, 
  Users, 
  Filter, 
  ArrowDown, 
  AlertTriangle, 
  RefreshCw, 
  Zap,
  PhoneCall,
  MessageSquare,
  Clock,
  PieChart,
  Target,
  CheckCircle // Added CheckCircle
} from 'lucide-react';

// --- Constants ---
const INITIAL_STATS = {
  added: 1200,    // 加微 (L1)
  entered: 480,   // 到房 (L2) - Initial Low Rate (40%)
  intent: 150,    // 意向 (L3)
  paid: 45        // 成交 (L4)
};

const ALERTS = [
  { id: 1, type: 'critical', msg: '警告：直播间 [到课率] 跌至 38% (阈值 45%)', time: '10:15:00' },
  { id: 2, type: 'warning', msg: '注意：销售组 B [意向转化] 低于平均值', time: '10:12:30' }
];

const SALES_REPS = [
  { name: '王冠军', group: 'A组', added: 400, entered: 200, paid: 25, rate: '6.2%' },
  { name: '李潜力', group: 'A组', added: 350, entered: 160, paid: 15, rate: '4.2%' },
  { name: '张落后', group: 'B组', added: 450, entered: 120, paid: 5, rate: '1.1%' }, // Low performance
];

export default function RealTimeFunnelDemo() {
  // --- State ---
  const [stats, setStats] = useState(INITIAL_STATS);
  const [alerts, setAlerts] = useState(ALERTS);
  const [selectedSlice, setSelectedSlice] = useState('ALL'); // ALL, GROUP_A, GROUP_B
  const [isLive, setIsLive] = useState(true);
  const [drillDownData, setDrillDownData] = useState(null); 
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // --- Simulation Effect ---
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      // Simulate real-time increments
      const newAdded = Math.floor(Math.random() * 3); 
      const newEntered = Math.random() > 0.6 ? 1 : 0;
      const newIntent = Math.random() > 0.8 ? 1 : 0;
      const newPaid = Math.random() > 0.9 ? 1 : 0;

      setStats(prev => ({
        added: prev.added + newAdded,
        entered: prev.entered + newEntered,
        intent: prev.intent + newIntent,
        paid: prev.paid + newPaid
      }));
      setLastUpdate(new Date());
    }, 1500); // Update every 1.5s

    return () => clearInterval(interval);
  }, [isLive]);

  // --- Helpers ---
  const getConversionRate = (curr, prev) => {
    if (prev === 0) return 0;
    return ((curr / prev) * 100).toFixed(1);
  };

  const handleDrillDown = (layer) => {
    if (layer === 'ENTERED') {
      // Simulate fetching "No Show" list
      setDrillDownData({
        title: '未进房学员名单 (No Show)',
        count: stats.added - stats.entered,
        action: '一键补戳 (Call/SMS)',
        actionCompleted: false, // New state for UI
        isProcessing: false,    // New state for UI
        list: [
          { name: '用户_9527', phone: '138****1234', tag: 'S级', status: '未触达' },
          { name: '用户_8866', phone: '139****5678', tag: 'A级', status: '已短信' },
          { name: '用户_1102', phone: '136****9999', tag: 'S级', status: '未触达' },
          { name: '用户_3344', phone: '150****0000', tag: 'B级', status: '未触达' },
        ]
      });
    } else if (layer === 'PAID') {
        setDrillDownData(null); // Reset
    }
  };

  const executeAction = () => {
    // 1. Set Processing State
    setDrillDownData(prev => ({ ...prev, isProcessing: true }));

    // 2. Count target users
    const targetCount = drillDownData.list.filter(u => u.status === '未触达').length;

    // 3. Simulate API Call delay
    setTimeout(() => {
        setDrillDownData(prev => ({
            ...prev,
            isProcessing: false,
            actionCompleted: true,
            actionResult: `成功下发 ${targetCount} 条强提醒任务`,
            list: prev.list.map(u => u.status === '未触达' ? { ...u, status: '已干预' } : u)
        }));
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-center bg-slate-800 p-5 rounded-2xl border border-slate-700 shadow-xl">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Activity className="text-indigo-400" />
              实时销售漏斗 <span className="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded-full uppercase tracking-wide border border-indigo-500/30">Live Monitor</span>
            </h1>
            <p className="text-slate-400 mt-1 text-sm flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              数据源: Webhook Stream • 刷新频率: 1s • 上次更新: {lastUpdate.toLocaleTimeString()}
            </p>
          </div>
          <div className="flex gap-3 mt-4 md:mt-0">
             <div className="flex bg-slate-700 p-1 rounded-lg border border-slate-600">
                {['ALL', 'A组', 'B组'].map(g => (
                  <button 
                    key={g}
                    onClick={() => setSelectedSlice(g)}
                    className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${selectedSlice === g ? 'bg-slate-500 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                  >
                    {g}
                  </button>
                ))}
             </div>
             <button 
                onClick={() => setIsLive(!isLive)}
                className={`px-4 py-2 rounded-lg border text-xs font-bold flex items-center gap-2 transition-all ${isLive ? 'bg-red-500/20 border-red-500/50 text-red-300 hover:bg-red-500/30' : 'bg-green-500/20 border-green-500/50 text-green-300'}`}
             >
               {isLive ? '⏸ 暂停更新' : '▶️ 恢复实时'}
             </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: The Funnel */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* KPI Cards */}
            <div className="grid grid-cols-4 gap-4">
               <KPICard label="今日加微 (L1)" value={stats.added} trend="+12%" color="blue" />
               <KPICard label="直播在线 (L2)" value={stats.entered} trend="-5%" color="purple" isAlert={stats.entered/stats.added < 0.45} />
               <KPICard label="意向咨询 (L3)" value={stats.intent} trend="+8%" color="orange" />
               <KPICard label="今日成交 (L4)" value={stats.paid} trend="+15%" color="green" />
            </div>

            {/* Funnel Visualization */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 relative overflow-hidden">
               <div className="absolute top-0 right-0 p-4 opacity-10">
                 <Filter className="w-32 h-32 text-slate-500" />
               </div>
               
               <h3 className="text-sm font-bold text-slate-400 uppercase mb-6 flex items-center gap-2">
                 <Target className="w-4 h-4" /> 转化链路视图
               </h3>

               <div className="space-y-2 relative z-10">
                 
                 {/* Layer 1: Added */}
                 <FunnelLayer 
                   label="流量接入 (加微)" 
                   count={stats.added} 
                   percent={100} 
                   color="bg-blue-600" 
                   width="100%" 
                 />
                 
                 <ConversionArrow rate={getConversionRate(stats.entered, stats.added)} label="到课率" isLow={stats.entered/stats.added < 0.45} />

                 {/* Layer 2: Entered */}
                 <FunnelLayer 
                   label="直播互动 (进房)" 
                   count={stats.entered} 
                   percent={getConversionRate(stats.entered, stats.added)} 
                   color="bg-purple-600" 
                   width={`${(stats.entered/stats.added)*100}%`}
                   onClick={() => handleDrillDown('ENTERED')}
                   warning={stats.entered/stats.added < 0.45}
                 />

                 <ConversionArrow rate={getConversionRate(stats.intent, stats.entered)} label="互动率" />

                 {/* Layer 3: Intent */}
                 <FunnelLayer 
                   label="商机培育 (意向)" 
                   count={stats.intent} 
                   percent={getConversionRate(stats.intent, stats.entered)} 
                   color="bg-orange-600" 
                   width={`${(stats.intent/stats.added)*100}%`} 
                 />

                 <ConversionArrow rate={getConversionRate(stats.paid, stats.intent)} label="成交率" />

                 {/* Layer 4: Paid */}
                 <FunnelLayer 
                   label="最终转化 (成交)" 
                   count={stats.paid} 
                   percent={getConversionRate(stats.paid, stats.intent)} 
                   color="bg-green-600" 
                   width={`${(stats.paid/stats.added)*100}%`} 
                 />

               </div>
            </div>

            {/* Sales Team Ranking */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
               <div className="p-4 border-b border-slate-700 bg-slate-700/30 flex justify-between items-center">
                 <h3 className="font-bold text-slate-300 text-sm">销售团队实时战报</h3>
                 <span className="text-xs text-slate-500">按成交率排序</span>
               </div>
               <table className="w-full text-left text-sm">
                 <thead className="bg-slate-700/50 text-xs text-slate-400 uppercase">
                   <tr>
                     <th className="px-4 py-3">销售姓名</th>
                     <th className="px-4 py-3">组别</th>
                     <th className="px-4 py-3 text-right">加微数</th>
                     <th className="px-4 py-3 text-right">到课数</th>
                     <th className="px-4 py-3 text-right">成交数</th>
                     <th className="px-4 py-3 text-right">转化率</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-700">
                   {SALES_REPS.map((rep, idx) => (
                     <tr key={idx} className="hover:bg-slate-700/30 transition-colors">
                       <td className="px-4 py-3 font-bold">{rep.name}</td>
                       <td className="px-4 py-3 text-slate-400">{rep.group}</td>
                       <td className="px-4 py-3 text-right font-mono">{rep.added}</td>
                       <td className="px-4 py-3 text-right font-mono">{rep.entered}</td>
                       <td className="px-4 py-3 text-right font-mono text-green-400">{rep.paid}</td>
                       <td className={`px-4 py-3 text-right font-bold ${parseFloat(rep.rate) < 2 ? 'text-red-400' : 'text-indigo-400'}`}>
                         {rep.rate}
                       </td>
                     </tr>
                   ))}
                 </tbody>
               </table>
            </div>

          </div>

          {/* Right: Insights & Action */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Real-time Alerts */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-5 shadow-lg relative overflow-hidden">
               <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
               <h3 className="font-bold text-red-400 flex items-center gap-2 mb-4 animate-pulse">
                 <AlertTriangle className="w-5 h-5" /> 实时异常告警
               </h3>
               <div className="space-y-3">
                 {alerts.map(alert => (
                   <div key={alert.id} className="bg-red-500/10 p-3 rounded-lg border border-red-500/20 text-xs">
                     <div className="flex justify-between text-red-300 mb-1 opacity-70">
                       <span>{alert.type === 'critical' ? 'CRITICAL' : 'WARNING'}</span>
                       <span>{alert.time}</span>
                     </div>
                     <p className="text-red-100 font-bold">{alert.msg}</p>
                   </div>
                 ))}
               </div>
            </div>

            {/* Drill Down Action Panel */}
            <div className={`bg-slate-800 rounded-xl border border-slate-700 p-5 transition-all duration-500 ${drillDownData ? 'translate-x-0 opacity-100' : 'translate-x-10 opacity-50 blur-sm pointer-events-none'}`}>
               {drillDownData ? (
                 <>
                   <div className="flex justify-between items-start mb-4 border-b border-slate-700 pb-4">
                     <div>
                       <h3 className="font-bold text-white flex items-center gap-2">
                         <Filter className="w-4 h-4 text-indigo-400" />
                         下钻分析: {drillDownData.title}
                       </h3>
                       <p className="text-xs text-slate-400 mt-1">共发现 {drillDownData.count} 名异常用户</p>
                     </div>
                     <button onClick={() => setDrillDownData(null)} className="text-slate-500 hover:text-white">✕</button>
                   </div>
                   
                   <div className="space-y-2 mb-6">
                     {drillDownData.list.map((u, i) => (
                       <div key={i} className="flex justify-between items-center bg-slate-900/50 p-2 rounded text-xs border border-slate-700/50">
                         <div className="flex items-center gap-2">
                           <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${u.tag === 'S级' ? 'bg-red-500 text-white' : 'bg-slate-600'}`}>{u.tag}</span>
                           <span className="text-slate-300">{u.name}</span>
                         </div>
                         <span className={`${u.status === '未触达' ? 'text-red-400' : 'text-green-400 font-bold'}`}>{u.status}</span>
                       </div>
                     ))}
                     <div className="text-center text-xs text-slate-500 py-1">...等 {drillDownData.count - 4} 人</div>
                   </div>

                   {drillDownData.actionCompleted ? (
                       <div className="w-full py-3 bg-green-500/20 text-green-400 border border-green-500/50 rounded-lg font-bold flex items-center justify-center gap-2 animate-in zoom-in">
                           <CheckCircle className="w-4 h-4" /> {drillDownData.actionResult}
                       </div>
                   ) : (
                       <button 
                         onClick={executeAction}
                         disabled={drillDownData.isProcessing}
                         className={`w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all shadow-lg ${drillDownData.isProcessing ? 'bg-slate-600 cursor-wait' : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/50'}`}
                       >
                         {drillDownData.isProcessing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />} 
                         {drillDownData.isProcessing ? '正在执行...' : `执行干预：${drillDownData.action}`}
                       </button>
                   )}
                 </>
               ) : (
                 <div className="h-40 flex flex-col items-center justify-center text-slate-600 text-sm">
                   <Users className="w-8 h-8 mb-2 opacity-20" />
                   <p>点击左侧漏斗层级可下钻查看详情</p>
                 </div>
               )}
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}

// --- Sub Components ---

function KPICard({ label, value, trend, color, isAlert }) {
  const colors = {
    blue: 'text-blue-400 border-l-blue-500',
    purple: 'text-purple-400 border-l-purple-500',
    orange: 'text-orange-400 border-l-orange-500',
    green: 'text-green-400 border-l-green-500',
  };

  return (
    <div className={`bg-slate-800 p-4 rounded-xl border border-slate-700 border-l-4 ${colors[color]} ${isAlert ? 'ring-2 ring-red-500 bg-red-900/10' : ''}`}>
      <div className="text-xs text-slate-400 uppercase font-bold mb-1">{label}</div>
      <div className="flex justify-between items-end">
        <div className="text-2xl font-black text-white font-mono">{value.toLocaleString()}</div>
        <div className={`text-xs font-bold ${trend.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
          {trend}
        </div>
      </div>
    </div>
  );
}

function FunnelLayer({ label, count, percent, color, width, onClick, warning }) {
  return (
    <div 
      onClick={onClick}
      className={`relative group cursor-pointer transition-all hover:brightness-110 ${warning ? 'animate-pulse' : ''}`}
    >
      {/* Background Track */}
      <div className="w-full h-12 bg-slate-900/50 rounded-lg absolute top-0 left-0"></div>
      
      {/* Bar */}
      <div 
        className={`h-12 rounded-lg flex items-center justify-between px-4 transition-all duration-1000 ease-out ${color} ${warning ? 'ring-2 ring-red-500 ring-offset-2 ring-offset-slate-800' : ''}`}
        style={{ width: width, minWidth: '140px' }}
      >
        <span className="font-bold text-white text-sm whitespace-nowrap">{label}</span>
      </div>

      {/* Label (Outside if bar is small, but for simplicty inside right aligned) */}
      <div className="absolute top-0 right-4 h-full flex items-center gap-4">
         <span className="font-mono font-bold text-white text-lg">{count}</span>
         <span className="text-xs font-mono bg-black/20 px-2 py-1 rounded text-white/80">{percent}%</span>
      </div>

      {/* Warning Icon */}
      {warning && (
        <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-red-500">
          <AlertTriangle className="w-5 h-5" />
        </div>
      )}
    </div>
  );
}

function ConversionArrow({ rate, label, isLow }) {
  return (
    <div className="flex justify-center items-center h-8 relative">
      <div className="h-full w-0.5 bg-slate-700"></div>
      <div className={`absolute bg-slate-800 px-2 py-0.5 rounded text-[10px] font-bold border ${isLow ? 'text-red-400 border-red-500/50' : 'text-slate-400 border-slate-700'}`}>
        {label}: {rate}%
      </div>
    </div>
  );
}