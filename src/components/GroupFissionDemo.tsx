import React, { useState, useEffect } from 'react';
import { 
  Users, 
  Gift, 
  Share2, 
  UserPlus, 
  DollarSign, 
  PieChart, 
  Activity, 
  Settings, 
  ShoppingBag,
  TrendingUp,
  Award,
  Smartphone
} from 'lucide-react';

// --- Constants ---
const INITIAL_CONFIG = {
  targetCount: 3, // Invite N people
  rewardAmount: 8.88, // Get X yuan
  budget: 1000,
  totalSpent: 0
};

const MOCK_INVITEES = [
  { id: 'u1', name: '微信用户_A', avatar: '🐱', status: 'JOINED', gmv: 0 },
  { id: 'u2', name: '微信用户_B', avatar: '🐶', status: 'JOINED', gmv: 0 },
  { id: 'u3', name: '微信用户_C', avatar: '🦊', status: 'JOINED', gmv: 0 },
];

export default function GroupFissionDemo() {
  // --- State ---
  const [config, setConfig] = useState(INITIAL_CONFIG);
  const [invitees, setInvitees] = useState([]);
  const [isTaskComplete, setIsTaskComplete] = useState(false);
  const [rewardClaimed, setRewardClaimed] = useState(false);
  const [logs, setLogs] = useState([]);
  
  // Stats
  const totalGMV = invitees.reduce((acc, curr) => acc + curr.gmv, 0);
  const currentROI = config.totalSpent > 0 ? (totalGMV / config.totalSpent).toFixed(2) : 0;

  // --- Actions ---
  const addLog = (msg, type = 'info') => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg, type }, ...prev]);
  };

  const simulateNewJoin = () => {
    if (invitees.length >= 10) return; // Limit for demo
    const newUser = {
      id: Date.now(),
      name: `新用户_${Math.floor(Math.random() * 1000)}`,
      avatar: ['🐹', '🐸', '🐼', '🐨', '🐯'][Math.floor(Math.random() * 5)],
      status: 'JOINED',
      gmv: 0
    };
    setInvitees(prev => [...prev, newUser]);
    addLog(`🔗 追踪到新裂变关系：[${newUser.name}] 通过分享链接入群`, 'success');
  };

  const simulatePurchase = (userId) => {
    setInvitees(prev => prev.map(u => {
      if (u.id === userId && u.gmv === 0) {
        addLog(`💰 转化成功：被邀请人 [${u.name}] 购买了课程 (¥399)`, 'money');
        return { ...u, gmv: 399 };
      }
      return u;
    }));
  };

  const claimReward = () => {
    if (!isTaskComplete || rewardClaimed) return;
    setRewardClaimed(true);
    setConfig(prev => ({ ...prev, totalSpent: prev.totalSpent + prev.rewardAmount }));
    addLog(`🧧 红包发放：向邀请人发放 ¥${config.rewardAmount} (微信零钱直达)`, 'reward');
  };

  // Check Task Status
  useEffect(() => {
    if (invitees.length >= config.targetCount && !isTaskComplete) {
      setIsTaskComplete(true);
      addLog(`🎉 裂变任务达标！邀请人数达到 ${config.targetCount} 人`, 'success');
    }
  }, [invitees, config.targetCount, isTaskComplete]);

  // Reset
  const reset = () => {
    setInvitees([]);
    setIsTaskComplete(false);
    setRewardClaimed(false);
    setConfig(prev => ({ ...prev, totalSpent: 0 }));
    setLogs([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Share2 className="text-indigo-600" />
              群红包裂变系统 <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2 Growth</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于社交关系链的指数级增长 • 实时 ROI 监控
            </p>
          </div>
          <button onClick={reset} className="text-sm text-slate-500 hover:text-indigo-600 flex items-center gap-1">
            <Activity className="w-4 h-4" /> 重置演示
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[800px] lg:h-[700px]">
          
          {/* Left: Config Panel */}
          <div className="lg:col-span-3 flex flex-col gap-4 h-full">
            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 h-full">
              <h2 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                <Settings className="w-4 h-4" /> 活动规则配置
              </h2>
              
              <div className="space-y-6">
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-2">
                    裂变门槛 (邀请人数)
                  </label>
                  <input 
                    type="range" min="1" max="10" step="1"
                    value={config.targetCount}
                    onChange={(e) => setConfig(prev => ({...prev, targetCount: Number(e.target.value)}))}
                    className="w-full accent-indigo-600 cursor-pointer"
                    disabled={invitees.length > 0}
                  />
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>1人</span>
                    <span className="font-bold text-indigo-600 text-lg">{config.targetCount} 人</span>
                    <span>10人</span>
                  </div>
                </div>

                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-2">
                    单个红包奖励 (元)
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">¥</span>
                    <input 
                      type="number"
                      value={config.rewardAmount}
                      onChange={(e) => setConfig(prev => ({...prev, rewardAmount: Number(e.target.value)}))}
                      className="w-full pl-7 pr-3 py-2 border border-slate-200 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                      disabled={invitees.length > 0}
                    />
                  </div>
                </div>

                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100 text-xs space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-500">活动总预算:</span>
                    <span className="font-mono">¥{config.budget}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">已消耗:</span>
                    <span className="font-mono text-red-500 font-bold">-¥{config.totalSpent.toFixed(2)}</span>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-100">
                  <p className="text-xs text-slate-400 leading-relaxed">
                    💡 提示：修改规则需在活动开始前。为了演示方便，您可以随时调整，但真实系统中会锁定规则。
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Middle: User Phone Interface */}
          <div className="lg:col-span-4 h-full flex justify-center">
             <div className="w-[340px] bg-white border-8 border-slate-200 rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col relative">
               
               {/* App Header */}
               <div className="bg-[#ededed] px-4 py-3 flex items-center justify-between z-10">
                 <span className="font-medium text-sm">微信 (WeChat)</span>
                 <div className="flex gap-1">
                   <div className="w-1 h-1 rounded-full bg-slate-800"></div>
                   <div className="w-1 h-1 rounded-full bg-slate-800"></div>
                 </div>
               </div>

               {/* Activity Page */}
               <div className="flex-1 bg-red-600 relative flex flex-col overflow-y-auto scrollbar-hide">
                 
                 {/* Top Visual */}
                 <div className="pt-8 pb-4 text-center px-6 relative">
                   <div className="absolute top-0 left-0 w-full h-full bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-20"></div>
                   <h2 className="text-yellow-200 text-2xl font-black italic tracking-wider drop-shadow-md relative z-10">
                     邀请好友领红包
                   </h2>
                   <p className="text-white/80 text-xs mt-2 relative z-10">
                     限时活动 · 真实提现 · 秒到账
                   </p>
                 </div>

                 {/* Main Card */}
                 <div className="mx-4 bg-white rounded-xl p-5 shadow-lg relative z-10 mb-4">
                   <div className="text-center mb-4">
                     <p className="text-sm text-slate-500">已邀请好友</p>
                     <div className="text-4xl font-black text-red-500 mt-1 font-mono">
                       {invitees.length}<span className="text-sm text-slate-400 font-normal ml-1">/ {config.targetCount}</span>
                     </div>
                   </div>

                   {/* Progress Bar */}
                   <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden mb-6">
                     <div 
                       className="h-full bg-gradient-to-r from-yellow-400 to-red-500 transition-all duration-500" 
                       style={{ width: `${Math.min((invitees.length / config.targetCount) * 100, 100)}%` }}
                     ></div>
                   </div>

                   {/* Action Button */}
                   {rewardClaimed ? (
                     <button className="w-full py-3 bg-slate-100 text-slate-400 rounded-full font-bold cursor-not-allowed">
                       ✅ 红包已领取
                     </button>
                   ) : isTaskComplete ? (
                     <button 
                       onClick={claimReward}
                       className="w-full py-3 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-full font-bold shadow-lg shadow-orange-200 animate-pulse active:scale-95 transition-transform"
                     >
                       🎁 立即拆开 ¥{config.rewardAmount}
                     </button>
                   ) : (
                     <button className="w-full py-3 bg-red-500 text-white rounded-full font-bold shadow-lg shadow-red-200 flex items-center justify-center gap-2">
                       <Share2 className="w-4 h-4" /> 转发到群聊
                     </button>
                   )}
                   
                   {!isTaskComplete && (
                     <p className="text-xs text-center text-slate-400 mt-3">
                       还差 {config.targetCount - invitees.length} 人即可解锁红包
                     </p>
                   )}
                 </div>

                 {/* Invitees List */}
                 <div className="mx-4 bg-white/90 backdrop-blur rounded-xl p-4 shadow-lg flex-1">
                   <h3 className="text-xs font-bold text-slate-500 mb-3 uppercase">邀请记录</h3>
                   <div className="space-y-3">
                     {invitees.length === 0 && <div className="text-center text-xs text-slate-400 py-4">暂无好友加入，快去分享吧~</div>}
                     {invitees.map(user => (
                       <div key={user.id} className="flex items-center justify-between animate-in slide-in-from-bottom-2">
                         <div className="flex items-center gap-2">
                           <div className="w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center text-sm">{user.avatar}</div>
                           <div className="text-sm font-bold text-slate-700">{user.name}</div>
                         </div>
                         <span className="text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full">已加入</span>
                       </div>
                     ))}
                   </div>
                 </div>

               </div>
             </div>
          </div>

          {/* Right: Simulation & ROI Dashboard */}
          <div className="lg:col-span-5 flex flex-col gap-4 h-full">
            
            {/* Simulation Controls */}
            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
              <h2 className="font-bold text-slate-700 mb-4 flex items-center gap-2">
                <Users className="w-4 h-4" /> 群友行为模拟器
              </h2>
              <div className="grid grid-cols-2 gap-3">
                <button 
                  onClick={simulateNewJoin}
                  className="p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded-lg text-sm font-bold hover:bg-blue-100 transition-colors flex flex-col items-center gap-1"
                >
                  <UserPlus className="w-5 h-5" />
                  模拟新用户进群
                </button>
                <div className="space-y-2">
                   {invitees.slice(0, 3).map(u => (
                     <button
                       key={u.id}
                       onClick={() => simulatePurchase(u.id)}
                       disabled={u.gmv > 0}
                       className={`w-full text-xs py-1.5 px-2 rounded border transition-all ${
                         u.gmv > 0 
                           ? 'bg-green-100 text-green-700 border-green-200' 
                           : 'bg-white border-slate-200 text-slate-500 hover:border-green-400 hover:text-green-600'
                       }`}
                     >
                       {u.gmv > 0 ? '✅ 已转化' : `让 ${u.name} 买课`}
                     </button>
                   ))}
                   {invitees.length === 0 && <div className="text-xs text-slate-400 text-center py-2">请先模拟用户进群...</div>}
                </div>
              </div>
            </div>

            {/* ROI Dashboard */}
            <div className="bg-slate-900 text-white p-6 rounded-xl shadow-lg border border-slate-800 flex-1 flex flex-col relative overflow-hidden">
               {/* Background Glow */}
               <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/20 blur-3xl rounded-full pointer-events-none"></div>

               <div className="flex justify-between items-start mb-6 z-10">
                 <div>
                   <h2 className="font-bold flex items-center gap-2">
                     <PieChart className="w-5 h-5 text-indigo-400" />
                     ROI 实时监控
                   </h2>
                   <p className="text-xs text-slate-400 mt-1">数据源: Coze 归因追踪</p>
                 </div>
                 <div className="text-right">
                   <div className="text-3xl font-black text-green-400 font-mono">{currentROI}</div>
                   <div className="text-xs text-slate-500 uppercase font-bold">ROI Ratio</div>
                 </div>
               </div>

               <div className="grid grid-cols-2 gap-4 mb-6 z-10">
                 <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                   <div className="text-xs text-slate-400 mb-1">总投入 (Cost)</div>
                   <div className="text-xl font-bold text-white">¥{config.totalSpent.toFixed(2)}</div>
                 </div>
                 <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                   <div className="text-xs text-slate-400 mb-1">总产出 (GMV)</div>
                   <div className="text-xl font-bold text-white">¥{totalGMV.toFixed(2)}</div>
                 </div>
               </div>

               {/* System Logs */}
               <div className="flex-1 bg-black/30 rounded-lg p-3 overflow-y-auto font-mono text-[10px] space-y-1.5 border border-white/10 z-10">
                 {logs.length === 0 && <span className="text-slate-600 italic">系统日志等待中...</span>}
                 {logs.map((log, i) => (
                   <div key={i} className={`flex gap-2 ${
                     log.type === 'reward' ? 'text-red-400' : 
                     log.type === 'money' ? 'text-green-400' : 
                     log.type === 'success' ? 'text-blue-300' : 'text-slate-400'
                   }`}>
                     <span className="opacity-50">[{log.time}]</span>
                     <span>{log.msg}</span>
                   </div>
                 ))}
               </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}