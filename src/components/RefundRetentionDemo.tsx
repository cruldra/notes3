import React, { useState } from 'react';
import { 
  AlertTriangle, 
  ShieldAlert, 
  Gift, 
  CalendarClock, 
  ArrowRightLeft, 
  XCircle, 
  CheckCircle,
  CreditCard,
  History,
  Info
} from 'lucide-react';

// --- Configuration ---
const STRATEGIES = {
  TRANSFER: {
    id: 'TRANSFER',
    title: '🔄 免费转班权益',
    icon: <ArrowRightLeft className="w-12 h-12 text-blue-500 mb-2" />,
    script: '同学请留步！检测到您是因为“时间冲突”申请退款。这期没时间没关系，我们可以帮您【免费调整到下期训练营】（原价需收 200 元手续费）。保留学籍，下个月再学，您看可以吗？',
    benefit: '免手续费转班',
    acceptText: '接受转班 (撤销退款)'
  },
  DEFER: {
    id: 'DEFER',
    title: '⏳ 7天免费延期',
    icon: <CalendarClock className="w-12 h-12 text-orange-500 mb-2" />,
    script: '别急着走！我知道最近工作忙可能跟不上进度。我们特别为您申请了【7 天免费延期权限】，您可以按照自己的节奏慢慢看回放，不用担心课程过期！',
    benefit: '课程有效期 +7天',
    acceptText: '接受延期 (撤销退款)'
  },
  COUPON: {
    id: 'COUPON',
    title: '🎁 高阶课优惠券',
    icon: <Gift className="w-12 h-12 text-red-500 mb-2" />,
    script: '是对价格不满意吗？先别退！作为我们的种子用户，现在撤销申请，立即赠送您一张【200元高阶课通用券】 + 【AI 实战资料包 (价值 99 元)】！',
    benefit: '¥200 优惠券 + 资料包',
    acceptText: '领取福利 (撤销退款)'
  }
};

const USERS = [
  { id: 'busy', name: '李忙碌 (没时间)', reason: '工作太忙/进度跟不上', strategy: 'DEFER', isRisk: false },
  { id: 'price', name: '王嫌贵 (性价比)', reason: '觉得价格贵/不划算', strategy: 'COUPON', isRisk: false },
  { id: 'conflict', name: '张冲突 (时间不合)', reason: '本期时间不合适', strategy: 'TRANSFER', isRisk: false },
  { id: 'risk', name: '赵黑产 (恶意退款)', reason: '其他原因', strategy: 'NONE', isRisk: true, riskDesc: '30天内退款次数 ≥ 2' }
];

export default function RefundRetentionDemo() {
  // --- State ---
  const [currentUser, setCurrentUser] = useState(USERS[0]);
  const [orderStatus, setOrderStatus] = useState('PAID'); // PAID, RETENTION_POPUP, REFUNDED, SAVED, BLOCKED
  const [logs, setLogs] = useState([]);

  // --- Actions ---
  const addLog = (msg, type = 'info') => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg, type }, ...prev]);
  };

  const handleApplyRefund = () => {
    addLog(`用户 [${currentUser.name}] 点击申请退款...`, 'action');
    
    // 1. Risk Control Check (Feature 9.3)
    if (currentUser.isRisk) {
      addLog(`🚨 风控触发: 检测到恶意退款行为 (${currentUser.riskDesc})`, 'error');
      setOrderStatus('BLOCKED');
      return;
    }

    // 2. Trigger Retention Strategy (Feature 9.1)
    const strategy = STRATEGIES[currentUser.strategy];
    addLog(`🛡️ 挽留拦截: 命中策略 [${strategy.title}]`, 'warning');
    setOrderStatus('RETENTION_POPUP');
  };

  const handleAcceptRetention = () => {
    const strategy = STRATEGIES[currentUser.strategy];
    addLog(`✅ 挽留成功: 用户接受了 [${strategy.benefit}]`, 'success');
    setOrderStatus('SAVED');
  };

  const handleConfirmRefund = () => {
    addLog(`❌ 挽留失败: 用户坚持退款`, 'error');
    setOrderStatus('REFUNDED');
  };

  const reset = () => {
    setOrderStatus('PAID');
    setLogs([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <ShieldAlert className="text-red-600" />
              退款挽留与风控系统 <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full uppercase tracking-wide">9.1 & 9.3</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              在退款前的“最后一公里”进行价值对冲 • 降低退款率 5-8%
            </p>
          </div>
          <button onClick={reset} className="text-sm text-slate-500 hover:text-indigo-600 flex items-center gap-1">
            <History className="w-4 h-4" /> 重置订单状态
          </button>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Left: User & Order Context */}
          <div className="space-y-6">
            
            {/* User Selector */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-700 mb-4 flex items-center gap-2">
                <Info className="w-4 h-4" /> 选择模拟场景
              </h3>
              <div className="grid grid-cols-1 gap-3">
                {USERS.map(user => (
                  <button
                    key={user.id}
                    onClick={() => { setCurrentUser(user); reset(); }}
                    className={`
                      text-left p-3 rounded-lg border flex justify-between items-center transition-all
                      ${currentUser.id === user.id 
                        ? 'bg-indigo-50 border-indigo-500 ring-1 ring-indigo-500' 
                        : 'bg-white border-slate-200 hover:border-indigo-200'}
                    `}
                  >
                    <div>
                      <div className="font-bold text-sm text-slate-800">{user.name}</div>
                      <div className="text-xs text-slate-500 mt-0.5">退款理由: {user.reason}</div>
                    </div>
                    {user.isRisk && (
                      <span className="text-[10px] bg-red-100 text-red-600 px-2 py-1 rounded font-bold">
                        高风险
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* System Logs */}
            <div className="bg-slate-900 text-slate-300 p-4 rounded-xl h-48 overflow-y-auto font-mono text-xs shadow-inner">
              <div className="border-b border-slate-700 pb-2 mb-2 font-bold text-slate-500">SYSTEM DECISION LOGS</div>
              {logs.length === 0 && <span className="opacity-30 italic">等待操作...</span>}
              {logs.map((log, i) => (
                <div key={i} className={`mb-1.5 ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-green-400' : log.type === 'warning' ? 'text-yellow-400' : 'text-slate-300'}`}>
                  <span className="opacity-50">[{log.time}]</span> {log.msg}
                </div>
              ))}
            </div>
          </div>

          {/* Right: The App Interface (Simulation) */}
          <div className="relative">
             {/* Phone Frame */}
             <div className="bg-white border-8 border-slate-200 rounded-[2.5rem] shadow-2xl overflow-hidden min-h-[600px] relative">
               
               {/* App Header */}
               <div className="bg-indigo-600 text-white p-6 pt-10 text-center">
                 <h2 className="text-lg font-bold">订单详情</h2>
               </div>

               {/* Order Content */}
               <div className="p-6 space-y-6">
                 <div className="flex gap-4 items-start">
                   <div className="w-20 h-20 bg-indigo-100 rounded-lg flex items-center justify-center">
                     <Gift className="w-8 h-8 text-indigo-500" />
                   </div>
                   <div>
                     <h3 className="font-bold text-slate-800">AI Agent 实战训练营 (第10期)</h3>
                     <p className="text-xs text-slate-500 mt-1">包含：4天直播 + 录播回放 + 源码</p>
                     <div className="mt-2 font-mono font-bold text-lg">¥ 399.00</div>
                   </div>
                 </div>

                 <div className="border-t border-slate-100 pt-4 space-y-2 text-sm">
                   <div className="flex justify-between">
                     <span className="text-slate-500">订单状态</span>
                     <span className={`font-bold ${orderStatus === 'PAID' ? 'text-green-600' : orderStatus === 'REFUNDED' ? 'text-slate-400' : orderStatus === 'SAVED' ? 'text-indigo-600' : 'text-red-600'}`}>
                       {orderStatus === 'PAID' ? '已支付' : 
                        orderStatus === 'RETENTION_POPUP' ? '退款处理中...' : 
                        orderStatus === 'SAVED' ? '已恢复 (权益已到账)' :
                        orderStatus === 'REFUNDED' ? '已退款' : '风控冻结'}
                     </span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-slate-500">下单时间</span>
                     <span className="text-slate-800">2023-11-11 10:23:45</span>
                   </div>
                 </div>

                 {/* The "Apply Refund" Button */}
                 {orderStatus === 'PAID' && (
                   <div className="pt-10">
                     <button 
                       onClick={handleApplyRefund}
                       className="w-full py-3 rounded-lg border border-slate-200 text-slate-500 font-bold hover:bg-slate-50 transition-colors"
                     >
                       申请退款
                     </button>
                     <p className="text-xs text-center text-slate-400 mt-3">
                       退款将原路返回支付账户，预计 1-3 个工作日到账
                     </p>
                   </div>
                 )}

                 {/* Result States */}
                 {orderStatus === 'SAVED' && (
                   <div className="bg-green-50 p-4 rounded-xl border border-green-200 text-center animate-in zoom-in duration-300">
                     <CheckCircle className="w-10 h-10 text-green-500 mx-auto mb-2" />
                     <h3 className="font-bold text-green-800">退款申请已撤销</h3>
                     <p className="text-xs text-green-700 mt-1">
                       您的专属权益已发放至账户，请查收！
                     </p>
                   </div>
                 )}

                 {orderStatus === 'REFUNDED' && (
                   <div className="bg-slate-100 p-4 rounded-xl border border-slate-200 text-center text-slate-500">
                     <Info className="w-10 h-10 mx-auto mb-2" />
                     <h3>退款已提交</h3>
                     <p className="text-xs mt-1">系统将尽快处理您的请求</p>
                   </div>
                 )}
                 
                 {orderStatus === 'BLOCKED' && (
                    <div className="bg-red-50 p-4 rounded-xl border border-red-200 text-center animate-in shake duration-300">
                     <ShieldAlert className="w-10 h-10 text-red-600 mx-auto mb-2" />
                     <h3 className="font-bold text-red-800">无法自动退款</h3>
                     <p className="text-xs text-red-700 mt-1">
                       检测到您的账户存在异常退款记录。请联系人工客服进行审核。
                     </p>
                     <button className="mt-3 text-xs bg-white border border-red-200 px-3 py-1 rounded text-red-600">
                       联系人工客服
                     </button>
                   </div>
                 )}
               </div>

               {/* Retention Popup Modal (The Core Feature) */}
               {orderStatus === 'RETENTION_POPUP' && !currentUser.isRisk && (
                 <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-end sm:items-center justify-center p-4 z-20 animate-in fade-in duration-300">
                   <div className="bg-white w-full max-w-sm rounded-2xl p-6 shadow-2xl transform transition-all scale-100">
                     
                     <div className="text-center mb-6">
                       {STRATEGIES[currentUser.strategy].icon}
                       <h3 className="text-xl font-black text-slate-900">
                         {STRATEGIES[currentUser.strategy].title}
                       </h3>
                       <p className="text-sm text-slate-600 mt-3 leading-relaxed text-left bg-slate-50 p-3 rounded-lg border border-slate-100">
                         "{STRATEGIES[currentUser.strategy].script}"
                       </p>
                     </div>

                     <div className="space-y-3">
                       <button 
                         onClick={handleAcceptRetention}
                         className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold shadow-lg shadow-indigo-200 transition-all flex items-center justify-center gap-2 group"
                       >
                         <Gift className="w-4 h-4 group-hover:animate-bounce" />
                         {STRATEGIES[currentUser.strategy].acceptText}
                       </button>
                       <button 
                         onClick={handleConfirmRefund}
                         className="w-full py-3 text-slate-400 text-sm font-medium hover:text-slate-600"
                       >
                         不需要，继续退款
                       </button>
                     </div>

                   </div>
                 </div>
               )}

             </div>
          </div>

        </div>
      </div>
    </div>
  );
}