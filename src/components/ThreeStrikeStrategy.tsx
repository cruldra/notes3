import React, { useState, useEffect, useRef } from 'react';
import { 
  Smartphone, 
  MessageSquare, 
  PhoneCall, 
  UserPlus, 
  Clock, 
  CheckCircle, 
  XCircle, 
  Play, 
  RotateCcw, 
  Settings,
  AlertTriangle,
  BellRing,
  ShieldCheck,
  Ban
} from 'lucide-react';

// --- Constants & Config ---
const DEFAULT_CONFIG = {
  strike1_channel: 'SMS', // SMS or WECOM
  strike1_delay: 0, // Immediate
  strike2_channel: 'CALL',
  strike2_delay: 5, // Seconds (simulating 30 mins)
  strike3_channel: 'SMS_FINAL',
  strike3_delay: 10, // Seconds (simulating 2 hours)
};

const STEPS = [
  { id: 1, name: '第一击 (触达)', desc: '线索入库即刻触发', timeLabel: 'T+0' },
  { id: 2, name: '第二击 (催化)', desc: '若未加微，AI 语音强提醒', timeLabel: 'T+30min' },
  { id: 3, name: '第三击 (兜底)', desc: '最后一次尝试或转人工', timeLabel: 'T+2h' },
];

export default function ThreeStrikeStrategy() {
  // --- State ---
  const [status, setStatus] = useState('IDLE'); // IDLE, RUNNING, CONVERTED, COMPLETED
  const [currentStep, setCurrentStep] = useState(0); // 0 (start), 1, 2, 3
  const [logs, setLogs] = useState([]);
  const [timer, setTimer] = useState(0);
  const [userAction, setUserAction] = useState(null); // 'ADDED_FRIEND'
  const [phoneNotifications, setPhoneNotifications] = useState([]);
  
  // Refs for timers
  const intervalRef = useRef(null);

  // --- Logic ---

  const addLog = (msg, type = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { id: Date.now(), time, msg, type }]);
  };

  const addNotification = (type, title, content) => {
    const id = Date.now();
    setPhoneNotifications(prev => [{ id, type, title, content, timestamp: new Date() }, ...prev]);
    // Auto remove notification banner after 4s (but keep in list)
    // In this demo we keep them in a list view on the phone screen
  };

  const startSimulation = () => {
    setStatus('RUNNING');
    setCurrentStep(0);
    setTimer(0);
    setLogs([]);
    setPhoneNotifications([]);
    setUserAction(null);
    addLog('🚀 线索入库：138****0000 (来源: 抖音投放)', 'start');
    
    // Start Timer Loop
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setTimer(t => t + 1);
    }, 1000);
  };

  const stopSimulation = (reason) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus(reason === 'CONVERTED' ? 'CONVERTED' : 'COMPLETED');
  };

  // --- The Core SOP Engine ---
  useEffect(() => {
    if (status !== 'RUNNING') return;

    // --- STRIKE 1: T+0 (Immediate) ---
    if (timer === 0 && currentStep === 0) {
      setCurrentStep(1);
      addLog('⚡️ 触发第一击：发送欢迎短信 + 企微好友申请', 'action');
      addNotification('sms', '【子午线教育】', '同学你好！您的AI实战课资料已生成，请通过一下微信，助教老师在线发送给您。回TD退订');
    }

    // --- STRIKE 2: T+5s (Simulating 30min) ---
    if (timer === DEFAULT_CONFIG.strike2_delay && currentStep === 1) {
      if (userAction === 'ADDED_FRIEND') {
        addLog('🛑 检测到用户已加微，【第二击】自动熔断取消', 'success');
        stopSimulation('CONVERTED');
      } else {
        setCurrentStep(2);
        addLog('📞 触发第二击：用户未加微，发起 AI 语音外呼', 'warning');
        addNotification('call', 'AI 助教老师', '正在来电...');
      }
    }

    // --- STRIKE 3: T+10s (Simulating 2h) ---
    if (timer === DEFAULT_CONFIG.strike3_delay && currentStep === 2) {
      if (userAction === 'ADDED_FRIEND') {
        addLog('🛑 检测到用户已加微，【第三击】自动熔断取消', 'success');
        stopSimulation('CONVERTED');
      } else {
        setCurrentStep(3);
        addLog('📩 触发第三击：外呼未接通/未加微，发送兜底短信', 'error');
        addNotification('sms', '【系统通知】', '您的 39.9 元课程名额保留最后 2 小时，请点击链接添加班主任：https://url.cn/xyz');
        stopSimulation('COMPLETED');
      }
    }

  }, [timer, status, currentStep, userAction]);

  // --- User Interaction ---
  const handleUserAddFriend = () => {
    if (status !== 'RUNNING') return;
    setUserAction('ADDED_FRIEND');
    addLog('✅ 回调信号接收：用户通过了企业微信好友申请', 'success');
    addNotification('wecom', '企业微信', '您已成功添加 "华坤AI助教" 为联系人');
  };

  const reset = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus('IDLE');
    setCurrentStep(0);
    setTimer(0);
    setLogs([]);
    setPhoneNotifications([]);
    setUserAction(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <UserPlus className="text-indigo-600" />
              多渠道加微三连击 <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full font-bold">P0 级核心功能</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              自动化 SOP 演示 • 模拟 30 分钟内的密集触达策略 • <span className="font-mono text-indigo-600">Simulating Time Scale: 1s = 6min</span>
            </p>
          </div>
          <div className="flex gap-3">
             {status === 'IDLE' || status === 'COMPLETED' || status === 'CONVERTED' ? (
                <button 
                  onClick={startSimulation}
                  className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 transition-all flex items-center gap-2 shadow-lg shadow-indigo-200"
                >
                  <Play className="w-4 h-4" /> 启动模拟 (Start SOP)
                </button>
             ) : (
               <button 
                  onClick={reset}
                  className="px-6 py-2 bg-slate-100 text-slate-600 rounded-lg font-bold hover:bg-slate-200 transition-all flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" /> 重置 (Reset)
                </button>
             )}
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[750px] lg:h-[650px]">
          
          {/* Left Column: Strategy Monitor */}
          <div className="lg:col-span-8 flex flex-col gap-6 h-full">
            
            {/* Timeline Visualization */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex-1 relative overflow-hidden">
               <div className="flex justify-between items-center mb-8">
                 <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                   <Settings className="w-4 h-4" /> 策略执行管道 (Pipeline Monitor)
                 </h2>
                 <div className="flex items-center gap-2 bg-slate-100 px-3 py-1 rounded-full">
                   <Clock className={`w-4 h-4 ${status === 'RUNNING' ? 'text-indigo-600 animate-spin' : 'text-slate-400'}`} />
                   <span className="font-mono font-bold text-slate-700">T + {timer}s</span>
                   {status === 'RUNNING' && <span className="text-xs text-slate-400">(Simulating...)</span>}
                 </div>
               </div>

               <div className="relative flex justify-between items-start pt-8 px-4">
                  {/* Connecting Line */}
                  <div className="absolute top-[4.5rem] left-0 w-full h-1 bg-slate-100 -z-0"></div>
                  <div 
                    className="absolute top-[4.5rem] left-0 h-1 bg-indigo-500 -z-0 transition-all duration-1000 ease-linear"
                    style={{ width: status === 'IDLE' ? '0%' : `${Math.min((timer / 12) * 100, 100)}%` }}
                  ></div>

                  {STEPS.map((step) => {
                    const isPassed = currentStep >= step.id;
                    const isCurrent = currentStep === step.id && status === 'RUNNING';
                    const isCancelled = userAction === 'ADDED_FRIEND' && step.id > currentStep;
                    
                    return (
                      <div key={step.id} className="relative z-10 flex flex-col items-center w-1/3">
                        <div className={`
                          w-12 h-12 rounded-full flex items-center justify-center border-4 transition-all duration-500
                          ${isCurrent ? 'bg-white border-indigo-600 scale-125 shadow-xl shadow-indigo-100' : 
                            isPassed ? 'bg-indigo-600 border-indigo-600 text-white' : 
                            isCancelled ? 'bg-slate-100 border-slate-200 opacity-50' : 'bg-white border-slate-200 text-slate-300'}
                        `}>
                          {isCancelled ? <Ban className="w-5 h-5 text-slate-400" /> : 
                           isPassed ? <CheckCircle className="w-5 h-5" /> : 
                           step.id === 1 ? <MessageSquare className="w-5 h-5" /> :
                           step.id === 2 ? <PhoneCall className="w-5 h-5" /> :
                           <AlertTriangle className="w-5 h-5" />}
                        </div>
                        <div className="mt-4 text-center">
                          <div className="text-xs font-bold text-slate-400 mb-1">{step.timeLabel}</div>
                          <h3 className={`font-bold ${isCurrent ? 'text-indigo-700' : isCancelled ? 'text-slate-300 line-through' : 'text-slate-700'}`}>
                            {step.name}
                          </h3>
                          <p className="text-xs text-slate-500 mt-1 max-w-[120px] mx-auto leading-tight">
                            {step.desc}
                          </p>
                        </div>
                        {isCancelled && (
                          <div className="absolute -top-8 bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold animate-bounce">
                            已熔断 (Cancelled)
                          </div>
                        )}
                      </div>
                    )
                  })}
               </div>

               {/* Configuration Info */}
               <div className="mt-12 bg-slate-50 rounded-lg p-4 text-xs text-slate-500 space-y-2 border border-slate-100">
                 <div className="flex gap-4">
                   <span className="font-bold">当前配置：</span>
                   <span>Strike 1: SMS (立即)</span>
                   <span>Strike 2: AI Call (延迟 5s)</span>
                   <span>Strike 3: Final SMS (延迟 10s)</span>
                 </div>
                 <div className="flex gap-4 text-amber-600">
                    <ShieldCheck className="w-3 h-3" />
                    <span>高频防护开启：单日同一号码最多外呼 1 次</span>
                 </div>
               </div>
            </div>

            {/* System Logs */}
            <div className="bg-slate-900 text-slate-300 rounded-xl p-4 font-mono text-xs h-48 overflow-y-auto shadow-inner flex flex-col">
              <div className="sticky top-0 bg-slate-900 pb-2 border-b border-slate-700 mb-2 flex justify-between items-center">
                <span className="font-bold text-slate-400 flex items-center gap-2">
                  <Settings className="w-3 h-3" /> SYSTEM KERNEL LOGS
                </span>
                {status === 'RUNNING' && <span className="text-green-500 animate-pulse">● Active</span>}
              </div>
              <div className="space-y-1.5 flex-1">
                {logs.length === 0 && <span className="text-slate-600 italic">Ready to start simulation...</span>}
                {logs.map((log) => (
                  <div key={log.id} className="flex gap-3">
                    <span className="text-slate-500 whitespace-nowrap">[{log.time}]</span>
                    <span className={`${
                      log.type === 'start' ? 'text-blue-400 font-bold' : 
                      log.type === 'success' ? 'text-green-400 font-bold' : 
                      log.type === 'error' ? 'text-red-400' : 
                      log.type === 'warning' ? 'text-amber-400' : 
                      log.type === 'action' ? 'text-indigo-300' : 'text-slate-300'
                    }`}>
                      {log.msg}
                    </span>
                  </div>
                ))}
              </div>
            </div>

          </div>

          {/* Right Column: User Phone Simulator */}
          <div className="lg:col-span-4 h-full flex justify-center">
             <div className="w-[320px] h-full bg-slate-800 rounded-[3rem] border-8 border-slate-900 shadow-2xl relative overflow-hidden flex flex-col">
               {/* Phone Notch */}
               <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-6 bg-slate-900 rounded-b-xl z-20"></div>
               
               {/* Status Bar */}
               <div className="bg-white px-6 pt-3 pb-1 flex justify-between text-[10px] font-bold text-slate-800 z-10">
                 <span>9:41</span>
                 <div className="flex gap-1">
                   <span>5G</span>
                   <span>100%</span>
                 </div>
               </div>

               {/* Screen Content */}
               <div className="flex-1 bg-slate-100 relative overflow-hidden flex flex-col">
                 
                 {/* App Interface Background (Fake WeCom) */}
                 <div className="flex-1 bg-white p-4">
                   <div className="flex items-center justify-between mb-4 mt-2">
                     <span className="font-bold text-lg text-slate-800">微信 (WeChat)</span>
                     <UserPlus className="w-5 h-5 text-slate-800" />
                   </div>
                   
                   {/* Friend List / Feed */}
                   <div className="space-y-4 opacity-30 blur-[1px]">
                     {[1,2,3,4,5].map(i => (
                       <div key={i} className="flex gap-3 items-center">
                         <div className="w-10 h-10 bg-slate-200 rounded-lg"></div>
                         <div className="flex-1 space-y-1">
                           <div className="w-20 h-2 bg-slate-200 rounded"></div>
                           <div className="w-full h-2 bg-slate-100 rounded"></div>
                         </div>
                       </div>
                     ))}
                   </div>

                   {/* Notification Overlay (The Action happens here) */}
                   <div className="absolute top-0 left-0 w-full h-full pointer-events-none p-2 pt-12 space-y-2 flex flex-col items-center">
                     {phoneNotifications.map((notif) => (
                       <div key={notif.id} className="w-full bg-white/95 backdrop-blur shadow-lg rounded-2xl p-3 border border-slate-100 animate-in slide-in-from-top-4 duration-500 pointer-events-auto">
                          <div className="flex justify-between items-start mb-1">
                             <div className="flex items-center gap-2">
                               <div className={`p-1 rounded ${
                                 notif.type === 'sms' ? 'bg-green-500' : 
                                 notif.type === 'call' ? 'bg-blue-500' : 'bg-indigo-500'
                               }`}>
                                 {notif.type === 'sms' ? <MessageSquare className="w-3 h-3 text-white" /> : 
                                  notif.type === 'call' ? <PhoneCall className="w-3 h-3 text-white" /> : 
                                  <UserPlus className="w-3 h-3 text-white" />}
                               </div>
                               <span className="text-xs font-bold text-slate-700 uppercase">{notif.type === 'wecom' ? '微信' : notif.type === 'call' ? '电话' : '信息'}</span>
                             </div>
                             <span className="text-[10px] text-slate-400">刚刚</span>
                          </div>
                          <div className="pl-7">
                            <h4 className="text-sm font-bold text-slate-900">{notif.title}</h4>
                            <p className="text-xs text-slate-600 leading-snug mt-0.5">{notif.content}</p>
                            
                            {/* Interactive Buttons on Notification */}
                            {notif.type === 'call' && status === 'RUNNING' && (
                               <div className="flex gap-2 mt-2">
                                  <div className="flex-1 bg-red-500 text-white text-center py-1.5 rounded-lg text-xs font-bold">挂断</div>
                                  <div className="flex-1 bg-green-500 text-white text-center py-1.5 rounded-lg text-xs font-bold">接听</div>
                               </div>
                            )}
                          </div>
                       </div>
                     ))}
                   </div>
                 </div>

                 {/* Bottom Action Area: Simulate User Adding Friend */}
                 <div className="p-4 bg-white border-t border-slate-100 z-20">
                    <p className="text-[10px] text-slate-400 text-center mb-2">
                      模拟真实用户行为 (Interrupt Logic)
                    </p>
                    <button
                      onClick={handleUserAddFriend}
                      disabled={status !== 'RUNNING' || userAction === 'ADDED_FRIEND'}
                      className={`
                        w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg
                        ${status === 'RUNNING' && userAction !== 'ADDED_FRIEND'
                          ? 'bg-green-500 text-white hover:bg-green-600 active:scale-95 shadow-green-200' 
                          : 'bg-slate-100 text-slate-400 cursor-not-allowed'}
                      `}
                    >
                      {userAction === 'ADDED_FRIEND' ? (
                         <>
                           <CheckCircle className="w-4 h-4" /> 已添加好友
                         </>
                      ) : (
                         <>
                           <UserPlus className="w-4 h-4" /> 模拟用户通过好友申请
                         </>
                      )}
                    </button>
                 </div>

               </div>
               
               {/* Home Bar */}
               <div className="h-1 bg-white absolute bottom-2 left-1/2 -translate-x-1/2 w-1/3 rounded-full opacity-50"></div>
             </div>
          </div>

        </div>
      </div>
    </div>
  );
}