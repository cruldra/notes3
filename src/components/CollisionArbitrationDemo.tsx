import React, { useState, useEffect, useRef } from 'react';
import { 
  Users, 
  Smartphone, 
  AlertTriangle, 
  GitMerge, 
  ShieldCheck, 
  Activity, 
  Clock, 
  TrendingUp, 
  Database,
  UserCheck,
  PhoneCall,
  MessageSquare
} from 'lucide-react';

// --- Mock Data & Constants ---
const CHANNELS = {
  DOUYIN: { id: 'douyin', name: '抖音信息流', color: 'bg-black', textColor: 'text-white', weight: 5, icon: '🎵' },
  BAIDU: { id: 'baidu', name: '百度搜索', color: 'bg-blue-600', textColor: 'text-white', weight: 10, icon: '🔍' }
};

const SALES_TEAMS = {
  DOUYIN_TEAM: { id: 'sales_a', name: '销售组 A (抖音)', avatar: '👨‍💼' },
  BAIDU_TEAM: { id: 'sales_b', name: '销售组 B (百度)', avatar: '👩‍💼' }
};

export default function App() {
  // --- State ---
  const [systemEnabled, setSystemEnabled] = useState(false); // Toggle between Pain Point & Solution
  const [strategy, setStrategy] = useState('weight'); // 'time' or 'weight'
  const [leads, setLeads] = useState([]);
  const [logs, setLogs] = useState([]);
  const [phoneNumber, setPhoneNumber] = useState('13800138000');
  const [processing, setProcessing] = useState(null); // 'douyin' or 'baidu' or null
  const [notification, setNotification] = useState(null);

  // --- Actions ---

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [{ id: Date.now(), time: timestamp, message, type }, ...prev]);
  };

  const showNotification = (title, message, type = 'info') => {
    setNotification({ title, message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  const handleLeadSubmit = async (channelKey) => {
    if (processing) return;
    setProcessing(channelKey);
    const channel = CHANNELS[channelKey];
    
    // Simulate Network Delay
    await new Promise(resolve => setTimeout(resolve, 600));

    const existingLeadIndex = leads.findIndex(l => l.phone === phoneNumber);
    const timestamp = Date.now();

    // --- Scenario 1: Pain Point (System Disabled) ---
    if (!systemEnabled) {
      if (existingLeadIndex >= 0) {
        // DUPLICATE CREATION (The Pain Point)
        const newLead = {
          id: timestamp,
          phone: phoneNumber,
          channel: channel.name,
          channelId: channel.id,
          salesRep: channel.id === 'douyin' ? SALES_TEAMS.DOUYIN_TEAM : SALES_TEAMS.BAIDU_TEAM,
          status: 'new',
          timestamp: timestamp,
          history: []
        };
        setLeads(prev => [newLead, ...prev]);
        addLog(`⚠️ 警告：手机号 ${phoneNumber} 重复入库！造成销售撞单风险。`, 'error');
        showNotification('撞单发生！', '客户将被两个销售同时骚扰', 'error');
      } else {
        // Normal Creation
        createNewLead(channel, timestamp);
      }
      setProcessing(null);
      return;
    }

    // --- Scenario 2: Smart Solution (Arbitration) ---
    if (existingLeadIndex >= 0) {
      // COLLISION DETECTED
      const existingLead = leads[existingLeadIndex];
      addLog(`⚡️ 触发仲裁：检测到 ${phoneNumber} 已存在，当前归属 [${existingLead.salesRep.name}]`, 'warning');
      
      let winner = null;
      let reason = '';
      
      // Arbitration Logic
      if (strategy === 'time') {
        // First Come First Served: Existing lead keeps ownership, new data is merged
        winner = 'existing';
        reason = '策略：先来先得 (保留原销售)';
      } else if (strategy === 'weight') {
        // Channel Weight Logic
        const newWeight = channel.weight;
        const oldWeight = CHANNELS[existingLead.channelId.toUpperCase()]?.weight || 0;
        
        if (newWeight > oldWeight) {
          winner = 'new';
          reason = `策略：权重优先 (${channel.name}权重${newWeight} > ${existingLead.channel}权重${oldWeight})`;
        } else {
          winner = 'existing';
          reason = `策略：权重优先 (原渠道权重${oldWeight} >= 新渠道权重${newWeight})`;
        }
      }

      // Execute Arbitration
      const updatedLeads = [...leads];
      const targetLead = { ...updatedLeads[existingLeadIndex] };
      
      if (winner === 'new') {
        // Transfer Ownership
        const oldSalesName = targetLead.salesRep.name;
        targetLead.salesRep = channel.id === 'douyin' ? SALES_TEAMS.DOUYIN_TEAM : SALES_TEAMS.BAIDU_TEAM;
        targetLead.channel = channel.name; // Update main channel source
        targetLead.channelId = channel.id;
        addLog(`✅ 仲裁结果：改派给 [${targetLead.salesRep.name}]。原因：${reason}`, 'success');
        showNotification('自动改派成功', `高权重渠道覆盖，线索已移交 ${targetLead.salesRep.name}`, 'success');
        
        // Log merge info
        targetLead.history.push(`原归属: ${oldSalesName} (被高权重渠道覆盖)`);
      } else {
        // Merge Only
        addLog(`🛡️ 仲裁结果：维持归属 [${targetLead.salesRep.name}]。原因：${reason}`, 'info');
        showNotification('自动合并完成', '线索维持原归属，新渠道信息已合并', 'info');
      }

      // Add merge history
      targetLead.history.push(`${new Date().toLocaleTimeString()} 从 ${channel.name} 再次提交 (已合并)`);
      targetLead.mergeCount = (targetLead.mergeCount || 0) + 1;
      
      updatedLeads[existingLeadIndex] = targetLead;
      setLeads(updatedLeads);

    } else {
      // No Collision -> Normal Create
      createNewLead(channel, timestamp);
    }
    
    setProcessing(null);
  };

  const createNewLead = (channel, timestamp) => {
    const newLead = {
      id: timestamp,
      phone: phoneNumber,
      channel: channel.name,
      channelId: channel.id,
      salesRep: channel.id === 'douyin' ? SALES_TEAMS.DOUYIN_TEAM : SALES_TEAMS.BAIDU_TEAM,
      status: 'new',
      timestamp: timestamp,
      mergeCount: 0,
      history: []
    };
    setLeads(prev => [newLead, ...prev]);
    addLog(`🆕 新线索入库：${phoneNumber} 来自 ${channel.name}`, 'normal');
    if(systemEnabled) {
        showNotification('新线索分配', `分配给 ${newLead.salesRep.name}`, 'success');
    }
  };

  const clearData = () => {
    setLeads([]);
    setLogs([]);
    setNotification(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row justify-between items-center bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <GitMerge className="text-indigo-600" />
              撞单仲裁与合并系统 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2 Demo</span>
            </h1>
            <p className="text-gray-500 mt-1 text-sm">
              演示多渠道线索冲突时的 <span className="font-semibold text-red-500">痛点</span> 与 <span className="font-semibold text-green-600">自动化解决方案</span>
            </p>
          </div>
          
          <div className="flex items-center gap-4 mt-4 md:mt-0">
            <div className="flex items-center gap-2 bg-gray-100 p-1 rounded-lg">
              <button 
                onClick={() => setSystemEnabled(false)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${!systemEnabled ? 'bg-white shadow-sm text-red-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <AlertTriangle className="w-4 h-4 inline mr-1" />
                关闭仲裁 (痛点演示)
              </button>
              <button 
                onClick={() => setSystemEnabled(true)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${systemEnabled ? 'bg-white shadow-sm text-green-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <ShieldCheck className="w-4 h-4 inline mr-1" />
                开启仲裁 (解决方案)
              </button>
            </div>
          </div>
        </header>

        {/* Main Configuration & Strategy (Visible only when Enabled) */}
        {systemEnabled && (
          <div className="bg-indigo-50 border border-indigo-100 p-4 rounded-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="flex items-center gap-3">
              <Activity className="text-indigo-600 w-5 h-5" />
              <div>
                <h3 className="font-semibold text-indigo-900">仲裁策略配置</h3>
                <p className="text-xs text-indigo-700">当检测到重复线索时，系统将依据以下规则判定归属权</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setStrategy('time')}
                className={`px-4 py-2 text-sm border rounded-lg flex items-center gap-2 transition-all ${strategy === 'time' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-600 border-gray-200 hover:border-indigo-300'}`}
              >
                <Clock className="w-4 h-4" /> 先来先得 (P1)
              </button>
              <button 
                onClick={() => setStrategy('weight')}
                className={`px-4 py-2 text-sm border rounded-lg flex items-center gap-2 transition-all ${strategy === 'weight' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-600 border-gray-200 hover:border-indigo-300'}`}
              >
                <TrendingUp className="w-4 h-4" /> 渠道权重 (P0)
              </button>
            </div>
            <div className="text-xs text-gray-500 bg-white px-3 py-2 rounded-lg border border-gray-200 hidden md:block">
              当前权重设置: 抖音({CHANNELS.DOUYIN.weight}) vs 百度({CHANNELS.BAIDU.weight})
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left Column: User Simulation */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                  <Smartphone className="text-gray-400" />
                  用户端模拟
                </h2>
                <button onClick={() => setPhoneNumber(`138${Math.floor(Math.random()*90000000)}`)} className="text-xs text-blue-500 hover:underline">
                  随机换号
                </button>
              </div>

              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700">输入测试手机号</label>
                <input 
                  type="text" 
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  className="w-full text-xl font-mono tracking-wider p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none text-center"
                />
              </div>

              <div className="mt-8 grid grid-cols-1 gap-4">
                {/* Douyin Simulator */}
                <div className="border border-gray-200 rounded-xl p-4 bg-gray-50 relative overflow-hidden group">
                  <div className="absolute top-0 left-0 w-1 h-full bg-black"></div>
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">🎵</span>
                      <span className="font-bold text-gray-800">抖音广告页</span>
                    </div>
                    <span className="text-xs bg-gray-200 px-2 py-1 rounded">权重: {CHANNELS.DOUYIN.weight}</span>
                  </div>
                  <p className="text-xs text-gray-500 mb-4">场景：用户刷到短视频广告，填写表单。</p>
                  <button 
                    onClick={() => handleLeadSubmit('DOUYIN')}
                    disabled={!!processing}
                    className="w-full py-2 bg-black text-white rounded-lg hover:bg-gray-800 active:scale-95 transition-all flex justify-center items-center gap-2 disabled:opacity-50"
                  >
                    {processing === 'DOUYIN' ? '提交中...' : '提交抖音线索'}
                  </button>
                </div>

                {/* Baidu Simulator */}
                <div className="border border-gray-200 rounded-xl p-4 bg-blue-50 relative overflow-hidden group">
                  <div className="absolute top-0 left-0 w-1 h-full bg-blue-600"></div>
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">🔍</span>
                      <span className="font-bold text-gray-800">百度落地页</span>
                    </div>
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">权重: {CHANNELS.BAIDU.weight}</span>
                  </div>
                  <p className="text-xs text-gray-500 mb-4">场景：用户主动搜索关键词，进入官网咨询。</p>
                  <button 
                    onClick={() => handleLeadSubmit('BAIDU')}
                    disabled={!!processing}
                    className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 active:scale-95 transition-all flex justify-center items-center gap-2 disabled:opacity-50"
                  >
                    {processing === 'BAIDU' ? '提交中...' : '提交百度线索'}
                  </button>
                </div>
              </div>
              
              <div className="mt-6 text-xs text-gray-400 text-center">
                提示：尝试使用同一手机号连续点击上方两个按钮
              </div>
            </div>
          </div>

          {/* Right Column: CRM System View */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* Notification Banner */}
            {notification && (
              <div className={`p-4 rounded-xl flex items-start gap-3 shadow-lg transform transition-all animate-in fade-in slide-in-from-top-2 ${
                notification.type === 'error' ? 'bg-red-50 text-red-800 border border-red-200' : 
                notification.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' : 
                'bg-blue-50 text-blue-800 border border-blue-200'
              }`}>
                {notification.type === 'error' ? <AlertTriangle className="mt-1" /> : notification.type === 'success' ? <ShieldCheck className="mt-1" /> : <GitMerge className="mt-1" />}
                <div>
                  <h4 className="font-bold">{notification.title}</h4>
                  <p className="text-sm opacity-90">{notification.message}</p>
                </div>
              </div>
            )}

            {/* Leads Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden min-h-[400px]">
              <div className="p-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                  <Database className="text-gray-400" />
                  CRM 线索池 & 销售分配
                </h2>
                <div className="flex items-center gap-2">
                   <span className="text-xs font-medium bg-gray-200 px-2 py-1 rounded-full text-gray-600">{leads.length} 条记录</span>
                   <button onClick={clearData} className="text-xs text-gray-500 hover:text-red-500 px-2">清空数据</button>
                </div>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
                    <tr>
                      <th className="px-6 py-3">客户手机号</th>
                      <th className="px-6 py-3">来源渠道</th>
                      <th className="px-6 py-3">归属销售</th>
                      <th className="px-6 py-3">状态 / 备注</th>
                      <th className="px-6 py-3">操作</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {leads.map((lead, idx) => (
                      <tr key={`${lead.id}-${idx}`} className={`group hover:bg-gray-50 transition-colors ${!systemEnabled && leads.filter(l => l.phone === lead.phone).length > 1 ? 'bg-red-50 hover:bg-red-100' : ''}`}>
                        <td className="px-6 py-4 font-mono font-medium text-gray-900">
                          {lead.phone}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            lead.channelId === 'douyin' ? 'bg-gray-800 text-white' : 'bg-blue-100 text-blue-800'
                          }`}>
                            {lead.channelId === 'douyin' ? '🎵 抖音' : '🔍 百度'}
                          </span>
                          {lead.mergeCount > 0 && (
                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-1">
                              <GitMerge className="w-3 h-3" /> 已合并 {lead.mergeCount} 条
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <span className="text-lg bg-gray-100 rounded-full p-1">{lead.salesRep.avatar}</span>
                            <span className="text-sm font-medium text-gray-700">{lead.salesRep.name}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                           {!systemEnabled && leads.filter(l => l.phone === lead.phone).length > 1 ? (
                             <span className="inline-flex items-center gap-1 text-xs font-bold text-red-600 bg-red-100 px-2 py-1 rounded">
                               <AlertTriangle className="w-3 h-3" /> 撞单风险
                             </span>
                           ) : (
                             <div className="space-y-1">
                                <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700 bg-green-100 px-2 py-1 rounded">
                                  <UserCheck className="w-3 h-3" /> 跟进中
                                </span>
                                {lead.history.length > 0 && (
                                  <p className="text-[10px] text-gray-400 max-w-[150px] truncate" title={lead.history[lead.history.length-1]}>
                                    最新: {lead.history[lead.history.length-1]}
                                  </p>
                                )}
                             </div>
                           )}
                        </td>
                        <td className="px-6 py-4">
                            <div className="flex gap-2 opacity-20 group-hover:opacity-100 transition-opacity">
                                <button className="p-1 hover:bg-blue-100 rounded text-blue-600" title="外呼"><PhoneCall className="w-4 h-4"/></button>
                                <button className="p-1 hover:bg-green-100 rounded text-green-600" title="企微"><MessageSquare className="w-4 h-4"/></button>
                            </div>
                        </td>
                      </tr>
                    ))}
                    {leads.length === 0 && (
                      <tr>
                        <td colSpan="5" className="px-6 py-12 text-center text-gray-400 text-sm">
                          暂无数据，请在左侧模拟线索提交
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* System Logs */}
            <div className="bg-gray-900 text-gray-300 rounded-xl p-4 font-mono text-xs h-48 overflow-y-auto shadow-inner border border-gray-800">
              <div className="sticky top-0 bg-gray-900 pb-2 border-b border-gray-800 mb-2 flex justify-between items-center">
                <span className="font-bold text-gray-400 uppercase tracking-wider">System Arbitration Logs</span>
                <span className="text-gray-600">Real-time</span>
              </div>
              <div className="space-y-1.5">
                {logs.length === 0 && <span className="text-gray-600 italic">等待系统事件...</span>}
                {logs.map((log) => (
                  <div key={log.id} className="flex gap-3">
                    <span className="text-gray-500 whitespace-nowrap">[{log.time}]</span>
                    <span className={`${
                      log.type === 'error' ? 'text-red-400 font-bold' : 
                      log.type === 'success' ? 'text-green-400' : 
                      log.type === 'warning' ? 'text-yellow-400' : 'text-gray-300'
                    }`}>
                      {log.message}
                    </span>
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