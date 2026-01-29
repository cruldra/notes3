import React, { useState, useEffect } from 'react';
import { 
  PieChart, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Target, 
  AlertTriangle,
  StopCircle,
  Zap,
  BarChart2,
  Filter,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

// --- Mock Data Constants ---
const INITIAL_CHANNELS = [
  { 
    id: 'douyin_live', 
    name: '抖音直播间 (Douyin Live)', 
    type: 'Video',
    tags: ['High Growth', 'Core'],
    cost: 50000, 
    leads: 1200, 
    orders: 180, 
    gmv: 179640, // ROI ~3.59
    status: 'ACTIVE',
    history: [3.2, 3.4, 3.5, 3.6, 3.59]
  },
  { 
    id: 'baidu_search', 
    name: '百度搜索 (Baidu Search)', 
    type: 'Search',
    tags: ['Legacy', 'Expensive'],
    cost: 30000, 
    leads: 150, 
    orders: 8, 
    gmv: 23920, // ROI ~0.8 (Loss)
    status: 'ACTIVE',
    history: [1.2, 1.0, 0.9, 0.85, 0.8]
  },
  { 
    id: 'wechat_moments', 
    name: '朋友圈广告 (Moments)', 
    type: 'Social',
    tags: ['Stable'],
    cost: 20000, 
    leads: 400, 
    orders: 45, 
    gmv: 44910, // ROI ~2.25
    status: 'ACTIVE',
    history: [2.1, 2.2, 2.3, 2.2, 2.25]
  }
];

const LIVE_FEED_MOCK = [
  { time: '10:42:01', source: '抖音直播间', action: '支付成功', amount: 998, utm: 'utm_source=douyin&utm_campaign=live_001' },
  { time: '10:41:45', source: '百度搜索', action: '线索入库', amount: 0, utm: 'utm_source=baidu&utm_campaign=sem_keyword_ai' },
  { time: '10:41:12', source: '朋友圈广告', action: '支付成功', amount: 998, utm: 'utm_source=wechat&utm_campaign=feed_img_03' },
  { time: '10:40:55', source: '抖音直播间', action: '支付成功', amount: 2980, utm: 'utm_source=douyin&utm_campaign=live_001' },
];

export default function ChannelROIDashboard() {
  // --- State ---
  const [channels, setChannels] = useState(INITIAL_CHANNELS);
  const [liveFeed, setLiveFeed] = useState(LIVE_FEED_MOCK);
  const [selectedPeriod, setSelectedPeriod] = useState('This Month');
  const [simulationMode, setSimulationMode] = useState(false);
  const [projectedSavings, setProjectedSavings] = useState(0);

  // --- Calculations ---
  const totalCost = channels.reduce((acc, c) => c.status === 'ACTIVE' ? acc + c.cost : acc, 0);
  const totalGMV = channels.reduce((acc, c) => c.status === 'ACTIVE' ? acc + c.gmv : acc, 0);
  const totalROI = totalCost > 0 ? (totalGMV / totalCost).toFixed(2) : 0;

  // --- Effects (Simulate Live Data) ---
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate a new order coming in randomly
      if (Math.random() > 0.6) {
        const isDouyin = Math.random() > 0.3; // Douyin converts more often
        const channelId = isDouyin ? 'douyin_live' : 'wechat_moments';
        const amount = isDouyin ? 2980 : 998;
        
        // Update Feed
        const newEvent = {
          time: new Date().toLocaleTimeString('en-US', { hour12: false }),
          source: isDouyin ? '抖音直播间' : '朋友圈广告',
          action: '支付成功',
          amount: amount,
          utm: isDouyin ? 'utm_source=douyin' : 'utm_source=wechat'
        };
        setLiveFeed(prev => [newEvent, ...prev.slice(0, 4)]);

        // Update Stats (Real-time calculation logic)
        setChannels(prev => prev.map(c => {
          if (c.id === channelId && c.status === 'ACTIVE') {
            return { ...c, gmv: c.gmv + amount, orders: c.orders + 1 };
          }
          return c;
        }));
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // --- Actions ---
  const handleToggleChannel = (id) => {
    setChannels(prev => prev.map(c => {
      if (c.id === id) {
        const newStatus = c.status === 'ACTIVE' ? 'PAUSED' : 'ACTIVE';
        if (newStatus === 'PAUSED') {
          setSimulationMode(true);
          setProjectedSavings(prev => prev + c.cost);
        } else {
          setProjectedSavings(prev => prev - c.cost);
          if (projectedSavings - c.cost <= 0) setSimulationMode(false);
        }
        return { ...c, status: newStatus };
      }
      return c;
    }));
  };

  const getROIColor = (roi) => {
    if (roi >= 3) return 'text-green-600 bg-green-50';
    if (roi >= 1.5) return 'text-blue-600 bg-blue-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-center bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <BarChart2 className="text-indigo-600" />
              渠道 ROI 实时罗盘 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Phase 2</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              全链路归因监控 • 流量-转化-成交实时闭环
            </p>
          </div>
          <div className="flex gap-3 mt-4 md:mt-0">
            <div className="flex bg-slate-100 p-1 rounded-lg">
              {['Last 7 Days', 'This Month', 'This Quarter'].map(p => (
                <button 
                  key={p}
                  onClick={() => setSelectedPeriod(p)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${selectedPeriod === p ? 'bg-white shadow-sm text-slate-800' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Top KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <KPICard 
            title="综合 ROI (Total ROI)" 
            value={totalROI} 
            prefix="" 
            trend="+0.2" 
            isGood={true}
            icon={<Target className="w-5 h-5 text-indigo-600" />}
            desc="投入产出比 (GMV / Cost)"
          />
          <KPICard 
            title="总成交金额 (GMV)" 
            value={totalGMV.toLocaleString()} 
            prefix="¥" 
            trend="+12%" 
            isGood={true}
            icon={<DollarSign className="w-5 h-5 text-green-600" />}
            desc="实际支付总额"
          />
          <KPICard 
            title="总投放消耗 (Spend)" 
            value={totalCost.toLocaleString()} 
            prefix="¥" 
            trend="-5%" 
            isGood={true} // Spending less is usually good if ROI is high
            icon={<Activity className="w-5 h-5 text-orange-600" />}
            desc="各渠道广告费总和"
          />
           <KPICard 
            title="平均获客成本 (CAC)" 
            value={Math.round(totalCost / channels.filter(c=>c.status==='ACTIVE').reduce((a,c)=>a+c.orders,0))} 
            prefix="¥" 
            trend="+2%" 
            isGood={false}
            icon={<TrendingUp className="w-5 h-5 text-blue-600" />}
            desc="Cost per Order"
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: Channel Detail List */}
          <div className="lg:col-span-8 space-y-4">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                <h2 className="font-bold text-slate-800 flex items-center gap-2">
                  <Filter className="w-4 h-4 text-slate-400" />
                  渠道表现明细
                </h2>
                {simulationMode && (
                  <div className="flex items-center gap-2 bg-yellow-50 text-yellow-700 px-3 py-1 rounded-full text-xs font-bold animate-pulse">
                    <Zap className="w-3 h-3" />
                    预测：停止劣质渠道将节省预算 ¥{projectedSavings.toLocaleString()}
                  </div>
                )}
              </div>
              
              <div className="divide-y divide-slate-100">
                {channels.map((channel) => {
                  const roi = (channel.gmv / channel.cost).toFixed(2);
                  const isLoss = roi < 1;
                  const isPaused = channel.status === 'PAUSED';

                  return (
                    <div key={channel.id} className={`p-5 transition-all ${isPaused ? 'bg-slate-50 opacity-60 grayscale' : 'hover:bg-slate-50'}`}>
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${channel.id.includes('douyin') ? 'bg-black text-white' : channel.id.includes('baidu') ? 'bg-blue-600 text-white' : 'bg-green-600 text-white'}`}>
                            {channel.id.includes('douyin') ? <TrendingUp className="w-5 h-5" /> : channel.id.includes('baidu') ? <Target className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                                <h3 className="font-bold text-lg text-slate-900">{channel.name}</h3>
                                {channel.tags.map(tag => (
                                    <span key={tag} className="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded border border-slate-200">{tag}</span>
                                ))}
                            </div>
                            <div className="text-xs text-slate-500 mt-1 flex gap-4">
                              <span>消耗: ¥{channel.cost.toLocaleString()}</span>
                              <span>线索: {channel.leads}</span>
                              <span>成交: {channel.orders} 单</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className={`text-2xl font-black ${getROIColor(roi).split(' ')[0]}`}>
                            {roi}
                          </div>
                          <div className="text-xs text-slate-400 font-medium">ROI</div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-100 border-dashed">
                        <div className="flex gap-8">
                            <div>
                                <div className="text-xs text-slate-400">GMV 贡献</div>
                                <div className="font-mono font-bold text-slate-700">¥{channel.gmv.toLocaleString()}</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-400">CAC (获客成本)</div>
                                <div className="font-mono font-bold text-slate-700">¥{Math.round(channel.cost / channel.orders)}</div>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <button 
                          onClick={() => handleToggleChannel(channel.id)}
                          className={`
                            px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-all
                            ${isPaused 
                                ? 'bg-slate-200 text-slate-500 hover:bg-slate-300' 
                                : isLoss 
                                    ? 'bg-red-100 text-red-700 hover:bg-red-200 border border-red-200 shadow-sm' 
                                    : 'bg-green-100 text-green-700 hover:bg-green-200 border border-green-200'}
                          `}
                        >
                          {isPaused ? (
                            <>恢复投放</>
                          ) : isLoss ? (
                            <><StopCircle className="w-3 h-3" /> 立即关停止损</>
                          ) : (
                            <><Zap className="w-3 h-3" /> 加大预算投放</>
                          )}
                        </button>
                      </div>
                      
                      {/* Warning Banner for Low ROI */}
                      {!isPaused && isLoss && (
                        <div className="mt-3 bg-red-50 text-red-800 text-xs p-2 rounded flex items-center gap-2 animate-in slide-in-from-top-2">
                           <AlertTriangle className="w-3 h-3" />
                           <span className="font-bold">警告：</span> 该渠道 ROI 低于 1.0，当前处于亏损状态，建议立即优化或关停。
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right: Live Data & Insights */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Live Transactions Feed */}
            <div className="bg-slate-900 text-slate-300 rounded-xl p-5 h-[400px] flex flex-col shadow-lg border border-slate-800">
               <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-3">
                 <h3 className="font-bold text-white flex items-center gap-2">
                   <Activity className="w-4 h-4 text-green-400 animate-pulse" /> 实时成交回传
                 </h3>
                 <span className="text-[10px] bg-slate-800 px-2 py-1 rounded text-slate-400">Webhook Active</span>
               </div>
               
               <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                 {liveFeed.map((item, idx) => (
                   <div key={idx} className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50 animate-in slide-in-from-left-4 fade-in">
                     <div className="flex justify-between items-center mb-1">
                       <span className="text-xs font-mono text-slate-500">{item.time}</span>
                       <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${item.source === '抖音直播间' ? 'bg-blue-900/30 text-blue-400' : item.source === '百度搜索' ? 'bg-red-900/30 text-red-400' : 'bg-green-900/30 text-green-400'}`}>
                         {item.source}
                       </span>
                     </div>
                     <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-300">{item.action}</span>
                        <span className="text-sm font-bold text-white font-mono">+ ¥{item.amount}</span>
                     </div>
                     <div className="mt-2 text-[10px] text-slate-500 font-mono truncate bg-black/20 p-1 rounded">
                       {item.utm}
                     </div>
                   </div>
                 ))}
               </div>
            </div>

            {/* Smart Insights */}
            <div className="bg-indigo-600 text-white rounded-xl p-5 shadow-lg shadow-indigo-200">
              <h3 className="font-bold mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4" /> AI 决策建议
              </h3>
              <div className="space-y-4 text-sm opacity-90">
                <div className="bg-white/10 p-3 rounded-lg border border-white/10">
                  <div className="font-bold text-yellow-300 mb-1">🔴 建议关停：百度搜索</div>
                  <p className="text-xs leading-relaxed">
                    当前 ROI 为 0.8，每投入 100 元亏损 20 元。建议将预算迁移至抖音直播间。
                  </p>
                </div>
                 <div className="bg-white/10 p-3 rounded-lg border border-white/10">
                  <div className="font-bold text-green-300 mb-1">🟢 机会：抖音直播间</div>
                  <p className="text-xs leading-relaxed">
                    ROI 稳定在 3.5 以上，且流量仍在增长。建议增加 20% 预算以测试流量天花板。
                  </p>
                </div>
              </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}

function KPICard({ title, value, prefix, trend, isGood, icon, desc }) {
  return (
    <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <div className="p-2 bg-slate-50 rounded-lg">{icon}</div>
        <div className={`flex items-center text-xs font-bold ${
          isGood ? (trend.includes('+') ? 'text-green-600' : 'text-red-600') 
                 : (trend.includes('+') ? 'text-red-600' : 'text-green-600')
        }`}>
          {trend.includes('+') ? <ArrowUpRight className="w-3 h-3 mr-0.5" /> : <ArrowDownRight className="w-3 h-3 mr-0.5" />}
          {trend}
        </div>
      </div>
      <div>
        <h3 className="text-2xl font-black text-slate-900 tracking-tight">
          <span className="text-lg font-medium text-slate-400 mr-0.5">{prefix}</span>
          {value}
        </h3>
        <p className="text-sm font-bold text-slate-600 mt-1">{title}</p>
        <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
      </div>
    </div>
  );
}