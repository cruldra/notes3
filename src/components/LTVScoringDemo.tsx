import React, { useState, useEffect } from 'react';
import { 
  Users, 
  TrendingUp, 
  DollarSign, 
  Activity, 
  Briefcase, 
  Clock, 
  MessageCircle, 
  Target, 
  Zap,
  Layers,
  ShoppingBag,
  Award,
  ArrowRight
} from 'lucide-react';

// --- Configuration ---
// 模拟职业对 LTV 的权重
const JOB_WEIGHTS = {
  'student': { label: '在校学生', score: 20, salary: 'Low' },
  'freelance': { label: '自由职业', score: 50, salary: 'Mid' },
  'manager': { label: '产品/运营经理', score: 75, salary: 'High' },
  'executive': { label: '企业高管/老板', score: 95, salary: 'Very High' }
};

// 模拟互动行为对 Intent (S/A/B) 的权重
const BEHAVIOR_WEIGHTS = {
  'silent': { label: '仅已读，无回复', score: 10 },
  'passive': { label: '被动回复 (嗯/哦)', score: 40 },
  'active': { label: '主动提问 (价格/细节)', score: 80 },
  'urgent': { label: '急切追问 (催单/付款)', score: 95 }
};

// 策略矩阵配置
const STRATEGY_MATRIX = {
  'High_High': {
    segment: '💎 核心金矿用户',
    salesScript: '话术策略：推高价 VIP 1v1 服务。强调“节省时间”、“极致服务”和“保姆级交付”，不谈性价比，谈ROI。',
    marketingAction: '动作：加入 [高净值_Lookalike] 种子包，推送 VIP 专属权益广告。',
    color: 'bg-purple-100 text-purple-700 border-purple-300'
  },
  'High_Low': {
    segment: '💤 沉睡富豪用户',
    salesScript: '话术策略：高价值内容种草。发送行业深度报告或大咖案例，建立专业信任，切勿频繁骚扰。',
    marketingAction: '动作：加入 [再营销_Retargeting] 列表，通过朋友圈展示品牌实力。',
    color: 'bg-blue-100 text-blue-700 border-blue-300'
  },
  'Low_High': {
    segment: '🔥 价格敏感活跃用户',
    salesScript: '话术策略：推分期/团购/基础版。强调“性价比”、“限时折扣”和“副业回本”，通过紧迫感逼单。',
    marketingAction: '动作：推送 [限时优惠券] 短信，引导至拼团页面。',
    color: 'bg-orange-100 text-orange-700 border-orange-300'
  },
  'Low_Low': {
    segment: '🍂 边缘流失用户',
    salesScript: '话术策略：自动化低频触达。仅在在大促节点群发通用通知，不占用真人销售精力。',
    marketingAction: '动作：移入 [公海池] 或 [低优先级队列]。',
    color: 'bg-slate-100 text-slate-600 border-slate-300'
  }
};

export default function LTVScoringDemo() {
  // --- User Profile State ---
  const [profile, setProfile] = useState({
    name: '张三',
    job: 'manager',
    budget: 5000,
    history: 0, // 历史消费额
    interaction: 'active'
  });

  // --- Calculated Metrics ---
  const [ltvScore, setLtvScore] = useState(0);
  const [ltvLevel, setLtvLevel] = useState('Low'); // High, Mid, Low
  
  const [intentScore, setIntentScore] = useState(0);
  const [intentLevel, setIntentLevel] = useState('Low'); // High (S), Mid (A), Low (B)

  const [strategy, setStrategy] = useState(null);

  // --- Calculation Engine ---
  useEffect(() => {
    // 1. Calculate LTV (Ability to Pay)
    // Formula: Job Base Score + (Budget / 100) + (History / 50)
    let lScore = JOB_WEIGHTS[profile.job].score;
    lScore += Math.min(profile.budget / 100, 30); // Cap budget impact
    if (profile.history > 0) lScore += 20; // Proven payer bonus
    lScore = Math.min(Math.round(lScore), 100);
    
    setLtvScore(lScore);
    const lLevel = lScore >= 70 ? 'High' : lScore >= 40 ? 'Mid' : 'Low';
    setLtvLevel(lLevel);

    // 2. Calculate Intent (Willingness to Pay)
    // Formula: Behavior Score
    let iScore = BEHAVIOR_WEIGHTS[profile.interaction].score;
    setIntentScore(iScore);
    const iLevel = iScore >= 70 ? 'High' : iScore >= 40 ? 'Mid' : 'Low';
    setIntentLevel(iLevel);

    // 3. Matrix Routing
    // Simplified to High vs Low for the 2x2 matrix demo
    const matrixKey = `${lScore >= 60 ? 'High' : 'Low'}_${iScore >= 60 ? 'High' : 'Low'}`;
    setStrategy(STRATEGY_MATRIX[matrixKey]);

  }, [profile]);

  // --- Helpers ---
  const handleProfileChange = (field, value) => {
    setProfile(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="text-indigo-600" />
              LTV 价值评分系统 <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full uppercase tracking-wide">Customer Value Engine</span>
            </h1>
            <p className="text-slate-500 mt-1 text-sm">
              基于“消费能力 (LTV)”与“购买意向 (Intent)”的双维度分层 • 指导差异化营销
            </p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: User Profile Editor */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full">
              <div className="flex items-center gap-2 mb-6 pb-4 border-b border-slate-100">
                <Users className="w-5 h-5 text-slate-400" />
                <h2 className="font-bold text-slate-700">客户画像模拟 (User Profile)</h2>
              </div>
              
              <div className="space-y-6">
                
                {/* LTV Factors Section */}
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-indigo-500 uppercase tracking-wider flex items-center gap-1">
                    <DollarSign className="w-3 h-3" /> LTV 影响因子 (消费力)
                  </h3>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">职业/身份</label>
                    <select 
                      value={profile.job}
                      onChange={(e) => handleProfileChange('job', e.target.value)}
                      className="w-full p-2 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                      {Object.entries(JOB_WEIGHTS).map(([key, val]) => (
                        <option key={key} value={key}>{val.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">填写的预算 (元)</label>
                    <input 
                      type="range" min="0" max="10000" step="500"
                      value={profile.budget}
                      onChange={(e) => handleProfileChange('budget', Number(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>¥0</span>
                      <span className="font-bold text-indigo-600">¥{profile.budget}</span>
                      <span>¥10k+</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <input 
                      type="checkbox" 
                      id="historyCheck"
                      checked={profile.history > 0}
                      onChange={(e) => handleProfileChange('history', e.target.checked ? 1 : 0)}
                      className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
                    />
                    <label htmlFor="historyCheck" className="text-sm text-slate-700 select-none">
                      有历史付费记录 (老学员)
                    </label>
                  </div>
                </div>

                <div className="border-t border-slate-100 my-4"></div>

                {/* Intent Factors Section */}
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-orange-500 uppercase tracking-wider flex items-center gap-1">
                    <Target className="w-3 h-3" /> Intent 影响因子 (意愿度)
                  </h3>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">互动行为特征</label>
                    <div className="space-y-2">
                      {Object.entries(BEHAVIOR_WEIGHTS).map(([key, val]) => (
                        <div 
                          key={key}
                          onClick={() => handleProfileChange('interaction', key)}
                          className={`p-2 rounded-lg border text-xs cursor-pointer transition-all flex justify-between items-center
                            ${profile.interaction === key 
                              ? 'bg-orange-50 border-orange-400 text-orange-800 font-bold' 
                              : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}
                          `}
                        >
                          <span>{val.label}</span>
                          {profile.interaction === key && <CheckCircleIcon />}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

              </div>
            </div>
          </div>

          {/* Center: Calculation Visualization */}
          <div className="lg:col-span-4 space-y-4">
            
            {/* Score Cards */}
            <div className="grid grid-cols-1 gap-4">
              
              {/* LTV Card */}
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-3 opacity-10">
                  <Briefcase className="w-20 h-20 text-indigo-600" />
                </div>
                <h3 className="text-sm font-bold text-slate-500 uppercase">LTV 价值分 (消费潜力)</h3>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-4xl font-black text-indigo-600">{ltvScore}</span>
                  <span className="text-sm text-slate-400">/ 100</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full mt-4 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ${ltvScore > 60 ? 'bg-indigo-600' : 'bg-slate-400'}`}
                    style={{ width: `${ltvScore}%` }}
                  ></div>
                </div>
                <div className="mt-2 flex gap-2">
                   {ltvScore >= 60 && <Tag label="高净值" color="indigo" />}
                   {profile.history > 0 && <Tag label="复购潜力" color="blue" />}
                </div>
              </div>

              {/* Intent Card */}
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-3 opacity-10">
                  <Target className="w-20 h-20 text-orange-600" />
                </div>
                <h3 className="text-sm font-bold text-slate-500 uppercase">Intent 意向分 (S/A/B)</h3>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-4xl font-black text-orange-500">{intentScore}</span>
                  <span className="text-sm text-slate-400">/ 100</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full mt-4 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ${intentScore > 60 ? 'bg-orange-500' : 'bg-slate-400'}`}
                    style={{ width: `${intentScore}%` }}
                  ></div>
                </div>
                 <div className="mt-2 flex gap-2">
                   {intentScore >= 80 ? <Tag label="S量 (极热)" color="red" /> : intentScore >= 50 ? <Tag label="A量 (温热)" color="orange" /> : <Tag label="B量 (冷淡)" color="slate" />}
                </div>
              </div>
            </div>

            {/* Matrix Visualization */}
            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 h-[280px] flex flex-col items-center justify-center relative">
               <h3 className="absolute top-3 left-4 text-xs font-bold text-slate-400">策略四象限 (Strategy Matrix)</h3>
               
               {/* Y-Axis Label */}
               <div className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] font-bold text-indigo-500 tracking-widest">
                  HIGH LTV ⬆
               </div>
               {/* X-Axis Label */}
               <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] font-bold text-orange-500 tracking-widest">
                  HIGH INTENT ➡
               </div>

               <div className="grid grid-cols-2 gap-2 w-full h-full p-6">
                 {/* Q2: High LTV, Low Intent */}
                 <MatrixCell 
                   active={ltvScore >= 60 && intentScore < 60} 
                   label="💤 沉睡富豪" 
                   desc="高潜/待激活"
                   color="bg-blue-100 border-blue-300" 
                 />
                 {/* Q1: High LTV, High Intent */}
                 <MatrixCell 
                   active={ltvScore >= 60 && intentScore >= 60} 
                   label="💎 核心金矿" 
                   desc="VIP/重点跟进"
                   color="bg-purple-100 border-purple-400" 
                 />
                 {/* Q3: Low LTV, Low Intent */}
                 <MatrixCell 
                   active={ltvScore < 60 && intentScore < 60} 
                   label="🍂 边缘流失" 
                   desc="低效/自动化"
                   color="bg-slate-100 border-slate-200" 
                 />
                 {/* Q4: Low LTV, High Intent */}
                 <MatrixCell 
                   active={ltvScore < 60 && intentScore >= 60} 
                   label="🔥 价格敏感" 
                   desc="走量/限时促"
                   color="bg-orange-100 border-orange-300" 
                 />
               </div>
            </div>

          </div>

          {/* Right: System Action Output */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800 h-full flex flex-col overflow-hidden">
               <div className="p-4 border-b border-slate-700 bg-slate-800/50 flex items-center justify-between">
                 <h2 className="font-bold flex items-center gap-2">
                   <Zap className="w-5 h-5 text-yellow-400" />
                   策略路由输出 (System Action)
                 </h2>
               </div>
               
               {strategy && (
                 <div className="p-6 space-y-8 animate-in slide-in-from-right-8 duration-500">
                   
                   {/* Segment Badge */}
                   <div className={`p-3 rounded-lg border-l-4 text-sm font-bold shadow-sm ${strategy.color}`}>
                     {strategy.segment}
                   </div>

                   {/* Sales Script */}
                   <div className="space-y-2">
                     <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase">
                       <MessageCircle className="w-3 h-3" /> AI 推荐销售策略
                     </div>
                     <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-sm leading-relaxed text-slate-300">
                       {strategy.salesScript}
                     </div>
                   </div>

                   {/* Marketing Action */}
                   <div className="space-y-2">
                     <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase">
                       <Activity className="w-3 h-3" /> 自动化运营动作
                     </div>
                     <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-sm leading-relaxed text-slate-300 flex items-start gap-3">
                       <div className="mt-1 w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                       {strategy.marketingAction}
                     </div>
                   </div>

                   {/* Business Value Highlight */}
                   <div className="pt-4 border-t border-slate-800 text-xs text-slate-500 italic text-center">
                     *该策略预计可提升 {ltvScore >= 60 ? '客单价 (AOV) 30%' : '转化率 (CVR) 15%'}
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

// --- Helper Components ---

function Tag({ label, color }) {
  const colors = {
    indigo: 'bg-indigo-100 text-indigo-700',
    blue: 'bg-blue-100 text-blue-700',
    red: 'bg-red-100 text-red-700',
    orange: 'bg-orange-100 text-orange-700',
    slate: 'bg-slate-100 text-slate-600'
  };
  return (
    <span className={`px-2 py-1 rounded text-[10px] font-bold ${colors[color]}`}>
      {label}
    </span>
  );
}

function CheckCircleIcon() {
  return (
    <div className="w-4 h-4 rounded-full bg-indigo-600 flex items-center justify-center">
      <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
      </svg>
    </div>
  );
}

function MatrixCell({ active, label, desc, color }) {
  return (
    <div className={`
      rounded-lg border-2 flex flex-col items-center justify-center text-center transition-all duration-500
      ${active ? `${color} scale-105 shadow-md z-10` : 'bg-slate-50 border-slate-100 opacity-40 grayscale'}
    `}>
      <div className="text-sm font-bold">{label}</div>
      <div className="text-[10px] opacity-80">{desc}</div>
    </div>
  );
}