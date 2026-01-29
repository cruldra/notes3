import React, { useState } from 'react';
import { 
  BarChart2, 
  AlertTriangle, 
  CheckCircle, 
  Download, 
  Zap, 
  Users, 
  TrendingUp, 
  TrendingDown, 
  Search,
  Filter,
  PieChart,
  Layout,
  ArrowRight
} from 'lucide-react';

// --- Mock Data ---

// 销售团队数据
const SALES_DATA = [
  { id: 1, name: '李明 (Top Sales)', group: '第10期-A组', completeness: 88, users: 120, riskUsers: 5, status: 'good' },
  { id: 2, name: '张伟', group: '第10期-B组', completeness: 72, users: 115, riskUsers: 12, status: 'normal' },
  { id: 3, name: '陈静', group: '第10期-A组', completeness: 65, users: 98, riskUsers: 20, status: 'normal' },
  { id: 4, name: '王强', group: '第10期-C组', completeness: 45, users: 130, riskUsers: 85, status: 'critical' }, // 模拟低分案例
  { id: 5, name: '刘洋', group: '第10期-B组', completeness: 92, users: 105, riskUsers: 2, status: 'good' },
];

// 字段缺失分析数据 (用于下钻)
const FIELD_ANALYSIS_DATA = {
  default: [
    { field: '预算范围 (Budget)', missingRate: 15, importance: 'High' },
    { field: '职业/行业 (Occupation)', missingRate: 20, importance: 'High' },
    { field: '痛点 (Pain Points)', missingRate: 10, importance: 'Critical' },
    { field: '所在城市 (City)', missingRate: 5, importance: 'Low' },
  ],
  critical_case: [ // 王强的数据
    { field: '预算范围 (Budget)', missingRate: 78, importance: 'High' },
    { field: '职业/行业 (Occupation)', missingRate: 65, importance: 'High' },
    { field: '痛点 (Pain Points)', missingRate: 40, importance: 'Critical' },
    { field: '年龄 (Age)', missingRate: 10, importance: 'Medium' },
  ]
};

// 趋势数据
const TREND_DATA = [
  { day: 'Day 1', score: 20 },
  { day: 'Day 2', score: 35 },
  { day: 'Day 3', score: 45 },
  { day: 'Day 4', score: 55 },
  { day: 'Day 5', score: 68 }, // 理想曲线
];

export default function CompletenessDashboard() {
  const [selectedRep, setSelectedRep] = useState(null);
  const [triggeringAction, setTriggeringAction] = useState(false);
  const [actionCompleted, setActionCompleted] = useState(false);

  // 计算全局指标
  const avgCompleteness = Math.round(SALES_DATA.reduce((acc, curr) => acc + curr.completeness, 0) / SALES_DATA.length);
  const totalRiskUsers = SALES_DATA.reduce((acc, curr) => acc + curr.riskUsers, 0);

  // 获取当前展示的字段分析数据
  const currentFieldData = selectedRep && selectedRep.id === 4 ? FIELD_ANALYSIS_DATA.critical_case : FIELD_ANALYSIS_DATA.default;

  const handleAction = () => {
    setTriggeringAction(true);
    setTimeout(() => {
      setTriggeringAction(false);
      setActionCompleted(true);
      // 3秒后重置状态，方便演示
      setTimeout(() => setActionCompleted(false), 3000);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg text-white">
              <Layout className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">画像质量治理驾驶舱</h1>
              <p className="text-sm text-slate-500">Data Completeness Dashboard • 需求编号 17.1</p>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 text-slate-600 rounded-lg hover:bg-slate-50 text-sm font-medium">
              <Download className="w-4 h-4" /> 导出报表
            </button>
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm">
              <span>数据更新时间: 今日 09:00</span>
            </div>
          </div>
        </header>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <KPICard 
            title="全局画像完整度" 
            value={`${avgCompleteness}%`} 
            trend="+5%" 
            trendUp={true} 
            icon={<PieChart className="w-5 h-5 text-indigo-600" />} 
            color="indigo"
          />
          <KPICard 
            title="高危缺失客户数" 
            subtitle="完整度 < 60%"
            value={totalRiskUsers} 
            trend="+12" 
            trendUp={false} 
            icon={<AlertTriangle className="w-5 h-5 text-red-600" />} 
            color="red"
            alert={true}
          />
          <KPICard 
            title="关键字段覆盖率" 
            subtitle="预算/职业/痛点"
            value="76%" 
            trend="+2%" 
            trendUp={true} 
            icon={<CheckCircle className="w-5 h-5 text-green-600" />} 
            color="green"
          />
          <KPICard 
            title="今日待补全任务" 
            value="142" 
            trend="-8" 
            trendUp={true} 
            icon={<Zap className="w-5 h-5 text-yellow-600" />} 
            color="yellow"
          />
        </div>

        {/* Main Content Split */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: Sales Team Ranking */}
          <div className="lg:col-span-7 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col">
            <div className="p-5 border-b border-slate-100 flex justify-between items-center">
              <h2 className="font-bold text-slate-800 flex items-center gap-2">
                <Users className="w-5 h-5 text-slate-500" />
                销售团队数据质量排名
              </h2>
              <div className="flex gap-2">
                 <button className="p-1.5 hover:bg-slate-100 rounded text-slate-400"><Search className="w-4 h-4" /></button>
                 <button className="p-1.5 hover:bg-slate-100 rounded text-slate-400"><Filter className="w-4 h-4" /></button>
              </div>
            </div>
            
            <div className="flex-1 overflow-auto">
              <table className="w-full text-left">
                <thead className="bg-slate-50 text-xs text-slate-500 uppercase font-medium">
                  <tr>
                    <th className="px-6 py-3">销售人员</th>
                    <th className="px-6 py-3">所属班级</th>
                    <th className="px-6 py-3">画像完整度</th>
                    <th className="px-6 py-3">状态</th>
                    <th className="px-6 py-3 text-right">操作</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm">
                  {SALES_DATA.map((item) => (
                    <tr 
                      key={item.id} 
                      onClick={() => setSelectedRep(item)}
                      className={`
                        cursor-pointer transition-colors hover:bg-slate-50
                        ${selectedRep?.id === item.id ? 'bg-indigo-50 hover:bg-indigo-50' : ''}
                      `}
                    >
                      <td className="px-6 py-4 font-medium text-slate-900">{item.name}</td>
                      <td className="px-6 py-4 text-slate-500">{item.group}</td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <span className={`font-bold ${getColorClass(item.completeness).text}`}>
                            {item.completeness}%
                          </span>
                          <div className="w-24 h-2 bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${getColorClass(item.completeness).bg}`} 
                              style={{ width: `${item.completeness}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        {item.completeness < 60 ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 animate-pulse">
                            <AlertTriangle className="w-3 h-3" /> 预警
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            正常
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <ArrowRight className={`w-4 h-4 ml-auto ${selectedRep?.id === item.id ? 'text-indigo-600' : 'text-slate-300'}`} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right: Analysis & Action Panel */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Detail Card */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 h-full flex flex-col">
              {!selectedRep ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-400 py-12">
                  <BarChart2 className="w-16 h-16 mb-4 opacity-20" />
                  <p>请点击左侧销售人员查看详细诊断</p>
                </div>
              ) : (
                <>
                  <div className="flex justify-between items-start mb-6 border-b border-slate-100 pb-4">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">
                        {selectedRep.name} 的数据诊断
                      </h3>
                      <p className="text-sm text-slate-500 mt-1">
                        所属: {selectedRep.group} | 客户数: {selectedRep.users}
                      </p>
                    </div>
                    <div className={`text-2xl font-black ${getColorClass(selectedRep.completeness).text}`}>
                      {selectedRep.completeness}%
                    </div>
                  </div>

                  {selectedRep.completeness < 60 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex gap-3 items-start">
                      <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-bold text-red-800">画像质量严重不足</h4>
                        <p className="text-xs text-red-700 mt-1">
                          该组有 {selectedRep.riskUsers} 名客户缺少关键决策字段，会导致 AI 无法生成精准的逼单话术。
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="space-y-4 mb-8 flex-1">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                      高频缺失字段 (Missing Fields)
                    </h4>
                    {currentFieldData.map((field, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-700">{field.field}</span>
                          <span className="font-mono text-slate-500">{field.missingRate}% 缺失</span>
                        </div>
                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${field.missingRate > 50 ? 'bg-red-500' : field.missingRate > 20 ? 'bg-orange-400' : 'bg-blue-400'}`} 
                            style={{ width: `${field.missingRate}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Governance Action */}
                  <div className="mt-auto pt-4 border-t border-slate-100">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                      治理建议 / Actions
                    </h4>
                    {actionCompleted ? (
                      <div className="bg-green-50 text-green-700 p-3 rounded-lg text-sm flex items-center justify-center gap-2 font-medium animate-in fade-in zoom-in">
                        <CheckCircle className="w-5 h-5" />
                        AI 自动追问任务已下发
                      </div>
                    ) : (
                      <button 
                        onClick={handleAction}
                        disabled={triggeringAction}
                        className={`
                          w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all
                          ${selectedRep.completeness < 60 
                            ? 'bg-red-600 hover:bg-red-700 text-white shadow-red-200 shadow-lg' 
                            : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 shadow-lg'}
                          ${triggeringAction ? 'opacity-80 cursor-wait' : ''}
                        `}
                      >
                        {triggeringAction ? (
                          <>
                            <Zap className="w-4 h-4 animate-spin" />
                            正在生成追问话术...
                          </>
                        ) : (
                          <>
                            <Zap className="w-4 h-4" />
                            {selectedRep.completeness < 60 ? '一键触发 AI 追问 (补全信息)' : '发送数据质量提醒'}
                          </>
                        )}
                      </button>
                    )}
                     <p className="text-xs text-center text-slate-400 mt-2">
                       {selectedRep.completeness < 60 
                        ? '系统将针对缺失字段，自动向客户发送 17.2 追问话术' 
                        : '将通过企业微信提醒销售注意数据录入'}
                     </p>
                  </div>
                </>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// Helper Components & Functions
function KPICard({ title, value, trend, trendUp, icon, color, subtitle, alert }) {
  const colorMap = {
    indigo: 'bg-indigo-50 text-indigo-600',
    red: 'bg-red-50 text-red-600',
    green: 'bg-green-50 text-green-600',
    yellow: 'bg-yellow-50 text-yellow-600',
  };

  return (
    <div className={`bg-white p-5 rounded-xl shadow-sm border ${alert ? 'border-red-200 ring-2 ring-red-50' : 'border-slate-200'}`}>
      <div className="flex justify-between items-start mb-2">
        <div className={`p-2 rounded-lg ${colorMap[color]}`}>
          {icon}
        </div>
        <div className={`flex items-center text-xs font-bold ${trendUp ? 'text-green-600' : 'text-red-500'}`}>
          {trendUp ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
          {trend}
        </div>
      </div>
      <div className="mt-2">
        <h3 className="text-2xl font-black text-slate-900">{value}</h3>
        <p className="text-sm font-medium text-slate-500">{title}</p>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function getColorClass(score) {
  if (score >= 80) return { text: 'text-green-600', bg: 'bg-green-500' };
  if (score >= 60) return { text: 'text-yellow-600', bg: 'bg-yellow-500' };
  return { text: 'text-red-600', bg: 'bg-red-500' };
}