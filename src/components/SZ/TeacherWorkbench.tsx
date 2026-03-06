import React, { useState } from 'react';
import { 
  AlertTriangle, Bell, Search, UserCheck, 
  Users, Activity, BrainCircuit, ShieldAlert,
  Calendar, FileText, ChevronRight, MessageSquare,
  Coffee, BookOpen, Clock, X, CheckCircle,
  BarChart3, PieChart, TrendingUp, Download, Filter, MoreVertical,
  Library, Lightbulb, Sparkles, Send, Target
} from 'lucide-react';

export default function TeacherWorkbench() {
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [activeTab, setActiveTab] = useState('ideological'); // 默认展示新增的导师端核心功能

  const [prepKeyword, setPrepKeyword] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiCases, setAiCases] = useState([
    {
      id: 1,
      title: "新质生产力与青年担当",
      majors: ["计算机", "软件工程"],
      content: "结合近期大模型与国产芯片突破热点，探讨数字经济转型下，工科生如何将个人职业规划与国家科技自立自强战略结合。",
      match: "98%",
      status: "published"
    },
    {
      id: 2,
      title: "平台经济下的劳动者权益",
      majors: ["法学", "工商管理"],
      content: "以外卖骑手算法困境为例，从马克思主义政治经济学角度剖析资本逻辑与科技向善，引导学生关注社会公平正义。",
      match: "95%",
      status: "draft"
    }
  ]);

  const handleGenerateCase = () => {
    if (!prepKeyword.trim()) return;
    setIsGenerating(true);
    setTimeout(() => {
      setAiCases([
        {
          id: Date.now(),
          title: `基于“${prepKeyword}”的跨学科洞察`,
          majors: ["全专业通用", "结合各院系特色"],
          content: `AI已根据您的关键词【${prepKeyword}】检索最新央媒时政语料，并结合了本校学生近期关注度最高的情绪痛点，自动生成了3组互动式思政问答卡片，可直接推送到学生的AI助手端。`,
          match: "99%",
          status: "draft"
        },
        ...aiCases
      ]);
      setIsGenerating(false);
      setPrepKeyword('');
    }, 1500);
  };

  // 模拟PRD中要求的四级预警数据
  const warningList = [
    {
      id: 1,
      name: "李明",
      idCard: "2023040122",
      major: "电子信息工程 23级",
      level: "red",
      levelText: "高危心理干预",
      reason: "AI助手监测到高频抑郁词汇 + 连续3天未归寝 + 缺课",
      time: "10分钟前",
      status: "pending"
    },
    {
      id: 2,
      name: "王晓宇",
      idCard: "2024010509",
      major: "软件工程 24级",
      level: "orange",
      levelText: "学业与行为异常",
      reason: "期中测试3科不及格 + 校园卡连续一周每日消费低于10元",
      time: "2小时前",
      status: "pending"
    }
  ];

  // 画像数据
  const studentProfileData = {
    name: "李明",
    idCard: "2023040122",
    major: "电子信息工程 23级2班",
    avatar: "李",
    tags: ["贫困生库", "性格内向", "近期成绩下滑"],
    dimensions: {
      aiAnalysis: {
        sentiment: "重度焦虑/抑郁倾向",
        keywords: ["失眠", "挂科", "没有意义", "想放弃"],
        recentChat: "昨晚凌晨02:30向AI助手咨询『学校周边有哪里可以看心理医生』"
      },
      academic: {
        gpa: "2.1 (年级后15%)",
        attendance: "本周缺勤 3 节 (高数、大物)"
      },
      behavior: {
        dorm: "南区4栋302 (本周异常：连续3天超晚归或未归)",
        canteen: "近期消费急剧减少，日均饮食开销 < 15元",
        network: "校园网深夜活跃度极高 (凌晨1点-4点)"
      }
    }
  };

  const getLevelColor = (level) => {
    switch(level) {
      case 'red': return 'bg-red-50 text-red-700 border-red-200';
      case 'orange': return 'bg-orange-50 text-orange-700 border-orange-200';
      case 'yellow': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'blue': return 'bg-blue-50 text-blue-700 border-blue-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#f4f7f9] text-slate-800 font-sans overflow-hidden">
      
      {/* 侧边栏 */}
      <div className="w-64 bg-slate-900 text-slate-300 flex flex-col flex-shrink-0 z-20 shadow-xl">
        <div className="h-16 flex items-center px-6 bg-slate-950 border-b border-slate-800">
          <Library className="text-red-500 mr-3" size={24} />
          <span className="font-bold text-white text-lg tracking-wide">AI 思政与学工中枢</span>
        </div>
        
        <div className="p-4 flex-1 overflow-y-auto">
          <div className="text-xs text-slate-500 font-bold mb-3 tracking-wider uppercase mt-2">导师核心业务</div>
          <nav className="space-y-1.5 mb-8">
            <button 
              onClick={() => setActiveTab('ideological')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === 'ideological' ? 'bg-red-600/20 text-red-400 border border-red-500/30 shadow-inner' : 'hover:bg-slate-800 text-slate-400'}`}
            >
              <Lightbulb size={18} /> 思政教学管家
            </button>
            <button 
              onClick={() => setActiveTab('profile')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === 'profile' ? 'bg-red-600/20 text-red-400 border border-red-500/30 shadow-inner' : 'hover:bg-slate-800 text-slate-400'}`}
            >
              <Users size={18} /> 全景学生画像
            </button>
          </nav>

          <div className="text-xs text-slate-500 font-bold mb-3 tracking-wider uppercase">辅导员/预警业务</div>
          <nav className="space-y-1.5">
            <button 
              onClick={() => setActiveTab('warning')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === 'warning' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-inner' : 'hover:bg-slate-800 text-slate-400'}`}
            >
              <AlertTriangle size={18} /> 风险调度中心
            </button>
            <button 
              onClick={() => setActiveTab('intervention')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === 'intervention' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-inner' : 'hover:bg-slate-800 text-slate-400'}`}
            >
              <MessageSquare size={18} /> 干预追踪台账
            </button>
            <button 
              onClick={() => setActiveTab('report')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === 'report' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-inner' : 'hover:bg-slate-800 text-slate-400'}`}
            >
              <FileText size={18} /> 智能学情月报
            </button>
          </nav>
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center text-white font-bold shadow-md">刘</div>
            <div>
              <div className="text-sm font-bold text-white">刘导师</div>
              <div className="text-[11px] text-slate-400">马克思主义学院 / 辅导员</div>
            </div>
          </div>
        </div>
      </div>

      {/* 主内容区 */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        
        {/* 顶栏 */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center justify-between px-8 z-10 sticky top-0">
          <h1 className="text-xl font-bold text-slate-800 flex items-center gap-2">
            {activeTab === 'ideological' && <><Lightbulb className="text-red-500" size={24}/> AI 思政教学与思想动态管家</>}
            {activeTab === 'warning' && <><AlertTriangle className="text-blue-500" size={24}/> 预警分级调度大厅</>}
            {activeTab === 'profile' && <><Users className="text-indigo-500" size={24}/> 全景学生数字画像库</>}
            {activeTab === 'intervention' && <><MessageSquare className="text-green-500" size={24}/> AI 辅助干预追踪台账</>}
            {activeTab === 'report' && <><FileText className="text-purple-500" size={24}/> 学工智能数据分析月报</>}
          </h1>
          <div className="flex items-center gap-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
              <input 
                type="text" 
                placeholder="搜索学生、教案、热点..." 
                className="pl-9 pr-4 py-2 bg-slate-100 border-none rounded-full text-sm w-64 focus:ring-2 focus:ring-red-100 outline-none transition-all"
              />
            </div>
            <button className="relative text-slate-500 hover:text-red-600 transition-colors">
              <Bell size={20} />
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
            </button>
          </div>
        </header>

        {/* 核心内容流 */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          
          {/* ================= 新增：导师专属思政板块 ================= */}
          {activeTab === 'ideological' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 max-w-7xl mx-auto space-y-6">
              
              {/* 顶部大盘指标 */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex flex-col justify-center">
                  <div className="text-sm text-slate-500 font-medium mb-1 flex items-center gap-1.5"><Target size={16} className="text-red-500"/>思政图谱覆盖率</div>
                  <div className="text-3xl font-black text-slate-800">94.2%</div>
                  <div className="w-full bg-slate-100 h-1.5 rounded-full mt-3 overflow-hidden"><div className="bg-red-500 h-full w-[94%]"></div></div>
                </div>
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex flex-col justify-center">
                  <div className="text-sm text-slate-500 font-medium mb-1 flex items-center gap-1.5"><BrainCircuit size={16} className="text-indigo-500"/>AI定制案例总数</div>
                  <div className="text-3xl font-black text-slate-800">1,284 <span className="text-xs text-green-500 font-bold ml-1">↑ 12本周</span></div>
                </div>
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex flex-col justify-center">
                  <div className="text-sm text-slate-500 font-medium mb-1 flex items-center gap-1.5"><MessageSquare size={16} className="text-blue-500"/>学生互动问答量</div>
                  <div className="text-3xl font-black text-slate-800">45.2k <span className="text-xs text-slate-400 font-normal ml-1">次调用</span></div>
                </div>
                <div className="bg-gradient-to-br from-red-600 to-rose-700 p-5 rounded-2xl shadow-md text-white flex flex-col justify-center relative overflow-hidden">
                  <div className="absolute -right-4 -bottom-4 opacity-10"><Sparkles size={80}/></div>
                  <div className="text-sm text-red-100 font-medium mb-1 relative z-10">综合学情健康度评估</div>
                  <div className="text-3xl font-black relative z-10">优良</div>
                  <div className="text-xs text-red-50 mt-1 relative z-10">基于情绪分析与学习进度研判</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* 左侧：AI 一键备课与案例分发 (占2列) */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50 rounded-bl-full -z-10"></div>
                    <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
                      <Sparkles className="text-indigo-500" size={20}/> 
                      智能教案生成器 (千人千面)
                    </h2>
                    <p className="text-sm text-slate-500 mb-5">输入近期新闻热点或理论关键词，AI 将自动融合学校知识图谱，为您生成针对不同专业学生的个性化思政学习卡片。</p>
                    
                    <div className="flex gap-3 mb-6">
                      <div className="relative flex-1">
                        <input 
                          type="text" 
                          value={prepKeyword}
                          onChange={(e) => setPrepKeyword(e.target.value)}
                          placeholder="例如：人工智能安全、新质生产力、反内卷..." 
                          className="w-full pl-4 pr-10 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400 outline-none transition-all shadow-inner"
                          onKeyDown={(e) => e.key === 'Enter' && handleGenerateCase()}
                        />
                      </div>
                      <button 
                        onClick={handleGenerateCase}
                        disabled={isGenerating || !prepKeyword.trim()}
                        className="px-6 py-3 bg-indigo-600 text-white font-bold rounded-xl text-sm hover:bg-indigo-700 transition-colors shadow-md disabled:bg-slate-300 flex items-center gap-2"
                      >
                        {isGenerating ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> : <BrainCircuit size={18}/>}
                        {isGenerating ? 'AI 生成中...' : '一键生成'}
                      </button>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">最新生成的专属案例</h3>
                      {aiCases.map(item => (
                        <div key={item.id} className="p-4 rounded-xl border border-slate-100 bg-slate-50/50 hover:bg-indigo-50/30 transition-colors group">
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="font-bold text-slate-800 text-base">{item.title}</h4>
                            <span className={`text-[10px] px-2 py-1 rounded-md font-bold ${item.status === 'published' ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'}`}>
                              {item.status === 'published' ? '已全校推送' : '草稿待审核'}
                            </span>
                          </div>
                          <div className="flex gap-2 mb-3">
                            {item.majors.map((m, i) => <span key={i} className="text-[10px] bg-white border border-slate-200 text-slate-600 px-2 py-0.5 rounded shadow-sm">{m}</span>)}
                            <span className="text-[10px] bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded font-medium flex items-center gap-1">图谱匹配度 {item.match}</span>
                          </div>
                          <p className="text-sm text-slate-600 leading-relaxed mb-4">{item.content}</p>
                          <div className="flex gap-2 opacity-100 md:opacity-0 group-hover:opacity-100 transition-opacity">
                            <button className="text-xs flex items-center gap-1 px-3 py-1.5 bg-white border border-slate-200 rounded-lg hover:text-indigo-600 hover:border-indigo-200 shadow-sm"><FileText size={14}/> 编辑教案</button>
                            <button className="text-xs flex items-center gap-1 px-3 py-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 shadow-sm"><Send size={14}/> 推送至学生 AI 助手</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 右侧：思想动态预警与词云监测 (占1列) */}
                <div className="space-y-6">
                  <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 h-full flex flex-col">
                    <h2 className="text-base font-bold text-slate-800 flex items-center gap-2 mb-1">
                      <Activity className="text-orange-500" size={18}/> 
                      实时思想动态监测雷达
                    </h2>
                    <p className="text-xs text-slate-500 mb-6">基于全校学生本月在 AI 助手的问答语义提取</p>

                    {/* 模拟词云区域 */}
                    <div className="flex-1 bg-slate-50 rounded-xl border border-slate-100 p-4 relative flex items-center justify-center min-h-[200px] mb-6 shadow-inner overflow-hidden">
                      <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-400 via-transparent to-transparent"></div>
                      <div className="relative w-full h-full flex items-center justify-center">
                        <span className="absolute text-xl font-bold text-red-500 top-[20%] left-[20%] animate-pulse cursor-pointer hover:scale-110 transition-transform">考研内卷</span>
                        <span className="absolute text-sm font-medium text-slate-600 top-[60%] left-[15%]">挂科焦虑</span>
                        <span className="absolute text-2xl font-black text-indigo-600 top-[40%] left-[40%] cursor-pointer hover:scale-110 transition-transform">就业出路</span>
                        <span className="absolute text-base font-bold text-orange-500 top-[30%] left-[65%]">人工智能替代</span>
                        <span className="absolute text-xs font-medium text-slate-400 top-[70%] left-[50%]">人际交往</span>
                        <span className="absolute text-lg font-bold text-blue-500 top-[65%] left-[70%] cursor-pointer hover:scale-110 transition-transform">四六级</span>
                        <span className="absolute text-sm font-medium text-emerald-500 top-[15%] left-[55%]">奖学金评定</span>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">异常聚集预警提示</h3>
                      <div className="bg-orange-50 border border-orange-100 p-3 rounded-lg flex items-start gap-2">
                        <AlertTriangle size={16} className="text-orange-500 flex-shrink-0 mt-0.5"/>
                        <p className="text-xs text-orange-800 leading-relaxed">
                          近期**“就业出路”**搜索频次环比上升 45%，主要集中在大三/大四工科院系。建议导师增设“职业规划与马克思主义劳动观”专题推送。
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* ================= 原有功能板块保留并适配 UI ================= */}
          {activeTab === 'warning' && (
             <div className="animate-in fade-in slide-in-from-bottom-2 max-w-5xl mx-auto">
              <div className="flex justify-between items-end mb-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-800">实时智能预警工单 (辅导员视图)</h2>
                  <p className="text-sm text-slate-500 mt-1">由 AI 中枢综合【教务、后勤、安防、心理】等子系统数据自动生成</p>
                </div>
              </div>
              <div className="space-y-4">
                {warningList.map((item) => (
                  <div key={item.id} className={`bg-white rounded-xl border p-5 flex items-start gap-6 transition-all hover:shadow-md cursor-pointer ${getLevelColor(item.level)} bg-opacity-30`} onClick={() => setSelectedStudent(item)}>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-bold text-slate-800">{item.name}</h3>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${getLevelColor(item.level)}`}>{item.levelText}</span>
                      </div>
                      <div className="bg-white/60 p-3 rounded-lg border border-white flex items-start gap-3">
                        <BrainCircuit className="text-indigo-500 mt-0.5" size={16} />
                        <div className="text-sm font-medium text-slate-700">{item.reason}</div>
                      </div>
                    </div>
                    <button className="py-2 px-4 bg-slate-800 text-white text-xs font-bold rounded-lg hover:bg-slate-700">处理工单</button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(activeTab === 'profile' || activeTab === 'intervention' || activeTab === 'report') && (
            <div className="animate-in fade-in flex flex-col items-center justify-center h-[60vh] text-center">
              <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-4 text-slate-400">
                <ShieldAlert size={32} />
              </div>
              <h3 className="text-xl font-bold text-slate-700 mb-2">该模块功能演示已在上一版本呈现</h3>
              <p className="text-slate-500 max-w-md text-sm">
                为突出本次“导师端（思政管家）”的核心诉求，画像与报表等模块已被折叠。您可以切换回“思政教学管家”体验 AI 一键备课功能。
              </p>
              <button onClick={() => setActiveTab('ideological')} className="mt-6 px-6 py-2 bg-red-600 text-white rounded-xl text-sm font-bold shadow-md hover:bg-red-700 transition-colors">
                返回思政教学管家
              </button>
            </div>
          )}

        </main>

        {/* 侧滑抽屉：预警详情弹窗 */}
        {selectedStudent && activeTab === 'warning' && (
          <div className="absolute inset-0 z-50 flex justify-end">
             <div className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm" onClick={() => setSelectedStudent(null)}></div>
             <div className="w-[500px] bg-white h-full shadow-2xl animate-in slide-in-from-right flex flex-col p-6 relative z-10">
                <h2 className="font-bold text-xl mb-4 text-slate-800">预警处置档案 - {selectedStudent.name}</h2>
                <div className="bg-slate-50 p-4 rounded-xl text-sm text-slate-600 border border-slate-200">
                  <p>您已调取该生的危险数据，建议结合左侧“思政管家”中的情绪词云，通过下发针对性学习任务进行隐性心理干预。</p>
                </div>
                <button onClick={() => setSelectedStudent(null)} className="mt-auto w-full py-3 bg-slate-800 text-white font-bold rounded-xl">关闭档案</button>
             </div>
          </div>
        )}

      </div>
    </div>
  );
}