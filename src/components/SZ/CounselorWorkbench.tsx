import React, { useState } from 'react';
import { 
  AlertTriangle, Bell, Search, UserCheck, 
  Users, Activity, BrainCircuit, ShieldAlert,
  Calendar, FileText, ChevronRight, MessageSquare,
  Coffee, BookOpen, Clock, X, CheckCircle,
  BarChart3, PieChart, TrendingUp, Download, Filter, MoreVertical
} from 'lucide-react';

export default function CounselorWorkbench() {
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [activeTab, setActiveTab] = useState('warning'); // 新增状态控制导航栏

  // 模拟PRD中要求的四级预警数据 (红、橙、黄、蓝)
  const warningList = [
    {
      id: 1,
      name: "李明",
      idCard: "2023040122",
      major: "电子信息工程 23级",
      level: "red", // 红牌预警
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
      level: "orange", // 橙牌预警
      levelText: "学业与行为异常",
      reason: "期中测试3科不及格 + 校园卡连续一周每日消费低于10元",
      time: "2小时前",
      status: "pending"
    },
    {
      id: 3,
      name: "陈辰",
      idCard: "2023080115",
      major: "汉语言文学 23级",
      level: "yellow", // 黄牌预警
      levelText: "社交与情绪低落",
      reason: "辅导猫请假频次过高 + 图书馆借阅量断崖下跌",
      time: "昨天 15:30",
      status: "processing"
    }
  ];

  // 模拟PRD中的 360度学生画像数据体系
  const studentProfileData = {
    name: "李明",
    idCard: "2023040122",
    major: "电子信息工程 23级2班",
    avatar: "李",
    tags: ["贫困生库", "性格内向", "近期成绩下滑"],
    
    // 多维度数据融合展示
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

  // 新增：全量学生数据（用于画像页面）
  const allStudents = [
    ...warningList,
    { id: 4, name: "赵雷", idCard: "2023040188", major: "电子信息工程 23级", level: "normal", status: "正常", avatar: "赵" },
    { id: 5, name: "林书", idCard: "2023080199", major: "汉语言文学 23级", level: "normal", status: "正常", avatar: "林" },
    { id: 6, name: "孙琪", idCard: "2024010577", major: "软件工程 24级", level: "normal", status: "正常", avatar: "孙" },
    { id: 7, name: "郭宇", idCard: "2024010588", major: "软件工程 24级", level: "normal", status: "正常", avatar: "郭" },
    { id: 8, name: "吴桐", idCard: "2023040199", major: "电子信息工程 23级", level: "normal", status: "正常", avatar: "吴" }
  ];

  // 新增：干预记录模拟数据
  const interventionRecords = [
    { id: 101, date: "2026-03-05 10:30", student: "王晓宇", trigger: "橙牌：学业与行为异常", action: "线下谈心", detail: "了解其因期中考试失利导致情绪低落，已联系专业课老师提供课后辅导，并对接资助中心发放临时餐补。", aiEval: "危机已初步缓解，建议后续持续关注其食堂消费记录及考勤情况。" },
    { id: 102, date: "2026-03-01 14:00", student: "张强", trigger: "红牌：高危心理干预", action: "转介心理中心", detail: "系统监测到严重自残倾向词汇，立即启动紧急预案，协同副书记共同面谈后，安全转介至校心理中心进行专业干预。", aiEval: "干预极其及时，符合学校标准处理流程，高危风险已有效转移至专业机构。" },
    { id: 103, date: "2026-02-28 09:15", student: "刘悦", trigger: "黄牌：社交与情绪低落", action: "家校协同", detail: "连续两周未参与任何集体活动，联系家长得知其家中突发变故。已协助其线上办理临时困难补助审批。", aiEval: "外部因素导致的情绪波动，已提供实质性支持，状态正在逐步恢复中。" }
  ];

  const getLevelColor = (level) => {
    switch(level) {
      case 'red': return 'bg-red-50 text-red-700 border-red-200';
      case 'orange': return 'bg-orange-50 text-orange-700 border-orange-200';
      case 'yellow': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'blue': return 'bg-blue-50 text-blue-700 border-blue-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const getLevelBadge = (level) => {
    switch(level) {
      case 'red': return 'bg-red-500';
      case 'orange': return 'bg-orange-500';
      case 'yellow': return 'bg-yellow-500';
      case 'blue': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  // 新增：动态获取画像数据（区分正常学生与异常学生）
  const getDimensions = (student) => {
    if (student.level === 'normal') {
      return {
        aiAnalysis: {
          sentiment: "平稳 / 积极乐观",
          keywords: ["考研资料", "六级报名", "社团活动", "图书馆"],
          recentChat: "下午14:00向AI助手咨询『计算机二级考试报名入口在哪里』"
        },
        academic: {
          gpa: "3.6 (年级前15%)",
          attendance: "本月全勤，无旷课记录"
        },
        behavior: {
          dorm: "南区4栋 (作息规律，无晚归记录)",
          canteen: "规律就餐，日均饮食开销 40-50元",
          network: "健康上网，无深夜沉迷特征"
        }
      };
    }
    return studentProfileData.dimensions; // 异常学生依然使用预警数据
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 text-slate-800 font-sans overflow-hidden">
      
      {/* 侧边栏 */}
      <div className="w-64 bg-slate-900 text-slate-300 flex flex-col flex-shrink-0">
        <div className="h-16 flex items-center px-6 bg-slate-950 border-b border-slate-800">
          <ShieldAlert className="text-blue-500 mr-3" size={24} />
          <span className="font-bold text-white text-lg tracking-wide">智慧学工中枢</span>
        </div>
        
        <div className="p-4">
          <div className="text-xs text-slate-500 font-semibold mb-4 tracking-wider">辅导员工作台</div>
          <nav className="space-y-2">
            <button 
              onClick={() => setActiveTab('warning')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${activeTab === 'warning' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/20' : 'hover:bg-slate-800 text-slate-300'}`}
            >
              <AlertTriangle size={18} /> 风险预警中心
            </button>
            <button 
              onClick={() => setActiveTab('profile')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${activeTab === 'profile' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/20' : 'hover:bg-slate-800 text-slate-300'}`}
            >
              <Users size={18} /> 学生数字画像
            </button>
            <button 
              onClick={() => setActiveTab('intervention')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${activeTab === 'intervention' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/20' : 'hover:bg-slate-800 text-slate-300'}`}
            >
              <MessageSquare size={18} /> AI 干预记录
            </button>
            <button 
              onClick={() => setActiveTab('report')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors ${activeTab === 'report' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/20' : 'hover:bg-slate-800 text-slate-300'}`}
            >
              <FileText size={18} /> 智能工作月报
            </button>
          </nav>
        </div>

        <div className="mt-auto p-4 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-white">王</div>
            <div>
              <div className="text-sm font-bold text-white">王辅导员</div>
              <div className="text-xs text-slate-400">信息工程学院</div>
            </div>
          </div>
        </div>
      </div>

      {/* 主内容区 */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        
        {/* 顶栏 */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 z-10">
          <h1 className="text-xl font-bold text-slate-800">
            {activeTab === 'warning' && '预警分级调度大厅'}
            {activeTab === 'profile' && '全景学生数字画像库'}
            {activeTab === 'intervention' && 'AI 辅助干预追踪记录'}
            {activeTab === 'report' && '学工智能数据分析月报'}
          </h1>
          <div className="flex items-center gap-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
              <input 
                type="text" 
                placeholder="搜索学号、姓名..." 
                className="pl-9 pr-4 py-2 bg-slate-100 border-none rounded-full text-sm w-64 focus:ring-2 focus:ring-blue-100 outline-none"
              />
            </div>
            <button className="relative text-slate-500 hover:text-blue-600">
              <Bell size={20} />
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
            </button>
          </div>
        </header>

        {/* 核心内容流 */}
        <main className="flex-1 overflow-y-auto p-8">
          
          {activeTab === 'warning' && (
            <div className="animate-in fade-in slide-in-from-bottom-2">
              {/* 数据概览卡片 */}
              <div className="grid grid-cols-4 gap-6 mb-8">
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center"><Users size={24}/></div>
                  <div>
                    <div className="text-2xl font-bold text-slate-800">218</div>
                    <div className="text-xs text-slate-500 mt-1">管辖学生总数</div>
                  </div>
                </div>
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center gap-4 relative overflow-hidden">
                  <div className="absolute right-0 top-0 w-2 h-full bg-red-500"></div>
                  <div className="w-12 h-12 rounded-xl bg-red-50 text-red-600 flex items-center justify-center animate-pulse"><AlertTriangle size={24}/></div>
                  <div>
                    <div className="text-2xl font-bold text-red-600">3</div>
                    <div className="text-xs text-slate-500 mt-1">待处理高危预警</div>
                  </div>
                </div>
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-orange-50 text-orange-600 flex items-center justify-center"><Activity size={24}/></div>
                  <div>
                    <div className="text-2xl font-bold text-slate-800">12</div>
                    <div className="text-xs text-slate-500 mt-1">本周异常行为</div>
                  </div>
                </div>
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-green-50 text-green-600 flex items-center justify-center"><UserCheck size={24}/></div>
                  <div>
                    <div className="text-2xl font-bold text-slate-800">15</div>
                    <div className="text-xs text-slate-500 mt-1">已干预解除危机</div>
                  </div>
                </div>
              </div>

              <div className="flex justify-between items-end mb-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-800">实时智能预警工单</h2>
                  <p className="text-sm text-slate-500 mt-1">由 AI 中枢综合【教务、后勤、安防、心理】等子系统数据自动生成</p>
                </div>
                <div className="flex gap-2">
                  <button className="px-3 py-1.5 text-xs font-medium bg-white border border-slate-200 rounded-lg shadow-sm">全部级别</button>
                  <button className="px-3 py-1.5 text-xs font-medium bg-red-50 text-red-600 border border-red-100 rounded-lg shadow-sm">仅看红牌 (1)</button>
                </div>
              </div>

              {/* 预警列表 */}
              <div className="space-y-4">
                {warningList.map((item) => (
                  <div 
                    key={item.id} 
                    className={`bg-white rounded-xl border p-5 flex items-start gap-6 transition-all hover:shadow-md cursor-pointer ${getLevelColor(item.level)} bg-opacity-30`}
                    onClick={() => setSelectedStudent(item)}
                  >
                    {/* 左侧状态标识 */}
                    <div className="flex flex-col items-center gap-2 mt-1">
                      <div className={`w-3 h-3 rounded-full shadow-sm ${getLevelBadge(item.level)}`}></div>
                      <div className="w-px h-12 bg-slate-200"></div>
                    </div>

                    {/* 核心信息 */}
                    <div className="flex-1">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-3">
                          <h3 className="text-lg font-bold text-slate-800">{item.name}</h3>
                          <span className="text-xs text-slate-500 font-medium bg-white px-2 py-0.5 rounded border border-slate-200">{item.major}</span>
                          <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${getLevelColor(item.level)}`}>
                            {item.levelText}
                          </span>
                        </div>
                        <span className="text-xs text-slate-400 flex items-center gap-1"><Clock size={12} /> {item.time}</span>
                      </div>
                      
                      <div className="bg-white/60 p-3 rounded-lg border border-white mt-3 flex items-start gap-3">
                        <BrainCircuit className="text-indigo-500 mt-0.5" size={16} />
                        <div>
                          <div className="text-xs font-bold text-indigo-700 mb-1">AI 预警模型触发原因：</div>
                          <div className="text-sm font-medium text-slate-700">{item.reason}</div>
                        </div>
                      </div>
                    </div>

                    {/* 右侧操作 */}
                    <div className="flex flex-col gap-2 mt-1 min-w-[120px]">
                      <button className="w-full py-2 bg-slate-800 text-white text-xs font-bold rounded-lg hover:bg-slate-700 transition-colors shadow-sm">
                        处理并干预
                      </button>
                      <button className="w-full py-2 bg-white border border-slate-200 text-slate-600 text-xs font-bold rounded-lg hover:bg-slate-50 transition-colors shadow-sm">
                        转介心理中心
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'profile' && (
            <div className="animate-in fade-in slide-in-from-bottom-2">
              <div className="flex justify-between items-center mb-6">
                <div className="flex gap-2">
                  <button className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg shadow-sm">全部学生 (218)</button>
                  <button className="px-4 py-2 bg-white border border-slate-200 text-slate-600 text-sm font-medium rounded-lg shadow-sm hover:bg-slate-50">高危预警 (3)</button>
                  <button className="px-4 py-2 bg-white border border-slate-200 text-slate-600 text-sm font-medium rounded-lg shadow-sm hover:bg-slate-50">重点关注 (12)</button>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 text-slate-600 text-sm font-medium rounded-lg shadow-sm hover:bg-slate-50">
                  <Filter size={16} /> 多维条件筛选
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {allStudents.map(student => (
                  <div key={student.id} onClick={() => setSelectedStudent(student)} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm hover:shadow-lg hover:border-blue-300 transition-all cursor-pointer flex flex-col items-center text-center group">
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center text-xl font-bold mb-3 transition-transform group-hover:scale-110 ${student.level === 'normal' ? 'bg-slate-100 text-slate-600' : getLevelBadge(student.level).replace('bg-', 'bg-').replace('500', '100') + ' text-' + student.level + '-600'}`}>
                      {student.avatar || student.name[0]}
                    </div>
                    <h3 className="text-lg font-bold text-slate-800 mb-1">{student.name}</h3>
                    <p className="text-xs text-slate-500 mb-3">{student.major}</p>
                    {student.level === 'normal' ? (
                      <span className="px-3 py-1 bg-green-50 text-green-600 border border-green-200 rounded-full text-xs font-bold">状态正常</span>
                    ) : (
                      <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getLevelColor(student.level)}`}>{student.levelText || '异常预警'}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'intervention' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2"><MessageSquare size={20} className="text-blue-500"/> 辅导员工作台 - 干预追踪记录</h2>
                <button className="text-sm text-blue-600 font-medium flex items-center gap-1 hover:text-blue-700">导出台账 <Download size={16}/></button>
              </div>
              <div className="divide-y divide-slate-100">
                {interventionRecords.map(record => (
                  <div key={record.id} className="p-6 hover:bg-slate-50 transition-colors flex gap-6">
                    <div className="w-32 flex-shrink-0 text-sm text-slate-500 font-medium pt-1">
                      {record.date}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="font-bold text-slate-800 text-lg">{record.student}</span>
                        <span className="px-2 py-1 bg-white text-slate-600 rounded text-xs font-bold border border-slate-200 shadow-sm">{record.action}</span>
                        <span className={`text-xs font-bold px-2 py-1 rounded ${record.trigger.includes('红') ? 'bg-red-50 text-red-600 border border-red-100' : record.trigger.includes('橙') ? 'bg-orange-50 text-orange-600 border border-orange-100' : 'bg-yellow-50 text-yellow-600 border border-yellow-100'}`}>
                          {record.trigger}
                        </span>
                      </div>
                      <p className="text-sm text-slate-700 mb-3 leading-relaxed">{record.detail}</p>
                      <div className="bg-indigo-50/50 p-3 rounded-lg border border-indigo-100 flex items-start gap-2">
                        <BrainCircuit size={16} className="text-indigo-500 mt-0.5 flex-shrink-0"/>
                        <p className="text-xs text-indigo-800 font-medium">AI 干预成效评估：{record.aiEval}</p>
                      </div>
                    </div>
                    <div className="pt-1">
                      <button className="text-slate-400 hover:text-slate-600 p-2 rounded-full hover:bg-slate-200 transition-colors"><MoreVertical size={20}/></button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'report' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 max-w-5xl mx-auto pb-10">
              <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden relative">
                
                {/* 报告头部 - 极具科技感 */}
                <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-blue-900 p-10 text-white relative overflow-hidden">
                  <div className="absolute right-0 top-0 w-64 h-64 bg-blue-500/20 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
                  <h1 className="text-3xl font-bold mb-3 relative z-10 flex items-center gap-3">
                    <FileText size={32} className="text-blue-400" />
                    2026年3月 学工智能分析月报
                  </h1>
                  <p className="text-blue-200 relative z-10 font-medium">基于全维度数据中台与AI社区助手深度分析生成 | 生成时间: 2026-03-31 18:00</p>
                </div>
                
                <div className="p-8">
                  {/* 核心指标统计 */}
                  <div className="grid grid-cols-3 gap-6 mb-8">
                    <div className="bg-slate-50 p-5 rounded-xl border border-slate-100 shadow-sm relative overflow-hidden">
                      <div className="absolute -right-4 -bottom-4 text-amber-500/10"><AlertTriangle size={80}/></div>
                      <div className="text-sm text-slate-500 font-bold mb-1 flex items-center gap-2"><AlertTriangle size={16} className="text-amber-500"/> 月度新增预警总数</div>
                      <div className="text-4xl font-black text-slate-800 mt-2">45 <span className="text-sm font-bold text-green-500 ml-2">↓ 12% 同比下降</span></div>
                    </div>
                    <div className="bg-slate-50 p-5 rounded-xl border border-slate-100 shadow-sm relative overflow-hidden">
                      <div className="absolute -right-4 -bottom-4 text-green-500/10"><CheckCircle size={80}/></div>
                      <div className="text-sm text-slate-500 font-bold mb-1 flex items-center gap-2"><CheckCircle size={16} className="text-green-500"/> 成功干预与闭环</div>
                      <div className="text-4xl font-black text-slate-800 mt-2">42 <span className="text-sm font-bold text-slate-400 ml-2">占比 93.3%</span></div>
                    </div>
                    <div className="bg-slate-50 p-5 rounded-xl border border-slate-100 shadow-sm relative overflow-hidden">
                      <div className="absolute -right-4 -bottom-4 text-blue-500/10"><TrendingUp size={80}/></div>
                      <div className="text-sm text-slate-500 font-bold mb-1 flex items-center gap-2"><TrendingUp size={16} className="text-blue-500"/> AI学生助手覆盖率</div>
                      <div className="text-4xl font-black text-slate-800 mt-2">92% <span className="text-sm font-bold text-green-500 ml-2">↑ 5% 同比上升</span></div>
                    </div>
                  </div>

                  {/* 伪图表区 */}
                  <div className="flex gap-6 mb-8">
                    <div className="flex-1 bg-white border border-slate-100 rounded-xl p-6 shadow-sm">
                      <h3 className="text-base font-bold text-slate-800 mb-6 flex items-center gap-2"><PieChart size={18} className="text-indigo-500"/> 本月风险预警类型分布</h3>
                      <div className="flex items-center justify-center gap-10">
                        {/* 纯CSS模拟环形图 */}
                        <div className="w-36 h-36 rounded-full border-[14px] border-slate-100 relative shadow-inner" style={{borderTopColor: '#ef4444', borderRightColor: '#f97316', borderBottomColor: '#3b82f6', borderLeftColor: '#3b82f6'}}></div>
                        <div className="space-y-3">
                          <div className="flex items-center gap-2 text-sm font-medium"><span className="w-3 h-3 rounded-full bg-blue-500 shadow-sm"></span> 学业压力预警 (45%)</div>
                          <div className="flex items-center gap-2 text-sm font-medium"><span className="w-3 h-3 rounded-full bg-orange-500 shadow-sm"></span> 行为与考勤异常 (30%)</div>
                          <div className="flex items-center gap-2 text-sm font-medium"><span className="w-3 h-3 rounded-full bg-red-500 shadow-sm"></span> 心理与社交高危 (25%)</div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex-1 bg-white border border-slate-100 rounded-xl p-6 shadow-sm">
                      <h3 className="text-base font-bold text-slate-800 mb-6 flex items-center gap-2"><BarChart3 size={18} className="text-indigo-500"/> 学生社区助手 - 咨询热词 Top 3</h3>
                      <div className="space-y-5 mt-2">
                        <div>
                          <div className="flex justify-between text-sm mb-1.5 font-bold"><span className="text-slate-700">1. 奖学金评定与申请规则</span> <span className="text-slate-500 text-xs">1,245 次交互</span></div>
                          <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden"><div className="bg-gradient-to-r from-blue-400 to-blue-600 h-full rounded-full w-full"></div></div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-1.5 font-bold"><span className="text-slate-700">2. 考研择校与就业政策</span> <span className="text-slate-500 text-xs">982 次交互</span></div>
                          <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden"><div className="bg-gradient-to-r from-indigo-400 to-indigo-600 h-full rounded-full w-4/5"></div></div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-1.5 font-bold"><span className="text-slate-700">3. 宿舍设施报修流程</span> <span className="text-slate-500 text-xs">654 次交互</span></div>
                          <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden"><div className="bg-slate-400 h-full rounded-full w-1/2"></div></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* AI 深度学情分析结论 */}
                  <div className="bg-indigo-50/70 p-6 rounded-xl border border-indigo-100">
                    <h3 className="text-base font-bold text-indigo-900 mb-3 flex items-center gap-2"><BrainCircuit size={20}/> AI 学情态势诊断与建议</h3>
                    <p className="text-sm text-slate-700 leading-relaxed mb-5">
                      系统综合研判：本月社区整体态势平稳。但需注意，受**期中考试**及**大三考研规划期**双重因素叠加影响，大三年级学生在 AI 社区助手中触发的“焦虑”、“迷茫”等情绪词汇环比上升了 <strong className="text-red-600 font-bold">18%</strong>。
                      <br/><br/>
                      <strong className="text-indigo-900">🔔 下月重点行动建议：</strong><br/>
                      1. 建议下月初联合就业指导中心，利用AI智能体向大三学生**定向推送**“升学与就业专场宣讲会”资讯。<br/>
                      2. 对本月产生的 3 例红牌重点预警学生，系统已自动生成待办工单，建议辅导员于下周一前落实二次线下回访，确保危机彻底闭环。
                    </p>
                    <button className="px-6 py-2.5 bg-indigo-600 text-white font-bold rounded-lg text-sm hover:bg-indigo-700 transition-colors shadow-md shadow-indigo-200 flex items-center gap-2">
                      一键导出完整 PDF 月度报告 <Download size={16}/>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

        </main>

        {/* 侧滑抽屉：学生全景画像 (PRD 数据融合核心体现) */}
        {selectedStudent && (
          <div className="absolute inset-0 z-50 flex justify-end">
            <div className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm" onClick={() => setSelectedStudent(null)}></div>
            <div className="w-[600px] bg-white h-full shadow-2xl animate-in slide-in-from-right flex flex-col border-l border-slate-200 relative z-10">
              
              {/* 抽屉头部 */}
              <div className="h-16 border-b border-slate-100 flex items-center justify-between px-6 bg-slate-50">
                <h2 className="font-bold text-slate-800 flex items-center gap-2">
                  <UserCheck size={18} className="text-blue-500" /> 学生全景画像
                </h2>
                <button onClick={() => setSelectedStudent(null)} className="p-2 hover:bg-slate-200 rounded-full text-slate-500 transition-colors">
                  <X size={20} />
                </button>
              </div>

              {/* 抽屉内容区 */}
              <div className="flex-1 overflow-y-auto p-6 bg-[#f8fafc]">
                
                {/* 个人基础信息卡 */}
                <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 mb-6 flex items-start gap-5">
                  <div className="w-16 h-16 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-2xl font-bold shadow-inner">
                    {selectedStudent.avatar || selectedStudent.name[0]}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-1">
                      <h3 className="text-xl font-bold text-slate-800">{selectedStudent.name}</h3>
                      <span className={`text-xs font-bold px-2 py-0.5 rounded border ${selectedStudent.level === 'normal' ? 'bg-green-100 text-green-600 border-green-200' : 'bg-red-100 text-red-600 border-red-200'}`}>
                        {selectedStudent.level === 'normal' ? '状态正常' : '重点关注'}
                      </span>
                    </div>
                    <div className="text-sm text-slate-500 mb-3">{selectedStudent.major} | 学号: {selectedStudent.idCard}</div>
                    <div className="flex gap-2">
                      {selectedStudent.level === 'normal' ? (
                        <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-md border border-slate-200">暂无异常业务标签</span>
                      ) : (
                        studentProfileData.tags.map((tag, idx) => (
                          <span key={idx} className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-md border border-slate-200">{tag}</span>
                        ))
                      )}
                    </div>
                  </div>
                </div>

                {/* 核心板块：AI干预建议 (体现系统智能价值) */}
                {selectedStudent.level !== 'normal' && (
                  <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl p-1 mb-6 shadow-sm border border-indigo-100">
                    <div className="bg-white/60 rounded-xl p-5 backdrop-blur-sm">
                      <h4 className="text-sm font-bold text-indigo-800 flex items-center gap-2 mb-3">
                        <BrainCircuit size={18} /> AI 综合分析与干预建议
                      </h4>
                      <p className="text-sm text-slate-700 leading-relaxed mb-4">
                        综合该生近半个月的跨系统数据：该生出现**学业受挫（挂科）**导致的**严重心理内耗**，且伴随经济压力（日均消费极低）。近期作息严重紊乱，在AI社区助手中表现出明显的**求助意愿与抑郁倾向**。
                      </p>
                      <div className="space-y-2">
                        <div className="flex items-start gap-2 bg-white p-3 rounded-lg border border-indigo-50 shadow-sm">
                          <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                          <span className="text-xs font-medium text-slate-700">建议立即安排线下谈话，优先安抚情绪，切忌直接施加学业压力。</span>
                        </div>
                        <div className="flex items-start gap-2 bg-white p-3 rounded-lg border border-indigo-50 shadow-sm">
                          <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                          <span className="text-xs font-medium text-slate-700">系统已自动为您匹配《心理危机干预标准话术库.pdf》，建议沟通前查阅。</span>
                        </div>
                        <div className="flex items-start gap-2 bg-white p-3 rounded-lg border border-indigo-50 shadow-sm">
                          <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                          <span className="text-xs font-medium text-slate-700">建议对接资助中心，查询该生是否符合临时困难补助发放条件。</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 数据融合面板展示 */}
                <h4 className="text-sm font-bold text-slate-800 mb-4 px-1">跨系统数据明细 (中央厨房汇聚)</h4>
                
                <div className="space-y-4">
                  {/* 心理与助手数据 */}
                  <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-700 mb-3 border-b border-slate-50 pb-2">
                      <MessageSquare className="text-purple-500" size={16} /> AI学生助手情感语义分析
                    </div>
                    <div className="text-sm space-y-2">
                      <div className="flex justify-between"><span className="text-slate-500">情绪判定：</span> <span className={`font-bold ${selectedStudent.level === 'normal' ? 'text-green-500' : 'text-red-500'}`}>{getDimensions(selectedStudent).aiAnalysis.sentiment}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">近期高频词：</span> <span>{getDimensions(selectedStudent).aiAnalysis.keywords.join("、")}</span></div>
                      <div className="mt-2 bg-slate-50 p-2 rounded text-xs text-slate-600 border border-slate-100">
                        <span className="font-bold">最新交互摘要：</span> {getDimensions(selectedStudent).aiAnalysis.recentChat}
                      </div>
                    </div>
                  </div>

                  {/* 教务数据 */}
                  <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-700 mb-3 border-b border-slate-50 pb-2">
                      <BookOpen className="text-blue-500" size={16} /> 教务系统同步数据
                    </div>
                    <div className="text-sm space-y-2">
                      <div className="flex justify-between"><span className="text-slate-500">当前绩点：</span> <span className={`font-bold ${selectedStudent.level === 'normal' ? 'text-slate-800' : 'text-orange-500'}`}>{getDimensions(selectedStudent).academic.gpa}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">考勤情况：</span> <span className="text-slate-800">{getDimensions(selectedStudent).academic.attendance}</span></div>
                    </div>
                  </div>

                  {/* 安防与后勤数据 */}
                  <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-700 mb-3 border-b border-slate-50 pb-2">
                      <Coffee className="text-amber-500" size={16} /> 安防/一卡通消费融合数据
                    </div>
                    <div className="text-sm space-y-2">
                      <div className="flex justify-between"><span className="text-slate-500">宿舍门禁：</span> <span className={`font-medium ${selectedStudent.level === 'normal' ? 'text-slate-800' : 'text-red-500'}`}>{getDimensions(selectedStudent).behavior.dorm}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">食堂消费：</span> <span className="text-slate-800">{getDimensions(selectedStudent).behavior.canteen}</span></div>
                      <div className="flex justify-between"><span className="text-slate-500">校园网活跃：</span> <span className="text-slate-800">{getDimensions(selectedStudent).behavior.network}</span></div>
                    </div>
                  </div>
                </div>

              </div>
              
              {/* 底部操作区 */}
              <div className="p-4 bg-white border-t border-slate-200 flex gap-3">
                {selectedStudent.level !== 'normal' ? (
                  <>
                    <button className="flex-1 py-3 bg-red-600 text-white font-bold rounded-xl hover:bg-red-700 transition-colors shadow-sm shadow-red-200">
                      录入干预谈话记录
                    </button>
                    <button className="flex-1 py-3 bg-white border-2 border-slate-200 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-colors">
                      发起跨部门协同工单
                    </button>
                  </>
                ) : (
                  <>
                    <button className="flex-1 py-3 bg-blue-600 text-white font-bold rounded-xl hover:bg-blue-700 transition-colors shadow-sm shadow-blue-200">
                      发送日常关怀消息
                    </button>
                    <button className="flex-1 py-3 bg-white border-2 border-slate-200 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-colors">
                      查看完整电子档案
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}