import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  MoreHorizontal, 
  UserPlus, 
  MessageSquare, 
  Phone, 
  Zap, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  ChevronDown,
  Layout,
  MessageCircle,
  User,
  PieChart,
  Users,
  Bell,
  RefreshCw,
  Send
} from 'lucide-react';

const SmartClientList = () => {
  const [activeTab, setActiveTab] = useState('current'); // 'current' | 'longterm'
  const [selectedUsers, setSelectedUsers] = useState([]);

  // --- 模拟数据：本期营 (Current Camp) ---
  const currentCampData = [
    { 
      id: 1, name: "张三", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=ZhangSan", 
      level: "S", score: 95, source: "抖音直播间", 
      status: "planning", statusText: "已申请规划", 
      tags: ["价格敏感", "急需转行"], 
      lastAction: "10分钟前 浏览了分期页面",
      aiTip: "高意向！建议立即电话逼单"
    },
    { 
      id: 2, name: "李四", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=LiSi", 
      level: "S", score: 92, source: "公众号", 
      status: "quoted", statusText: "已报价 (2980)", 
      tags: ["决策人是妻子"], 
      lastAction: "30分钟前 回复了 '再想想'",
      aiTip: "需解决顾虑，发送学员案例"
    },
    { 
      id: 3, name: "王五", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=WangWu", 
      level: "A", score: 85, source: "转介绍", 
      status: "watching", statusText: "观看直播中", 
      tags: ["技术小白"], 
      lastAction: "当前在线 (直播间)",
      aiTip: "互动引导，提及零基础友好"
    },
    { 
      id: 4, name: "赵六", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=ZhaoLiu", 
      level: "A", score: 82, source: "百度投放", 
      status: "pending", statusText: "待建联", 
      tags: [], 
      lastAction: "2小时前 通过好友",
      aiTip: "发送破冰话术 A3"
    },
    { 
      id: 5, name: "钱七", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=QianQi", 
      level: "B", score: 60, source: "抖音表单", 
      status: "cold", statusText: "意向一般", 
      tags: ["只是看看"], 
      lastAction: "昨天",
      aiTip: "保持日常SOP跟进"
    },
  ];

  // --- 模拟数据：长期池 (Long-term Pool) ---
  const longTermPoolData = [
    // 短期高价值 (近期流失的 S/A 量)
    { 
      id: 101, name: "陈八 (第5期)", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=ChenBa", 
      level: "S", score: 88, type: "short_high",
      lostReason: "等发工资 (15号)", 
      lostTime: "5天前",
      status: "waiting", statusText: "待激活",
      activation: "Today is 15th! 发送工资日优惠券"
    },
    { 
      id: 102, name: "周九 (第5期)", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=ZhouJiu", 
      level: "A", score: 84, type: "short_high",
      lostReason: "家人反对", 
      lostTime: "7天前",
      status: "waiting", statusText: "需攻坚",
      activation: "发送 '副业回本' 真实案例"
    },
    // 长期存量
    { 
      id: 103, name: "吴十 (第2期)", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=WuShi", 
      level: "B", score: 45, type: "long_stock",
      lostReason: "嫌贵/失联", 
      lostTime: "30天前",
      status: "lost", statusText: "沉睡中",
      activation: "群发 618 大促活动"
    },
    { 
      id: 104, name: "郑十一", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Zheng", 
      level: "B", score: 40, type: "long_stock",
      lostReason: "无需求", 
      lostTime: "45天前",
      status: "lost", statusText: "沉睡中",
      activation: "推送免费公开课"
    },
  ];

  // 切换 Tab 清空选择
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setSelectedUsers([]);
  };

  // 处理多选
  const toggleUser = (id) => {
    if (selectedUsers.includes(id)) {
      setSelectedUsers(selectedUsers.filter(uid => uid !== id));
    } else {
      setSelectedUsers([...selectedUsers, id]);
    }
  };

  const selectAll = (data) => {
    if (selectedUsers.length === data.length) {
      setSelectedUsers([]);
    } else {
      setSelectedUsers(data.map(u => u.id));
    }
  };

  const getCurrentData = () => activeTab === 'current' ? currentCampData : longTermPoolData;

  // 渲染分级标签
  const renderLevelBadge = (level) => {
    const styles = {
      S: "bg-red-100 text-red-600 border-red-200",
      A: "bg-orange-100 text-orange-600 border-orange-200",
      B: "bg-yellow-100 text-yellow-700 border-yellow-200",
      C: "bg-gray-100 text-gray-500 border-gray-200",
    };
    return (
      <span className={`w-6 h-6 flex items-center justify-center rounded text-xs font-bold border ${styles[level] || styles.C}`}>
        {level}
      </span>
    );
  };

  // 渲染状态
  const renderStatus = (status, text) => {
    const icons = {
      planning: <CheckCircle2 size={14} className="text-blue-500" />,
      quoted: <Zap size={14} className="text-purple-500" />,
      watching: <Users size={14} className="text-green-500" />,
      pending: <Clock size={14} className="text-gray-400" />,
      cold: <XCircle size={14} className="text-gray-300" />,
      waiting: <AlertCircle size={14} className="text-orange-500" />,
      lost: <XCircle size={14} className="text-gray-300" />
    };
    return (
      <div className="flex items-center gap-1.5 text-sm text-gray-700 font-medium">
        {icons[status]}
        <span>{text}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans flex flex-col">
      
      {/* 顶部导航 (Global Header) */}
      <div className="bg-white border-b border-gray-200 px-6 py-3 flex justify-between items-center h-16 shrink-0 sticky top-0 z-20">
        <div className="flex items-center gap-4">
          <div className="bg-blue-600 p-1.5 rounded text-white shadow-sm">
            <Layout size={20} />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-800 leading-none">客户智能列表</h1>
            <span className="text-[10px] text-gray-400">Smart Client List v2.0</span>
          </div>
        </div>
        
        {/* 顶部工具栏 */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
            <input 
              type="text" 
              placeholder="搜索姓名、手机号..." 
              className="pl-9 pr-4 py-1.5 bg-gray-100 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-100 w-64 border-transparent border focus:border-blue-300 transition-all"
            />
          </div>
          <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full relative">
            <Bell size={18} />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border border-white"></span>
          </button>
          <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Sales" className="w-8 h-8 rounded-full border border-gray-200" alt="Me" />
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        
        {/* 左侧菜单 (Sidebar) */}
        <div className="w-[64px] bg-[#1e293b] flex flex-col items-center py-6 gap-6 shrink-0 text-slate-400">
           <div className="p-2 bg-blue-600 text-white rounded-lg shadow-lg shadow-blue-900/50 cursor-pointer">
             <Users size={20} />
           </div>
           <div className="p-2 hover:text-white cursor-pointer"><MessageSquare size={20} /></div>
           <div className="p-2 hover:text-white cursor-pointer"><PieChart size={20} /></div>
           <div className="p-2 hover:text-white cursor-pointer mt-auto"><User size={20} /></div>
        </div>

        {/* 主内容区 */}
        <div className="flex-1 flex flex-col bg-gray-50 overflow-hidden relative">
          
          {/* Tabs & Filters */}
          <div className="px-8 pt-6 pb-2">
            <div className="flex items-center justify-between mb-4">
              <div className="flex bg-white p-1 rounded-xl shadow-sm border border-gray-200">
                <button 
                  onClick={() => handleTabChange('current')}
                  className={`px-6 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${activeTab === 'current' ? 'bg-blue-50 text-blue-600 shadow-sm border border-blue-100' : 'text-gray-500 hover:text-gray-700'}`}
                >
                  <Users size={16} />
                  本期营 (第6期)
                  <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-xs ml-1">128</span>
                </button>
                <button 
                  onClick={() => handleTabChange('longterm')}
                  className={`px-6 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${activeTab === 'longterm' ? 'bg-indigo-50 text-indigo-600 shadow-sm border border-indigo-100' : 'text-gray-500 hover:text-gray-700'}`}
                >
                  <RefreshCw size={16} />
                  长期池 (捡漏)
                  <span className="bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded text-xs ml-1">570</span>
                </button>
              </div>

              <div className="flex gap-2">
                 <button className="flex items-center gap-1 px-3 py-1.5 bg-white border border-gray-200 rounded-md text-sm text-gray-600 hover:bg-gray-50">
                    <Filter size={14} /> 筛选
                 </button>
                 <button className="flex items-center gap-1 px-3 py-1.5 bg-white border border-gray-200 rounded-md text-sm text-gray-600 hover:bg-gray-50">
                    <Clock size={14} /> 最近活跃 <ChevronDown size={12} />
                 </button>
              </div>
            </div>
            
            {/* Banner/Notice for Long-term */}
            {activeTab === 'longterm' && (
               <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 text-indigo-800 px-4 py-2 rounded-lg mb-4 text-sm flex items-center gap-2">
                 <Zap size={16} className="text-indigo-600" />
                 <span>AI 已为您识别 <strong>12</strong> 位“短期高价值”流失客户，建议今日优先激活。</span>
               </div>
            )}
          </div>

          {/* List Content */}
          <div className="flex-1 overflow-y-auto px-8 pb-20">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              
              {/* Table Header */}
              <div className="grid grid-cols-12 gap-4 px-6 py-3 bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                <div className="col-span-1 flex items-center">
                   <input type="checkbox" className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer" 
                     checked={selectedUsers.length > 0 && selectedUsers.length === getCurrentData().length}
                     onChange={() => selectAll(getCurrentData())}
                   />
                </div>
                <div className="col-span-3">客户信息 / 来源</div>
                <div className="col-span-1 text-center">分级</div>
                <div className="col-span-2">{activeTab === 'current' ? '当前状态' : '流失原因'}</div>
                <div className="col-span-3">{activeTab === 'current' ? 'AI 建议 / 下一步' : '激活建议'}</div>
                <div className="col-span-2 text-right">{activeTab === 'current' ? '最近活跃' : '流失时间'}</div>
              </div>

              {/* Table Body */}
              <div className="divide-y divide-gray-100">
                {activeTab === 'current' ? (
                  currentCampData.map(user => (
                    <div key={user.id} className={`grid grid-cols-12 gap-4 px-6 py-4 items-center hover:bg-blue-50/50 transition-colors group ${selectedUsers.includes(user.id) ? 'bg-blue-50' : ''}`}>
                      <div className="col-span-1">
                        <input 
                          type="checkbox" 
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                          checked={selectedUsers.includes(user.id)}
                          onChange={() => toggleUser(user.id)}
                        />
                      </div>
                      <div className="col-span-3 flex items-center gap-3">
                        <div className="relative">
                           <img src={user.avatar} alt="" className="w-10 h-10 rounded-full bg-gray-100" />
                           {user.level === 'S' && <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-white"></div>}
                        </div>
                        <div>
                          <div className="font-bold text-gray-900 text-sm flex items-center gap-2">
                            {user.name}
                            {user.tags.map((tag, i) => (
                              <span key={i} className="px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded text-[10px] font-normal border border-gray-200">{tag}</span>
                            ))}
                          </div>
                          <div className="text-xs text-gray-400 mt-0.5">{user.source}</div>
                        </div>
                      </div>
                      <div className="col-span-1 flex justify-center">
                        {renderLevelBadge(user.level)}
                      </div>
                      <div className="col-span-2">
                        {renderStatus(user.status, user.statusText)}
                      </div>
                      <div className="col-span-3">
                        <div className={`text-xs px-2 py-1.5 rounded border inline-block ${user.level === 'S' ? 'bg-red-50 text-red-700 border-red-100 font-medium' : 'bg-gray-50 text-gray-600 border-gray-200'}`}>
                           💡 {user.aiTip}
                        </div>
                      </div>
                      <div className="col-span-2 text-right text-xs text-gray-400 font-mono">
                        {user.lastAction}
                      </div>
                    </div>
                  ))
                ) : (
                  <>
                    {/* Long Term Pool Group 1: High Value */}
                    <div className="px-6 py-2 bg-orange-50/50 border-b border-orange-100 text-xs font-bold text-orange-800 flex items-center gap-2">
                      <Zap size={14} /> 短期高价值 (近期流失S/A量)
                    </div>
                    {longTermPoolData.filter(u => u.type === 'short_high').map(user => (
                      <div key={user.id} className={`grid grid-cols-12 gap-4 px-6 py-4 items-center hover:bg-orange-50 transition-colors ${selectedUsers.includes(user.id) ? 'bg-orange-50' : ''}`}>
                         <div className="col-span-1"><input type="checkbox" checked={selectedUsers.includes(user.id)} onChange={() => toggleUser(user.id)} className="rounded text-orange-600 focus:ring-orange-500"/></div>
                         <div className="col-span-3 flex items-center gap-3">
                            <img src={user.avatar} alt="" className="w-10 h-10 rounded-full grayscale" />
                            <div>
                               <div className="font-bold text-gray-700 text-sm">{user.name}</div>
                               <div className="text-xs text-gray-400 mt-0.5">历史评分: {user.score}</div>
                            </div>
                         </div>
                         <div className="col-span-1 flex justify-center">{renderLevelBadge(user.level)}</div>
                         <div className="col-span-2">
                            <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs font-medium border border-gray-200">{user.lostReason}</span>
                         </div>
                         <div className="col-span-3">
                            <div className="text-xs text-green-700 font-medium flex items-center gap-1">
                               <CheckCircle2 size={12} /> 推荐激活: {user.activation}
                            </div>
                         </div>
                         <div className="col-span-2 text-right text-xs text-gray-400">{user.lostTime}</div>
                      </div>
                    ))}
                    
                    {/* Long Term Pool Group 2: Stock */}
                    <div className="px-6 py-2 bg-gray-50 border-b border-gray-200 text-xs font-bold text-gray-500 mt-2">
                      长期存量
                    </div>
                    {longTermPoolData.filter(u => u.type === 'long_stock').map(user => (
                       <div key={user.id} className={`grid grid-cols-12 gap-4 px-6 py-4 items-center opacity-70 hover:opacity-100 hover:bg-gray-50 transition-all ${selectedUsers.includes(user.id) ? 'bg-gray-100' : ''}`}>
                          <div className="col-span-1"><input type="checkbox" checked={selectedUsers.includes(user.id)} onChange={() => toggleUser(user.id)} className="rounded"/></div>
                          <div className="col-span-3 flex items-center gap-3">
                             <img src={user.avatar} alt="" className="w-10 h-10 rounded-full grayscale opacity-50" />
                             <div className="font-medium text-gray-600 text-sm">{user.name}</div>
                          </div>
                          <div className="col-span-1 flex justify-center">{renderLevelBadge(user.level)}</div>
                          <div className="col-span-2 text-xs text-gray-400">{user.lostReason}</div>
                          <div className="col-span-3 text-xs text-gray-400">{user.activation}</div>
                          <div className="col-span-2 text-right text-xs text-gray-400">{user.lostTime}</div>
                       </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* 底部批量操作栏 (Floating Action Bar) */}
          <div className={`absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white px-6 py-3 rounded-xl shadow-2xl flex items-center gap-6 transition-all duration-300 ${selectedUsers.length > 0 ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'}`}>
            <div className="flex items-center gap-2 border-r border-gray-700 pr-6">
               <span className="bg-blue-600 text-white text-xs font-bold px-2 py-0.5 rounded">{selectedUsers.length}</span>
               <span className="text-sm">已选择客户</span>
            </div>
            
            <div className="flex items-center gap-4">
              <button className="flex items-col flex-col items-center gap-1 group hover:text-blue-400 transition-colors">
                 <MessageSquare size={18} />
                 <span className="text-[10px]">群发消息</span>
              </button>
              <button className="flex items-col flex-col items-center gap-1 group hover:text-green-400 transition-colors">
                 <Phone size={18} />
                 <span className="text-[10px]">智能外呼</span>
              </button>
              {activeTab === 'longterm' ? (
                 <button className="flex items-col flex-col items-center gap-1 group hover:text-yellow-400 transition-colors">
                    <Zap size={18} />
                    <span className="text-[10px]">一键激活</span>
                 </button>
              ) : (
                 <button className="flex items-col flex-col items-center gap-1 group hover:text-purple-400 transition-colors">
                    <UserPlus size={18} />
                    <span className="text-[10px]">分配规划师</span>
                 </button>
              )}
            </div>
            
            <button 
               onClick={() => setSelectedUsers([])}
               className="ml-2 p-1 hover:bg-gray-800 rounded-full"
            >
               <XCircle size={18} className="text-gray-500" />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SmartClientList;