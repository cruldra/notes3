import React, { useState } from 'react';
import { Info, ChefHat, Wind, Droplets, Refrigerator, Zap, Package, Maximize, Truck, Coffee, Flame, Utensils, ShieldCheck, Video, BellRing } from 'lucide-react';

const CommercialKitchenLayout = () => {
  const [activeItem, setActiveItem] = useState(null);

  // 商用厨房设备数据配置
  const equipment = [
    {
      id: 'storage_cold',
      name: '冷库/冷藏区 (Storage)',
      icon: <Refrigerator className="w-5 h-5" />,
      desc: '步入式冷库或四门雪柜，用于储存大量生鲜食材。',
      x: 20, y: 20, width: 100, height: 120,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={isActive ? '#93c5fd' : '#e2e8f0'} stroke="#475569" strokeWidth="2" />
          <line x1={x} y1={y+h/2} x2={x+w} y2={y+h/2} stroke="#94a3b8" />
          <text x={x + w/2} y={y + h/2 - 15} fontSize="12" textAnchor="middle" fill="#475569" fontWeight="bold">冷冻库</text>
          <text x={x + w/2} y={y + h/2 + 25} fontSize="12" textAnchor="middle" fill="#475569" fontWeight="bold">冷藏库</text>
        </g>
      )
    },
    {
      id: 'prep_area',
      name: '粗加工/备菜区 (Prep)',
      icon: <Package className="w-5 h-5" />,
      desc: '独立的水池和台面，生熟分开，负责切配和清洗。',
      x: 140, y: 20, width: 140, height: 80,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={isActive ? '#a7f3d0' : '#f0fdf4'} stroke="#86efac" strokeWidth="1" />
          {/* 水池 */}
          <rect x={x + 10} y={y + 10} width={30} height={30} fill="#fff" stroke="#cbd5e1" />
          <rect x={x + 50} y={y + 10} width={30} height={30} fill="#fff" stroke="#cbd5e1" />
          <text x={x + w/2} y={y + 60} fontSize="12" textAnchor="middle" fill="#166534">粗加工/切配台</text>
        </g>
      )
    },
    {
      id: 'hot_line',
      name: '热厨烹饪线 (Hot Line)',
      icon: <Flame className="w-5 h-5" />,
      desc: '核心区域：猛火灶、炸炉、扒炉一字排开，上方必须有强力排烟运水烟罩。',
      x: 140, y: 150, width: 260, height: 80,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={isActive ? '#fca5a5' : '#fee2e2'} stroke="#ef4444" strokeWidth="1" />
          {/* 灶头示意 */}
          <circle cx={x + 30} cy={y + 40} r="15" stroke="#b91c1c" fill="none" strokeWidth="2" />
          <circle cx={x + 70} cy={y + 40} r="15" stroke="#b91c1c" fill="none" strokeWidth="2" />
          <rect x={x + 100} y={y + 20} width={40} height={40} stroke="#b91c1c" fill="none" /> {/* 扒炉 */}
          <rect x={x + 150} y={y + 20} width={30} height={40} stroke="#b91c1c" fill="none" /> {/* 炸炉 */}
           <rect x={x + 200} y={y + 10} width={50} height={60} stroke="#b91c1c" fill="none" /> {/* 万能蒸烤箱 */}
          <text x={x + w/2} y={y + 75} fontSize="12" textAnchor="middle" fill="#991b1b" fontWeight="bold">热厨烹饪线</text>
        </g>
      )
    },
    {
      id: 'hood',
      name: '排烟/新风系统',
      icon: <Wind className="w-5 h-5" />,
      desc: '覆盖整个热厨区，商场要求极高，通常包含油烟净化。',
      x: 135, y: 145, width: 270, height: 90,
      isOverlay: true,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill="none" stroke={isActive ? '#ef4444' : '#ef4444'} strokeWidth="2" strokeDasharray="5,5" />
          <text x={x + w - 40} y={y - 5} fontSize="10" textAnchor="middle" fill="#ef4444">运水烟罩</text>
        </g>
      )
    },
    {
      id: 'pass_station',
      name: '出餐台/打荷区 (Pass)',
      icon: <Utensils className="w-5 h-5" />,
      desc: '厨师将做好的菜放在这里，服务员从外侧取餐。上方通常有保温灯。',
      x: 140, y: 280, width: 260, height: 40,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={isActive ? '#fdba74' : '#ffedd5'} stroke="#ea580c" strokeWidth="2" />
          <line x1={x} y1={y+h/2} x2={x+w} y2={y+h/2} stroke="#fb923c" strokeDasharray="2,2"/>
          <text x={x + w/2} y={y + 25} fontSize="12" textAnchor="middle" fill="#c2410c">出餐台 (Pass Window)</text>
        </g>
      )
    },
    {
      id: 'dish_pit',
      name: '洗消间 (Dishwashing)',
      icon: <Droplets className="w-5 h-5" />,
      desc: '污碟回收、残食垃圾处理、高压花洒预洗、洗碗机清洗、消毒存放。',
      x: 450, y: 20, width: 120, height: 150,
      render: (x, y, w, h, isActive) => (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={isActive ? '#bfdbfe' : '#eff6ff'} stroke="#3b82f6" strokeWidth="1" />
          {/* 洗碗机流线 */}
          <rect x={x + 10} y={y + 100} width={30} height={30} fill="#ddd" stroke="#999" /> {/* 污碟台 */}
          <rect x={x + 45} y={y + 100} width={40} height={40} fill="#93c5fd" stroke="#2563eb" /> {/* 洗碗机 */}
          <rect x={x + 90} y={y + 100} width={20} height={30} fill="#ddd" stroke="#999" /> {/* 洁碟台 */}
          <text x={x + w/2} y={y + 80} fontSize="12" textAnchor="middle" fill="#1e40af">洗消流水线</text>
           <path d={`M${x+10} ${y+10} L${x+50} ${y+10}`} stroke="#3b82f6" strokeWidth="2"/>
           <text x={x + 60} y={y + 15} fontSize="10" fill="#3b82f6">回收口</text>
        </g>
      )
    },
    {
      id: 'security_sys',
      name: '安防监控系统 (Security)',
      icon: <ShieldCheck className="w-5 h-5" />,
      desc: '包含高清CCTV监控、热厨温感、仓库烟感及消防联动控制。',
      x: 480, y: 320, width: 90, height: 60, // 放置在右下角墙面
      render: (x, y, w, h, isActive) => (
        <g>
           {/* 安防主控箱 */}
           <rect x={x} y={y} width={w} height={h} fill={isActive ? '#fecaca' : '#f1f5f9'} stroke="#b91c1c" strokeWidth="2" />
           <rect x={x+10} y={y+10} width={w-20} height={h-20} fill="#fff" stroke="#e2e8f0" />
           <circle cx={x+w-15} cy={y+15} r="3" fill="#ef4444" className={isActive ? "animate-pulse" : ""} />
           <text x={x + w/2} y={y + 40} fontSize="10" textAnchor="middle" fill="#b91c1c" fontWeight="bold">安防主控</text>

           {/* 全局安防设备 (绘制在图纸各个位置，坐标为绝对坐标) */}
           
           {/* 1. 摄像头 - 后厨入口 */}
           <g transform="translate(30, 80)">
              <Video size={16} fill="#1e293b" stroke="#1e293b" />
              {isActive && (
                 <path d="M 8 16 L -20 60 L 40 60 Z" fill="#ef4444" opacity="0.2" />
              )}
           </g>

           {/* 2. 摄像头 - 出餐口/大厅连接处 */}
           <g transform="translate(120, 360)">
              <Video size={16} fill="#1e293b" stroke="#1e293b" transform="rotate(-45 8 8)"/>
              {isActive && (
                 <path d="M 8 16 L -20 60 L 40 60 Z" fill="#ef4444" opacity="0.2" transform="rotate(-45 8 8)" />
              )}
           </g>

           {/* 3. 摄像头 - 洗消间/污碟口 */}
           <g transform="translate(560, 40)">
              <Video size={16} fill="#1e293b" stroke="#1e293b" transform="rotate(135 8 8)"/>
              {isActive && (
                 <path d="M 8 16 L -20 60 L 40 60 Z" fill="#ef4444" opacity="0.2" transform="rotate(135 8 8)" />
              )}
           </g>

           {/* 4. 摄像头 - 热厨烹饪区监控 (新增) */}
           <g transform="translate(270, 120)">
              <Video size={16} fill="#1e293b" stroke="#1e293b" transform="rotate(90 8 8)"/>
              {isActive && (
                 <path d="M 8 16 L -20 60 L 40 60 Z" fill="#ef4444" opacity="0.2" transform="rotate(90 8 8)" />
              )}
           </g>

           {/* 5. 温感探测器 - 热厨区 (高温环境用温感，不用烟感) */}
           <circle cx={275} cy={130} r={5} fill="#ef4444" stroke="white" strokeWidth="1"/>
           {isActive && <text x={275} y={120} fontSize="10" textAnchor="middle" fill="#ef4444" fontWeight="bold">温感</text>}

           {/* 6. 烟感探测器 - 仓库区 */}
           <circle cx={70} cy={70} r={5} fill="#f59e0b" stroke="white" strokeWidth="1"/>
           {isActive && <text x={70} y={60} fontSize="10" textAnchor="middle" fill="#d97706" fontWeight="bold">烟感</text>}
           
           {/* 7. 烟感探测器 - 走廊 */}
           <circle cx={300} cy={80} r={5} fill="#f59e0b" stroke="white" strokeWidth="1"/>
        </g>
      )
    }
  ];

  return (
    <div className="flex flex-col md:flex-row h-screen bg-gray-50 p-4 gap-4 font-sans text-slate-800">
      
      {/* 左侧：平面图显示区域 */}
      <div className="flex-1 bg-white rounded-xl shadow-lg border border-gray-200 p-6 flex flex-col overflow-hidden relative">
        <div className="mb-4 border-b pb-2 flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-slate-800">商用餐饮厨房布局 (Mall Standard)</h2>
            <p className="text-sm text-slate-500">强调动线分区：进货 &rarr; 加工 &rarr; 烹饪 &rarr; 出餐 | 污碟独立回收</p>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center overflow-auto bg-grid-slate-100">
          <svg 
            viewBox="0 0 600 400" 
            className="w-full max-w-3xl h-auto drop-shadow-xl"
            style={{ maxHeight: '100%' }}
          >
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e2e8f0" strokeWidth="1"/>
              </pattern>
              <marker id="arrow-flow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
              </marker>
               <marker id="arrow-dirty" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
              </marker>
            </defs>

            {/* 地板网格 */}
            <rect width="600" height="400" fill="#fff" />
            <rect width="600" height="400" fill="url(#grid)" />

            {/* 墙体 */}
            <path d="M 10 380 L 10 10 L 590 10 L 590 380" fill="none" stroke="#334155" strokeWidth="8" strokeLinecap="round" />
            
            {/* 门/通道示意 */}
            {/* 后厨入口 (进货/员工) */}
            <g transform="translate(10, 80)">
               <line x1="0" y1="0" x2="0" y2="40" stroke="#fff" strokeWidth="10" />
               <text x="15" y="25" fontSize="10" fill="#64748b">后厨/进货门</text>
            </g>
            
            {/* 出餐口 (连接餐厅) */}
             <g transform="translate(100, 380)">
               <line x1="0" y1="0" x2="340" y2="0" stroke="#fff" strokeWidth="10" />
               <text x="170" y="-10" textAnchor="middle" fontSize="12" fill="#ea580c" fontWeight="bold">前厅/就餐区 (出餐方向)</text>
            </g>

            {/* 污碟回收口 */}
            <g transform="translate(590, 80)">
               <line x1="0" y1="0" x2="0" y2="40" stroke="#fff" strokeWidth="10" />
               <text x="-60" y="25" fontSize="10" fill="#64748b">污碟回收窗</text>
            </g>

            {/* 渲染设备 */}
            {equipment.map((item) => (
              <g 
                key={item.id} 
                className="cursor-pointer transition-all duration-200 hover:opacity-90"
                onMouseEnter={() => setActiveItem(item)}
                onMouseLeave={() => setActiveItem(null)}
                onClick={() => setActiveItem(item)}
                opacity={activeItem && activeItem.id !== item.id && !item.isOverlay && activeItem.id !== 'security_sys' ? 0.6 : 1}
              >
                {item.render(item.x, item.y, item.width, item.height, activeItem?.id === item.id)}
              </g>
            ))}

            {/* 动线箭头示意 (仅当未选中任何物体或选中安防系统时显示背景流线) */}
            {(!activeItem || activeItem.id === 'security_sys') && (
               <g opacity="0.3" pointerEvents="none">
                 {/* 食品流线 (绿色) */}
                 <path d="M 30 80 L 60 80" fill="none" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-flow)" />
                 <path d="M 120 80 L 140 80" fill="none" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-flow)" />
                 <path d="M 210 100 L 210 150" fill="none" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-flow)" />
                 <path d="M 270 230 L 270 280" fill="none" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-flow)" />
                 <path d="M 270 320 L 270 360" fill="none" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-flow)" />
                 
                 {/* 污碟流线 (灰色) */}
                 <path d="M 570 80 L 540 80" fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="4,4" markerEnd="url(#arrow-dirty)" />
               </g>
            )}

            {/* 地沟示意 (商用厨房必备) */}
            <path d="M 140 250 L 400 250" stroke="#cbd5e1" strokeWidth="4" strokeDasharray="10,5" opacity="0.5"/>
            <text x="410" y="255" fontSize="10" fill="#94a3b8">地沟/排水网</text>

          </svg>
        </div>
      </div>

      {/* 右侧：设备详情列表 */}
      <div className="w-full md:w-80 flex flex-col gap-4">
        <div className="bg-white p-4 rounded-xl shadow-md border border-gray-200 h-full overflow-y-auto">
          <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-500" />
            商用厨房配置
          </h3>
          
          <div className="space-y-3">
            {equipment.map((item) => (
              <div 
                key={item.id}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  activeItem?.id === item.id 
                    ? 'bg-blue-50 border-blue-400 shadow-sm ring-1 ring-blue-200' 
                    : 'bg-white border-gray-100 hover:border-blue-200 hover:bg-gray-50'
                }`}
                onMouseEnter={() => setActiveItem(item)}
                onMouseLeave={() => setActiveItem(null)}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-full ${activeItem?.id === item.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'}`}>
                    {item.icon}
                  </div>
                  <div>
                    <h4 className={`font-medium ${activeItem?.id === item.id ? 'text-blue-700' : 'text-slate-700'}`}>
                      {item.name}
                    </h4>
                    <p className="text-xs text-gray-500 mt-1 leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-amber-50 rounded-lg border border-amber-100 text-sm text-amber-800">
            <h4 className="font-bold mb-2 flex items-center gap-2">
               商场安防审核重点
            </h4>
            <ul className="list-disc list-inside space-y-1 opacity-80 text-xs">
              <li><strong>无死角监控：</strong>收银、出餐、进货通道必须有高清监控覆盖。</li>
              <li><strong>防误报：</strong>热厨区禁止安装烟感（油烟会触发），必须使用温感探测器。</li>
              <li><strong>联动控制：</strong>消防系统需与燃气切断阀和抽排设备联动。</li>
            </ul>
          </div>
        </div>
      </div>

    </div>
  );
};

export default CommercialKitchenLayout;