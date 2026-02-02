import React, { useState, useEffect, useRef } from 'react';
import { Maximize, Move, MousePointer2, BoxSelect } from 'lucide-react';

const GodotRect2Demo = () => {
  // 模拟 Godot 的 Rect2 数据结构
  const [rect, setRect] = useState({ position: { x: 100, y: 100 }, size: { x: 200, y: 150 } });
  
  // 用于测试相交的另一个 Rect (干扰项)
  const [enemyRect, setEnemyRect] = useState({ position: { x: 350, y: 200 }, size: { x: 100, y: 100 } });
  const [isDraggingEnemy, setIsDraggingEnemy] = useState(false);

  // 鼠标状态
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  // Godot API 模拟实现 --------------------------------------------
  
  // 计算 end (position + size)
  const end = { x: rect.position.x + rect.size.x, y: rect.position.y + rect.size.y };
  
  // 计算 center (position + size/2)
  const center = { x: rect.position.x + rect.size.x / 2, y: rect.position.y + rect.size.y / 2 };
  
  // 计算 area (width * height)
  const area = rect.size.x * rect.size.y;

  // 模拟 rect2.has_point(point)
  const hasPoint = (r, p) => {
    return p.x >= r.position.x && p.x <= r.position.x + r.size.x &&
           p.y >= r.position.y && p.y <= r.position.y + r.size.y;
  };

  // 模拟 rect2.intersects(other_rect)
  const intersects = (r1, r2) => {
    return !(r2.position.x > r1.position.x + r1.size.x || 
             r2.position.x + r2.size.x < r1.position.x || 
             r2.position.y > r1.position.y + r1.size.y || 
             r2.position.y + r2.size.y < r1.position.y);
  };

  // 状态检查
  const isMouseOver = hasPoint(rect, mousePos);
  const isColliding = intersects(rect, enemyRect);

  // -------------------------------------------------------------

  // 处理鼠标移动
  const handleMouseMove = (e) => {
    if (containerRef.current) {
      const bounds = containerRef.current.getBoundingClientRect();
      const x = e.clientX - bounds.left;
      const y = e.clientY - bounds.top;
      setMousePos({ x, y });

      if (isDraggingEnemy) {
        setEnemyRect(prev => ({
          ...prev,
          position: { x: x - prev.size.x / 2, y: y - prev.size.y / 2 }
        }));
      }
    }
  };

  return (
    <div className="flex flex-col h-screen max-h-[800px] bg-[#212529] text-gray-200 font-sans border border-gray-700 rounded-lg overflow-hidden shadow-2xl">
      
      {/* 顶部标题栏 */}
      <div className="bg-[#333b4f] p-3 border-b border-gray-600 flex justify-between items-center">
        <h2 className="font-bold text-[#8da5f5] flex items-center gap-2">
          <BoxSelect size={20} /> Rect2 交互实验室
        </h2>
        <span className="text-xs text-gray-400">Godot Engine Concept Simulator</span>
      </div>

      <div className="flex flex-1 overflow-hidden">
        
        {/* 左侧 Inspector 面板 */}
        <div className="w-80 bg-[#2d323c] p-4 border-r border-gray-700 flex flex-col gap-6 overflow-y-auto">
          
          {/* Properties Section */}
          <div>
            <h3 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2">
              <Move size={14} /> Position (位置)
            </h3>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 items-center text-sm">
              <label className="text-red-400 font-mono">x</label>
              <input 
                type="range" min="0" max="500" 
                value={rect.position.x} 
                onChange={(e) => setRect({...rect, position: {...rect.position, x: parseInt(e.target.value)}})}
                className="accent-[#8da5f5] w-full"
              />
              <label className="text-green-400 font-mono">y</label>
              <input 
                type="range" min="0" max="500" 
                value={rect.position.y} 
                onChange={(e) => setRect({...rect, position: {...rect.position, y: parseInt(e.target.value)}})}
                className="accent-[#8da5f5] w-full"
              />
            </div>
            <div className="mt-2 text-right font-mono text-xs text-gray-400">
              Vector2({rect.position.x}, {rect.position.y})
            </div>
          </div>

          <div>
            <h3 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2">
              <Maximize size={14} /> Size (大小)
            </h3>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 items-center text-sm">
              <label className="text-red-400 font-mono">w</label>
              <input 
                type="range" min="10" max="400" 
                value={rect.size.x} 
                onChange={(e) => setRect({...rect, size: {...rect.size, x: parseInt(e.target.value)}})}
                className="accent-[#8da5f5] w-full"
              />
              <label className="text-green-400 font-mono">h</label>
              <input 
                type="range" min="10" max="400" 
                value={rect.size.y} 
                onChange={(e) => setRect({...rect, size: {...rect.size, y: parseInt(e.target.value)}})}
                className="accent-[#8da5f5] w-full"
              />
            </div>
            <div className="mt-2 text-right font-mono text-xs text-gray-400">
              Vector2({rect.size.x}, {rect.size.y})
            </div>
          </div>

          {/* Computed Properties Section */}
          <div className="bg-[#212529] p-3 rounded border border-gray-600">
            <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Computed Props (只读)</h3>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-[#cba6f7]">end</span>
                <span>({end.x}, {end.y})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#cba6f7]">center</span>
                <span>({center.x}, {center.y})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#cba6f7]">area</span>
                <span>{area} px²</span>
              </div>
            </div>
          </div>

          {/* Functions Status */}
          <div className="bg-[#212529] p-3 rounded border border-gray-600">
            <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">API Returns</h3>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between items-center">
                <span className="text-[#89b4fa]">has_point(mouse)</span>
                <span className={`px-2 py-0.5 rounded ${isMouseOver ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                  {String(isMouseOver)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[#89b4fa]">intersects(enemy)</span>
                <span className={`px-2 py-0.5 rounded ${isColliding ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                  {String(isColliding)}
                </span>
              </div>
            </div>
          </div>

        </div>

        {/* 右侧视口 Viewport */}
        <div 
          ref={containerRef}
          className="flex-1 bg-[#202020] relative overflow-hidden cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseUp={() => setIsDraggingEnemy(false)}
          onMouseLeave={() => setIsDraggingEnemy(false)}
        >
          {/* 网格背景 */}
          <div 
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{ 
              backgroundImage: 'linear-gradient(#444 1px, transparent 1px), linear-gradient(90deg, #444 1px, transparent 1px)',
              backgroundSize: '20px 20px'
            }} 
          />
          
          {/* 坐标轴辅助线 */}
          <div className="absolute top-0 left-0 w-full h-[1px] bg-[#8da5f5] opacity-50 z-0"></div>
          <div className="absolute top-0 left-0 h-full w-[1px] bg-[#ff8080] opacity-50 z-0"></div>
          <div className="absolute top-2 left-2 text-xs text-gray-500 font-mono pointer-events-none">
            Origin (0,0) <br/> +Y Down ▼
          </div>

          {/* 主要的 Rect2 (蓝色) */}
          <div 
            className="absolute border-2 transition-colors duration-75 flex items-center justify-center group"
            style={{
              left: rect.position.x,
              top: rect.position.y,
              width: rect.size.x,
              height: rect.size.y,
              borderColor: isColliding ? '#ff4d4d' : '#8da5f5',
              backgroundColor: isMouseOver ? 'rgba(141, 165, 245, 0.2)' : 'rgba(141, 165, 245, 0.05)',
            }}
          >
            {/* Center Point */}
            <div className="absolute w-1.5 h-1.5 bg-[#cba6f7] rounded-full left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"></div>
            
            {/* Labels */}
            <span className="absolute -top-6 left-0 text-xs font-mono text-[#8da5f5] bg-[#212529] px-1 rounded">
              pos ({rect.position.x}, {rect.position.y})
            </span>
            <span className="absolute -bottom-6 right-0 text-xs font-mono text-[#cba6f7] bg-[#212529] px-1 rounded">
              end ({end.x}, {end.y})
            </span>
             <span className="text-[#8da5f5] opacity-0 group-hover:opacity-100 font-mono text-sm pointer-events-none">
               MyRect
            </span>
          </div>

          {/* 敌人 Rect (用于测试 Intersects) */}
          <div 
            className="absolute border-2 border-dashed border-gray-400 flex items-center justify-center cursor-move hover:bg-white/5"
            onMouseDown={() => setIsDraggingEnemy(true)}
            style={{
              left: enemyRect.position.x,
              top: enemyRect.position.y,
              width: enemyRect.size.x,
              height: enemyRect.size.y,
              borderColor: isColliding ? '#ff4d4d' : '#fab387',
              backgroundColor: isColliding ? 'rgba(255, 77, 77, 0.1)' : 'transparent',
            }}
          >
            <span className="text-[#fab387] font-mono text-xs pointer-events-none">
              {isDraggingEnemy ? "Dragging..." : "Drag Me (Other Rect)"}
            </span>
          </div>
          
           {/* Mouse Debugger */}
           <div 
            className="absolute pointer-events-none text-[10px] font-mono text-gray-400 bg-black/50 px-1 rounded whitespace-nowrap"
            style={{ left: mousePos.x + 10, top: mousePos.y + 10}}
           >
             ({Math.round(mousePos.x)}, {Math.round(mousePos.y)})
           </div>

        </div>
      </div>

      {/* 底部代码预览 */}
      <div className="bg-[#1e1e1e] p-3 border-t border-gray-700 font-mono text-xs overflow-x-auto">
        <div className="text-gray-500 mb-1"># GDScript Live Preview</div>
        <div className="text-[#dcdcaa]">
          <span className="text-[#569cd6]">var</span> my_rect = <span className="text-[#4ec9b0]">Rect2</span>(
          <span className="text-[#b5cea8]">{rect.position.x}</span>, <span className="text-[#b5cea8]">{rect.position.y}</span>, <span className="text-[#b5cea8]">{rect.size.x}</span>, <span className="text-[#b5cea8]">{rect.size.y}</span>
          )
        </div>
        <div className="mt-1">
          <span className="text-[#569cd6]">if</span> my_rect.<span className="text-[#dcdcaa]">has_point</span>(mouse_pos):
          <span className="text-[#6a9955]"> # Result: {String(isMouseOver)}</span>
        </div>
        <div>
          <span className="text-[#569cd6]">if</span> my_rect.<span className="text-[#dcdcaa]">intersects</span>(other_rect):
          <span className="text-[#6a9955]"> # Result: {String(isColliding)}</span>
        </div>
      </div>
    </div>
  );
};

export default GodotRect2Demo;