import React, { useState, useEffect } from 'react';
import { 
  GitMerge, 
  GitBranch, 
  GitPullRequest, 
  ArrowRight, 
  Copy, 
  Check, 
  AlertTriangle, 
  Terminal, 
  ShieldAlert,
  Save,
  FileCode,
  Settings,
  BookOpen,
  Anchor
} from 'lucide-react';

// 通用组件 - 按钮
const Button = ({ children, onClick, variant = 'primary', className = '' }) => {
  const baseStyle = "px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2";
  const variants = {
    primary: "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/30",
    secondary: "bg-slate-700 hover:bg-slate-600 text-slate-200",
    outline: "border border-slate-600 text-slate-400 hover:text-slate-200 hover:border-slate-500"
  };
  
  return (
    <button onClick={onClick} className={`${baseStyle} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
};

// 通用组件 - 代码块
const CodeBlock = ({ code, label }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-800 my-3 font-mono text-sm shadow-inner group">
      {label && (
        <div className="bg-slate-800/50 px-4 py-1 text-xs text-slate-400 border-b border-slate-800 flex justify-between items-center">
          <span>{label}</span>
          <span className="text-[10px] opacity-50">bash</span>
        </div>
      )}
      <div className="p-4 relative">
        <pre className="text-slate-300 overflow-x-auto whitespace-pre-wrap break-all pr-12">
          {code}
        </pre>
        <button 
          onClick={handleCopy}
          className="absolute top-3 right-3 p-2 bg-slate-800 hover:bg-slate-700 rounded-md text-slate-400 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
        >
          {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
        </button>
      </div>
    </div>
  );
};

// 冲突卡片
const ConflictCard = ({ title, context, strategy, icon: Icon, color }) => (
  <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-5 hover:border-slate-600 transition-colors">
    <div className="flex items-start gap-4">
      <div className={`p-3 rounded-lg bg-${color}-500/10 text-${color}-400 shrink-0`}>
        <Icon size={24} />
      </div>
      <div>
        <h3 className="text-lg font-bold text-slate-100 mb-1">{title}</h3>
        <p className="text-sm text-slate-400 mb-3">{context}</p>
        <div className={`text-sm bg-${color}-500/10 text-${color}-300 p-2 rounded border border-${color}-500/20`}>
          <strong>解法：</strong> {strategy}
        </div>
      </div>
    </div>
  </div>
);

// Git 抽象可视化 - 通用版
const GitVisualizer = () => {
  const [mode, setMode] = useState('rebase'); 

  return (
    <div className="w-full bg-slate-900 rounded-xl p-6 border border-slate-800 mb-8 overflow-hidden relative min-h-[280px]">
      <div className="flex justify-between items-center mb-12 z-20 relative">
        <h3 className="text-slate-400 font-medium flex items-center gap-2">
          <GitBranch size={18} /> 上游 vs. Fork 演变图
        </h3>
        <div className="flex gap-2 bg-slate-800 p-1 rounded-lg">
          <button 
            onClick={() => setMode('messy')}
            className={`px-3 py-1 rounded text-sm transition-colors ${mode === 'messy' ? 'bg-red-500/20 text-red-400' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Merge (混乱)
          </button>
          <button 
            onClick={() => setMode('rebase')}
            className={`px-3 py-1 rounded text-sm transition-colors ${mode === 'rebase' ? 'bg-indigo-500/20 text-indigo-400' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Rebase (推荐)
          </button>
        </div>
      </div>

      <div className="relative h-32 flex items-center justify-center">
        {/* 基础线 (Upstream) */}
        <div className="absolute left-10 flex items-center gap-12 top-1/2 -translate-y-1/2">
           {/* 初始点 */}
           <div className="flex flex-col items-center">
             <div className="w-4 h-4 rounded-full bg-slate-600"></div>
             <span className="absolute -bottom-6 text-xs text-slate-600">v1.0</span>
           </div>
           <div className="w-16 h-0.5 bg-slate-700"></div>
           
           {/* 上游新更新 */}
           <div className="flex flex-col items-center relative z-10">
             <div className="w-5 h-5 rounded-full bg-indigo-500 border-2 border-slate-900 shadow-[0_0_15px_rgba(99,102,241,0.5)]"></div>
             <span className="absolute -top-8 text-xs text-indigo-400 font-bold w-32 text-center">上游更新 (v1.1)</span>
             <span className="absolute -bottom-8 text-[10px] text-slate-500 w-32 text-center">官方修复 Bug / 新功能</span>
           </div>
        </div>

        {/* 你的分支 (Origin) */}
        <div className={`transition-all duration-700 ease-in-out absolute left-[220px] top-1/2 -translate-y-1/2 ${
          mode === 'rebase' ? 'translate-x-[60px] translate-y-0' : 'translate-x-0 translate-y-[-60px]'
        }`}>
          <div className="flex items-center">
             {/* 连线 */}
             <div className={`h-0.5 bg-green-500/50 transition-all duration-500 origin-left ${
               mode === 'rebase' ? 'w-12 rotate-0' : 'w-12 rotate-[45deg] translate-y-[14px]'
             }`}></div>

             {/* 你的提交 */}
             <div className="flex flex-col items-center relative">
               <div className="w-5 h-5 rounded-full bg-green-500 border-2 border-slate-900 shadow-[0_0_15px_rgba(34,197,94,0.5)]"></div>
               <span className="absolute -top-8 text-xs text-green-400 font-bold whitespace-nowrap">你的改动 (MyFork)</span>
               <span className="absolute -bottom-8 text-[10px] text-slate-500 whitespace-nowrap bg-slate-900/80 px-1 rounded">
                 你的定制功能 / Hack
               </span>
             </div>
          </div>
        </div>

        {/* Merge 的那条乱线 */}
        {mode === 'messy' && (
          <div className="absolute left-[340px] top-1/2 -translate-y-1/2 animate-pulse opacity-60">
             {/* 菱形结构模拟 */}
             <div className="w-24 h-[60px] border-r-2 border-dashed border-slate-600 rounded-r-[50px] absolute top-[-30px] left-[-40px]"></div>
             <div className="absolute right-[-10px] flex flex-col items-center">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span className="text-[10px] text-red-500 mt-1">Merge Commit</span>
             </div>
          </div>
        )}
      </div>

      <div className="mt-12 text-center text-sm">
        {mode === 'rebase' ? (
          <p className="text-indigo-300 bg-indigo-500/10 p-2 rounded inline-block border border-indigo-500/20">
            ✨ <strong>Rebase 效果：</strong> 相当于把你做的修改“拔下来”，等上游更新完，再“插回去”。<br/>你的修改永远是最新的补丁。
          </p>
        ) : (
          <p className="text-slate-400 bg-slate-800 p-2 rounded inline-block">
            🌪️ <strong>Merge 效果：</strong> 历史分叉再合并，产生无意义的 "Merge remote-tracking branch..." 节点。<br/>一旦冲突，很难理清是谁覆盖了谁。
          </p>
        )}
      </div>
    </div>
  );
};

export default function App() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: "定义你的战场 (Setup Remote)",
      desc: "Git 默认只知道你自己的仓库 (origin)，你需要告诉它原作者的仓库 (upstream) 在哪。",
      code: `git remote add upstream https://github.com/原作者/项目名.git\n\n# 验证一下\ngit remote -v`,
      tips: "origin = 你的 Fork; upstream = 官方原版。"
    },
    {
      title: "拉取并变基 (Fetch & Rebase)",
      desc: "这是日常维护最核心的命令。不要用 pull，要用 fetch + rebase。",
      code: `git fetch upstream\n\n# 切换到你的主分支\ngit checkout main\n\n# 施展魔法：把我的修改暂存，更新地基，再重演我的修改\ngit rebase upstream/main`,
      tips: "Rebase = Re-Base (重新更换基地)。"
    },
    {
      title: "处理冲突 (The Reality)",
      desc: "当上游改了 A 文件，你也改了 A 文件，Git 就会停下来问你：听谁的？",
      customContent: (
        <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 my-2">
          <div className="grid grid-cols-2 gap-4 text-sm mb-2">
            <div className="bg-slate-900 p-2 rounded text-slate-400">
              <span className="block text-xs text-slate-500 mb-1">Incoming / Theirs</span>
              官方的新代码
            </div>
            <div className="bg-indigo-900/30 p-2 rounded text-indigo-300 border border-indigo-500/30">
              <span className="block text-xs text-indigo-400 mb-1">Current / Ours</span>
              你的修改
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            *注意：在 rebase 过程中，"Ours" 和 "Theirs" 的定义有时会反直觉，建议直接看代码内容判断。
          </p>
        </div>
      )
    },
    {
      title: "解决后继续 (Continue)",
      desc: "解决完一个文件的冲突后，告诉 Git 继续处理下一个提交。",
      code: `git add <冲突的文件名>\n\n# 不需要 commit，直接 continue\ngit rebase --continue`,
      tips: "如果在中间搞砸了，随时可以用 `git rebase --abort` 回到开始前的状态。"
    },
    {
      title: "强制推送 (Force Push)",
      desc: "因为 Rebase 修改了历史时间线，你必须强制覆盖你 GitHub 上的旧记录。",
      code: `git push -f origin main`,
      tips: "Force Push 在多人合作的 Shared Branch 是禁忌，但在你自己维护的 Fork 里是常规操作。"
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30 pb-20">
      <div className="max-w-4xl mx-auto px-6 py-12">
        
        {/* Header */}
        <header className="mb-12 text-center">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-6 ring-1 ring-indigo-500/30">
            <Anchor className="text-indigo-400 mr-2" size={20} />
            <span className="text-indigo-400 font-bold tracking-wider uppercase text-sm">Fork Maintenance Guide</span>
          </div>
          <h1 className="text-3xl md:text-5xl font-extrabold text-white mb-6 leading-tight">
            如何优雅地维护<br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">
              开源项目 Fork
            </span>
          </h1>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg">
            解决 "Upstream 更新了，我的 Fork 冲突了" 的终极指南。<br/>
            核心心法：<span className="text-indigo-400 font-mono bg-indigo-900/30 px-2 py-0.5 rounded">Rebase Always</span>
          </p>
        </header>

        {/* Visualization */}
        <GitVisualizer />

        {/* Conflict Scenarios */}
        <section className="mb-16 grid md:grid-cols-2 gap-6">
          <ConflictCard 
            title="配置/版本冲突"
            context="例如：上游把依赖 react 升到了 v18，而你的 package.json 还在用 v17 并加了其他库。"
            strategy="通常接受上游 (Upstream) 的版本号，然后手动把你加的库补回去。"
            icon={Settings}
            color="orange"
          />
          <ConflictCard 
            title="核心逻辑冲突"
            context="例如：上游重构了 Login 函数，而你在旧的 Login 函数里加了验证码功能。"
            strategy="这是最难的。你需要理解上游的新逻辑，然后把你的验证码功能“移植”到新函数里。"
            icon={FileCode}
            color="red"
          />
        </section>

        {/* Workflow Steps */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <Terminal className="text-indigo-400" />
            <h2 className="text-2xl font-bold text-white">通用维护流程</h2>
          </div>
          
          <div className="space-y-4">
            {steps.map((step, index) => (
              <div 
                key={index}
                className={`transition-all duration-300 border rounded-xl overflow-hidden ${
                  activeStep === index 
                    ? 'bg-slate-900 border-indigo-500/50 shadow-lg shadow-indigo-500/10' 
                    : 'bg-slate-900/30 border-slate-800 hover:border-slate-700 opacity-60 hover:opacity-100'
                }`}
              >
                <div 
                  className="p-5 cursor-pointer flex items-center gap-4"
                  onClick={() => setActiveStep(index)}
                >
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                    activeStep === index ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div className="flex-grow">
                    <h3 className={`font-bold text-lg ${activeStep === index ? 'text-indigo-100' : 'text-slate-400'}`}>
                      {step.title}
                    </h3>
                  </div>
                  <ArrowRight 
                    className={`transition-transform duration-300 ${activeStep === index ? 'text-indigo-400 rotate-90' : 'text-slate-600'}`} 
                    size={20} 
                  />
                </div>

                {activeStep === index && (
                  <div className="px-5 pb-6 pl-[4.5rem] animate-fadeIn">
                    <p className="text-slate-300 mb-4 leading-relaxed">{step.desc}</p>
                    {step.customContent}
                    {step.code && <CodeBlock code={step.code} />}
                    {step.tips && (
                      <div className="mt-4 flex gap-2 text-xs text-slate-500 bg-slate-950/50 p-3 rounded border border-slate-800/50 items-start">
                        <BookOpen size={14} className="mt-0.5 shrink-0" />
                        <span>{step.tips}</span>
                      </div>
                    )}
                    
                    <div className="mt-6 flex gap-3">
                      {index < steps.length - 1 && (
                        <Button onClick={(e) => { e.stopPropagation(); setActiveStep(index + 1); }}>
                          下一步
                        </Button>
                      )}
                      {index === steps.length - 1 && (
                        <Button variant="outline" onClick={(e) => { e.stopPropagation(); setActiveStep(0); }}>
                          重头再来
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Golden Rules */}
        <section className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 rounded-2xl p-8 border border-indigo-500/20">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <ShieldAlert className="text-yellow-400" />
            维护 Fork 的三大黄金法则
          </h2>
          <div className="grid md:grid-cols-3 gap-6 text-sm">
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800">
              <strong className="text-indigo-300 block mb-2">1. 经常 Rebase</strong>
              <p className="text-slate-400">不要等上游积攒了 100 个提交才同步。差距越大，冲突越难解。建议每周或每两周同步一次。</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800">
              <strong className="text-indigo-300 block mb-2">2. 保持改动原子化</strong>
              <p className="text-slate-400">你的修改最好集中在独立的 Feature 分支，或者独立的几个提交里。不要在 formatting 或空格上和上游纠缠。</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800">
              <strong className="text-indigo-300 block mb-2">3. 读懂上游的意图</strong>
              <p className="text-slate-400">解决冲突不是简单的“保留我的”或“保留他的”。如果是核心逻辑冲突，你必须理解上游为什么要改这段代码。</p>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}