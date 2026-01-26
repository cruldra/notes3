export interface Caption {
  startMs: number;
  endMs: number;
  text: string;
  timestampMs?: number;
}

export const subtitles: Caption[] = [
  // Intro (0-30s)
  { startMs: 2000, endMs: 14000, text: "欢迎了解 Remotion 框架" },
  { startMs: 15000, endMs: 28000, text: "用 React 创建视频的革命性工具" },

  // WhatIsRemotion (30-90s)
  { startMs: 32000, endMs: 40000, text: "Remotion 是什么?" },
  { startMs: 42000, endMs: 52000, text: "一个基于 React 的可编程视频创建框架" },
  { startMs: 54000, endMs: 64000, text: "用代码而非时间轴创建视频" },
  { startMs: 66000, endMs: 76000, text: "完全使用你熟悉的 Web 技术" },
  { startMs: 78000, endMs: 88000, text: "无需学习复杂的视频编辑软件" },

  // CoreFeatures (90-150s)
  { startMs: 92000, endMs: 100000, text: "核心特性" },
  { startMs: 102000, endMs: 110000, text: "React 组件化 - 像写 UI 一样写视频" },
  { startMs: 112000, endMs: 120000, text: "TypeScript 支持 - 类型安全" },
  { startMs: 122000, endMs: 130000, text: "完全可编程 - 用算法驱动动画" },
  { startMs: 132000, endMs: 140000, text: "本地渲染 - 无需渲染农场" },
  { startMs: 142000, endMs: 148000, text: "支持 MP4, WebM, GIF 等多种格式" },

  // UseCases (150-210s)
  { startMs: 152000, endMs: 160000, text: "实际应用场景" },
  { startMs: 162000, endMs: 170000, text: "自动化营销视频生成" },
  { startMs: 172000, endMs: 180000, text: "数据可视化视频" },
  { startMs: 182000, endMs: 190000, text: "教程和演示视频" },
  { startMs: 192000, endMs: 200000, text: "社交媒体内容批量生产" },
  { startMs: 202000, endMs: 208000, text: "动态个性化视频" },

  // Comparison (210-270s)
  { startMs: 212000, endMs: 220000, text: "与传统工具对比" },
  { startMs: 222000, endMs: 230000, text: "可编程性: Remotion 胜出" },
  { startMs: 232000, endMs: 240000, text: "批量生成: Remotion 胜出" },
  { startMs: 242000, endMs: 250000, text: "版本控制: Remotion 胜出" },
  { startMs: 252000, endMs: 260000, text: "学习曲线: 传统工具更友好" },
  { startMs: 262000, endMs: 268000, text: "适合程序员的视频创作方式" },

  // Outro (270-300s)
  { startMs: 272000, endMs: 280000, text: "用代码创造无限可能" },
  { startMs: 282000, endMs: 290000, text: "访问 remotion.dev 了解更多" },
  { startMs: 292000, endMs: 298000, text: "感谢观看!" },
];
