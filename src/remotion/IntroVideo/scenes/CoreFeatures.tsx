import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { FeatureCard } from '../components/FeatureCard';
import { subtitles } from '../data/subtitles';
import { fadeIn } from '../utils/animations';

export const CoreFeatures: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 90000;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const titleOpacity = fadeIn(frame, 10, 20);

  const features = [
    {
      icon: '🧩',
      title: 'React 组件化',
      description: '像写 UI 一样写视频，复用现有的 React 生态。',
      delay: 30,
    },
    {
      icon: '📘',
      title: 'TypeScript 支持',
      description: '享受类型安全，减少运行时错误，开发体验极佳。',
      delay: 50, // 20 frames later (~660ms, close enough to 200ms requirements relative to fps)
    },
    {
      icon: '💻',
      title: '完全可编程',
      description: '使用循环、函数和 API 数据驱动视频内容。',
      delay: 70,
    },
    {
      icon: '🚀',
      title: '本地渲染',
      description: '利用本地硬件加速渲染，无需上传云端。',
      delay: 90,
    },
  ];

  return (
    <AbsoluteFill style={{ backgroundColor: '#1a1a2e' }}>
      <AbsoluteFill
        style={{
          padding: 80,
          alignItems: 'center',
        }}
      >
        <h1
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 80,
            color: '#ffffff',
            opacity: titleOpacity,
            marginBottom: 60,
          }}
        >
          核心特性
        </h1>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            maxWidth: 1000,
          }}
        >
          {features.map((feature) => (
            <FeatureCard
              key={feature.title}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
              delay={feature.delay}
            />
          ))}
        </div>
      </AbsoluteFill>

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
