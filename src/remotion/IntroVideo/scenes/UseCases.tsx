import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { FeatureCard } from '../components/FeatureCard';
import { subtitles } from '../data/subtitles';
import { fadeIn } from '../utils/animations';

export const UseCases: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 150000;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const titleOpacity = fadeIn(frame, 10, 20);

  const cases = [
    {
      icon: '📢',
      title: '营销视频',
      description: '批量生成个性化营销视频，提高转化率。',
      delay: 30,
    },
    {
      icon: '📊',
      title: '数据可视化',
      description: '将复杂数据转化为动态、易懂的视频图表。',
      delay: 50,
    },
    {
      icon: '🎓',
      title: '教程视频',
      description: '自动生成代码演示和操作指南。',
      delay: 70,
    },
    {
      icon: '📱',
      title: '社交媒体',
      description: '快速适配不同平台的短视频内容。',
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
          实际应用场景
        </h1>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            maxWidth: 1000,
          }}
        >
          {cases.map((c) => (
            <FeatureCard
              key={c.title}
              icon={c.icon}
              title={c.title}
              description={c.description}
              delay={c.delay}
            />
          ))}
        </div>
      </AbsoluteFill>

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
