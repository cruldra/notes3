import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { ComparisonChart } from '../components/ComparisonChart';
import { subtitles } from '../data/subtitles';
import { fadeIn } from '../utils/animations';

export const Comparison: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 210000;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const titleOpacity = fadeIn(frame, 10, 20);

  const comparisonData = [
    { name: '可编程性', remotionScore: 95, traditionalScore: 30 },
    { name: '批量生成', remotionScore: 90, traditionalScore: 20 },
    { name: '版本控制', remotionScore: 100, traditionalScore: 10 },
    { name: '学习曲线', remotionScore: 60, traditionalScore: 90 },
  ];

  return (
    <AbsoluteFill style={{ backgroundColor: '#1a1a2e' }}>
      <AbsoluteFill
        style={{
          paddingTop: 100,
          alignItems: 'center',
        }}
      >
        <h1
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 80,
            color: '#ffffff',
            opacity: titleOpacity,
            marginBottom: 80,
          }}
        >
          与传统工具对比
        </h1>

        <div style={{ width: '100%', maxWidth: 1200 }}>
             <ComparisonChart items={comparisonData} />
             
             <div style={{ display: 'flex', justifyContent: 'center', marginTop: 40, gap: 40 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ width: 20, height: 20, backgroundColor: '#0b84f3', borderRadius: 4 }}></div>
                    <span style={{ color: 'white', fontSize: 24, fontFamily: 'Arial' }}>Remotion</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ width: 20, height: 20, backgroundColor: '#e94560', borderRadius: 4 }}></div>
                    <span style={{ color: 'white', fontSize: 24, fontFamily: 'Arial' }}>传统工具</span>
                </div>
             </div>
        </div>

      </AbsoluteFill>

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
