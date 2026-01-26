import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, Img, staticFile } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { Title } from '../components/Title';
import { subtitles } from '../data/subtitles';
import { springScale, fadeIn } from '../utils/animations';

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 0;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const logoOpacity = fadeIn(frame, 10, 30);
  const logoScale = springScale(frame, fps, 10);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(to bottom right, #001f3f, #3a0ca3)',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div style={{ opacity: logoOpacity, transform: `scale(${logoScale})`, marginBottom: 400 }}>
        {/* Placeholder for Logo since we can't use images yet, using text or simple svg shape */}
        <div
          style={{
            width: 150,
            height: 150,
            borderRadius: 20,
            backgroundColor: '#0b84f3',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            fontSize: 80,
            color: 'white',
            fontWeight: 'bold',
            boxShadow: '0 0 40px rgba(11, 132, 243, 0.6)',
          }}
        >
          R
        </div>
      </div>

      <Title 
        text="用 React 创建视频" 
        subtitle="Remotion 框架介绍" 
        animationStart={30}
      />

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
