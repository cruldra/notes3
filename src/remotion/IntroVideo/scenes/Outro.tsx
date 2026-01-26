import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { subtitles } from '../data/subtitles';
import { fadeIn, fadeOut, springScale } from '../utils/animations';

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 270000;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const logoScale = springScale(frame, fps, 10);
  const textOpacity = fadeIn(frame, 40, 30);
  
  // Fade out everything at the very end
  const contentOpacity = fadeOut(frame, 840, 30); // Last 1 second fade out

  return (
    <AbsoluteFill 
      style={{ 
        background: 'linear-gradient(to top left, #001f3f, #3a0ca3)',
        justifyContent: 'center',
        alignItems: 'center',
        opacity: contentOpacity 
      }}
    >
      <div style={{ transform: `scale(${logoScale})`, marginBottom: 60 }}>
        <div
          style={{
            width: 200,
            height: 200,
            borderRadius: 30,
            backgroundColor: '#0b84f3',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            fontSize: 100,
            color: 'white',
            fontWeight: 'bold',
            boxShadow: '0 0 60px rgba(11, 132, 243, 0.8)',
          }}
        >
          R
        </div>
      </div>

      <div style={{ textAlign: 'center', opacity: textOpacity }}>
        <h1
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 80,
            color: '#ffffff',
            marginBottom: 20,
          }}
        >
          用代码创造无限可能
        </h1>
        <h2
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 50,
            color: '#0b84f3',
            margin: 0,
            textDecoration: 'underline',
          }}
        >
          remotion.dev
        </h2>
      </div>

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
