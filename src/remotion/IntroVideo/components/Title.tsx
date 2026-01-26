import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { springScale, fadeIn } from '../utils/animations';

interface TitleProps {
  text: string;
  subtitle?: string;
  animationStart?: number;
  color?: string;
  subtitleColor?: string;
}

export const Title: React.FC<TitleProps> = ({
  text,
  subtitle,
  animationStart = 0,
  color = '#ffffff',
  subtitleColor = '#0b84f3',
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = springScale(frame, fps, animationStart);
  const subtitleOpacity = fadeIn(frame, animationStart + 20, 30);

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        flexDirection: 'column',
      }}
    >
      <h1
        style={{
          fontFamily: 'Arial, sans-serif',
          fontSize: 100,
          fontWeight: 'bold',
          color: color,
          margin: 0,
          transform: `scale(${scale})`,
          textShadow: '0 4px 10px rgba(0,0,0,0.3)',
        }}
      >
        {text}
      </h1>
      {subtitle && (
        <h2
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 50,
            fontWeight: 'normal',
            color: subtitleColor,
            marginTop: 20,
            opacity: subtitleOpacity,
          }}
        >
          {subtitle}
        </h2>
      )}
    </AbsoluteFill>
  );
};
