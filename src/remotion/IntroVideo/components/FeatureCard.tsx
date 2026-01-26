import React from 'react';
import { useCurrentFrame, useVideoConfig } from 'remotion';
import { slideInFromLeft } from '../utils/animations';

interface FeatureCardProps {
  icon: string;
  title: string;
  description: string;
  delay: number;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({
  icon,
  title,
  description,
  delay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const translateX = slideInFromLeft(frame, delay, 30);
  const opacity = frame < delay ? 0 : 1;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        padding: 40,
        borderRadius: 20,
        width: 400,
        height: 300,
        transform: `translateX(${translateX}%)`,
        opacity,
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        margin: 20,
      }}
    >
      <div style={{ fontSize: 60, marginBottom: 20 }}>{icon}</div>
      <h3
        style={{
          fontSize: 32,
          color: '#0b84f3',
          marginBottom: 10,
          fontFamily: 'Arial, sans-serif',
        }}
      >
        {title}
      </h3>
      <p
        style={{
          fontSize: 24,
          color: '#ffffff',
          lineHeight: 1.5,
          fontFamily: 'Arial, sans-serif',
        }}
      >
        {description}
      </p>
    </div>
  );
};
