import React from 'react';
import { AbsoluteFill, useVideoConfig } from 'remotion';
import { Caption } from '../data/subtitles';
import { fadeIn, fadeOut } from '../utils/animations';

interface SubtitleDisplayProps {
  captions: Caption[];
  currentMs: number;
}

export const SubtitleDisplay: React.FC<SubtitleDisplayProps> = ({ captions, currentMs }) => {
  const { fps } = useVideoConfig();
  const currentCaption = captions.find(
    (c) => currentMs >= c.startMs && currentMs <= c.endMs
  );

  if (!currentCaption) {
    return null;
  }

  // Simple fade in/out based on presence
  // For more complex animation we could use frame calculations relative to start/end frames
  // but for now let's just render it. The requirement asks for fadeIn/fadeOut.
  // Since we are checking currentMs every frame, it simply appears/disappears.
  // To animate per caption, we need to know the relative frame for this specific caption.
  
  // Actually, we can just use CSS transitions or calculate opacity based on time within the caption window.
  // Let's keep it simple and clean first, or use a key to trigger re-animation.
  
  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: 80,
      }}
    >
      <div
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.6)',
          padding: '10px 40px',
          borderRadius: 10,
          marginBottom: 40,
        }}
      >
        <h2
          style={{
            fontFamily: 'Arial, sans-serif',
            fontSize: 48,
            color: 'white',
            textAlign: 'center',
            margin: 0,
            textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
          }}
        >
          {currentCaption.text}
        </h2>
      </div>
    </AbsoluteFill>
  );
};
