import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from 'remotion';
import { SubtitleDisplay } from '../components/SubtitleDisplay';
import { subtitles } from '../data/subtitles';
import { fadeIn, slideInFromBottom } from '../utils/animations';

export const WhatIsRemotion: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const offsetMs = 30000;
  const currentMs = (frame / fps) * 1000 + offsetMs;

  const titleOpacity = fadeIn(frame, 10, 20);
  const codeOpacity = fadeIn(frame, 60, 30);
  const codeSlide = slideInFromBottom(frame, 60, 30);

  const codeSnippet = `import { Composition } from 'remotion';
import { MyVideo } from './MyVideo';

export const RemotionVideo: React.FC = () => {
  return (
    <Composition
      id="MyVideo"
      component={MyVideo}
      durationInFrames={150}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};`;

  return (
    <AbsoluteFill style={{ backgroundColor: '#1a1a2e' }}>
      <AbsoluteFill
        style={{
          justifyContent: 'center',
          alignItems: 'center',
          flexDirection: 'column',
          paddingTop: 100,
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
          Remotion 是什么?
        </h1>

        <div
          style={{
            opacity: codeOpacity,
            transform: `translateY(${codeSlide}px)`,
            backgroundColor: '#282c34',
            padding: 40,
            borderRadius: 20,
            boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
            maxWidth: '80%',
          }}
        >
          <pre
            style={{
              fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
              fontSize: 24,
              color: '#abb2bf',
              margin: 0,
              textAlign: 'left',
              whiteSpace: 'pre-wrap',
            }}
          >
            {codeSnippet.split('\n').map((line, i) => (
              <div key={i}>
                {line
                  .replace('Composition', '⭐⭐Composition⭐⭐')
                  .split('⭐⭐')
                  .map((part, j) =>
                    part === 'Composition' ? (
                      <span key={j} style={{ color: '#61afef', fontWeight: 'bold' }}>
                        {part}
                      </span>
                    ) : (
                      <span key={j}>{part}</span>
                    )
                  )}
              </div>
            ))}
          </pre>
        </div>
      </AbsoluteFill>

      <SubtitleDisplay captions={subtitles} currentMs={currentMs} />
    </AbsoluteFill>
  );
};
