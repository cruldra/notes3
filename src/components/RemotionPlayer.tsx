import React, { useMemo } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

interface RemotionPlayerProps {
  component: React.ComponentType<any>;
  durationInFrames: number;
  fps: number;
  compositionWidth: number;
  compositionHeight: number;
  inputProps?: Record<string, any>;
  controls?: boolean;
  loop?: boolean;
  style?: React.CSSProperties;
}

export const RemotionPlayer: React.FC<RemotionPlayerProps> = ({
  component,
  durationInFrames,
  fps,
  compositionWidth,
  compositionHeight,
  inputProps,
  controls = true,
  loop = true,
  style,
}) => {
  const memoizedInputProps = useMemo(() => inputProps || {}, [inputProps]);
  const playerStyle = useMemo(
    () => ({
      width: '100%',
      maxWidth: compositionWidth,
      margin: '0 auto',
      borderRadius: '8px',
      overflow: 'hidden',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      ...style,
    }),
    [compositionWidth, style]
  );

  return (
    <BrowserOnly fallback={<div>加载播放器中...</div>}>
      {() => {
        const { Player } = require('@remotion/player');
        return (
          <Player
            component={component}
            durationInFrames={durationInFrames}
            fps={fps}
            compositionWidth={compositionWidth}
            compositionHeight={compositionHeight}
            inputProps={memoizedInputProps}
            controls={controls}
            loop={loop}
            style={playerStyle}
          />
        );
      }}
    </BrowserOnly>
  );
};
