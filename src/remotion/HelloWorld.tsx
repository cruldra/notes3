import { AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';

export const HelloWorld: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const translateY = interpolate(frame, [0, 30], [50, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const scale = interpolate(frame, [30, 60], [1, 1.1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#1a1a2e',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          opacity,
          transform: `translateY(${translateY}px) scale(${scale})`,
          transition: 'transform 0.3s ease',
        }}
      >
        <h1
          style={{
            fontSize: 100,
            fontWeight: 'bold',
            color: '#ffffff',
            textAlign: 'center',
            margin: 0,
            textShadow: '0 0 20px rgba(255,255,255,0.5)',
          }}
        >
          Hello Remotion!
        </h1>
        <p
          style={{
            fontSize: 30,
            color: '#e94560',
            textAlign: 'center',
            marginTop: 20,
          }}
        >
          这是一个简单的Remotion示例
        </p>
      </div>
    </AbsoluteFill>
  );
};
