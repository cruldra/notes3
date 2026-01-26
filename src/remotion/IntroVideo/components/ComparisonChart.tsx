import React from 'react';
import { useCurrentFrame, useVideoConfig } from 'remotion';
import { growHeight } from '../utils/animations';

interface ComparisonItem {
  name: string;
  remotionScore: number;
  traditionalScore: number;
}

interface ComparisonChartProps {
  items: ComparisonItem[];
}

export const ComparisonChart: React.FC<ComparisonChartProps> = ({ items }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'flex-end',
        height: 600,
        width: '100%',
        padding: '0 100px',
        gap: 60,
      }}
    >
      {items.map((item, index) => {
        const delay = index * 15;
        const progress = growHeight(frame, delay, 45); // 0 to 100

        return (
          <div
            key={item.name}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              width: 150,
            }}
          >
            <div
              style={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'flex-end',
                height: 400,
                width: '100%',
                gap: 10,
              }}
            >
              {/* Remotion Bar */}
              <div
                style={{
                  width: '50%',
                  height: `${(item.remotionScore / 100) * progress}%`,
                  backgroundColor: '#0b84f3',
                  borderRadius: '8px 8px 0 0',
                  position: 'relative',
                }}
              >
                 <span style={{ 
                    position: 'absolute', 
                    top: -30, 
                    left: 0, 
                    right: 0, 
                    textAlign: 'center',
                    color: '#fff',
                    opacity: progress > 90 ? 1 : 0
                 }}>{item.remotionScore}</span>
              </div>
              {/* Traditional Bar */}
              <div
                style={{
                  width: '50%',
                  height: `${(item.traditionalScore / 100) * progress}%`,
                  backgroundColor: '#e94560',
                  borderRadius: '8px 8px 0 0',
                  position: 'relative',
                }}
              >
                <span style={{ 
                    position: 'absolute', 
                    top: -30, 
                    left: 0, 
                    right: 0, 
                    textAlign: 'center',
                    color: '#fff',
                    opacity: progress > 90 ? 1 : 0
                 }}>{item.traditionalScore}</span>
              </div>
            </div>
            <div
              style={{
                marginTop: 20,
                color: 'white',
                fontSize: 24,
                textAlign: 'center',
                fontFamily: 'Arial, sans-serif',
                fontWeight: 'bold',
              }}
            >
              {item.name}
            </div>
          </div>
        );
      })}
    </div>
  );
};
