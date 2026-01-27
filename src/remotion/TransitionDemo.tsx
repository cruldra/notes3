import { AbsoluteFill, useCurrentFrame, interpolate, useVideoConfig } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { slide } from '@remotion/transitions/slide';
import { wipe } from '@remotion/transitions/wipe';
import { flip } from '@remotion/transitions/flip';
import { clockWipe } from '@remotion/transitions/clock-wipe';

/**
 * 场景卡片组件
 * 展示当前场景信息和即将使用的过渡效果
 */
const SceneCard: React.FC<{
  color: string;
  label: string;
  nextTransition: string;
  description?: string;
}> = ({ color, label, nextTransition, description }) => {
  const frame = useCurrentFrame();

  // 标题缩放动画
  const scale = interpolate(frame, [0, 15], [0.8, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // 文字淡入动画
  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(135deg, ${color} 0%, ${adjustColor(
          color,
          -30
        )} 100%)`,
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      <div
        style={{
          textAlign: 'center',
          opacity,
          transform: `scale(${scale})`,
        }}
      >
        {/* 场景标题 */}
        <h1
          style={{
            fontSize: 140,
            fontWeight: 'bold',
            color: '#ffffff',
            margin: 0,
            textShadow: '0 4px 20px rgba(0,0,0,0.3)',
          }}
        >
          {label}
        </h1>

        {/* 场景描述 */}
        {description && (
          <p
            style={{
              fontSize: 36,
              color: '#f0f0f0',
              marginTop: 20,
              marginBottom: 40,
            }}
          >
            {description}
          </p>
        )}

        {/* 下一个过渡效果提示 */}
        <div
          style={{
            marginTop: 60,
            padding: '20px 40px',
            backgroundColor: 'rgba(0,0,0,0.3)',
            borderRadius: '50px',
            display: 'inline-block',
          }}
        >
          <p
            style={{
              fontSize: 32,
              color: '#ffffff',
              margin: 0,
              fontWeight: '500',
            }}
          >
            下一个过渡: <strong>{nextTransition}</strong>
          </p>
        </div>
      </div>
    </AbsoluteFill>
  );
};

/**
 * 最后一个场景 - 总结卡片
 */
const SummaryCard: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      <div
        style={{
          textAlign: 'center',
          opacity,
        }}
      >
        <h1
          style={{
            fontSize: 100,
            fontWeight: 'bold',
            color: '#ffffff',
            margin: 0,
            textShadow: '0 4px 20px rgba(0,0,0,0.3)',
          }}
        >
          🎬 过渡效果演示完成
        </h1>
        <p
          style={{
            fontSize: 42,
            color: '#f0f0f0',
            marginTop: 30,
          }}
        >
          查看下方文档了解更多过渡效果
        </p>
      </div>
    </AbsoluteFill>
  );
};

/**
 * Remotion 过渡效果演示组件
 * 
 * 展示多种常用的过渡效果：
 * - Fade（淡入淡出）
 * - Slide（滑动）
 * - Wipe（擦除）
 * - Flip（翻转）
 * - Clock Wipe（时钟擦除）
 * 
 * 总时长计算 (durationInFrames):
 * 6 个序列 (每个 95 帧) - 5 个过渡 (每个 30 帧) = 6 * 95 - 5 * 30 = 570 - 150 = 420 帧
 * 在 30 fps 下，总时长为 14 秒，与 MDX 配置一致。
 */
export const TransitionDemo: React.FC = () => {
  const { width, height } = useVideoConfig();
  
  return (
    <TransitionSeries>
      {/* 场景 1 - 介绍 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SceneCard
          color="#0b84f3"
          label="场景 1"
          nextTransition="Fade（淡入淡出）"
          description="欢迎来到 Remotion 过渡效果演示"
        />
      </TransitionSeries.Sequence>

      {/* Fade 过渡 */}
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      {/* 场景 2 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SceneCard
          color="#f093fb"
          label="场景 2"
          nextTransition="Slide（滑动）"
          description="平滑的透明度过渡"
        />
      </TransitionSeries.Sequence>

      {/* Slide 过渡 - 从右向左滑动 */}
      <TransitionSeries.Transition
        presentation={slide({ direction: 'from-right' })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      {/* 场景 3 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SceneCard
          color="#4facfe"
          label="场景 3"
          nextTransition="Wipe（擦除）"
          description="滑入并推出前一个场景"
        />
      </TransitionSeries.Sequence>

      {/* Wipe 过渡 - 从上到下擦除 */}
      <TransitionSeries.Transition
        presentation={wipe({ direction: 'from-top' })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      {/* 场景 4 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SceneCard
          color="#43e97b"
          label="场景 4"
          nextTransition="Flip（翻转）"
          description="滑动覆盖前一个场景"
        />
      </TransitionSeries.Sequence>

      {/* Flip 过渡 - 从左翻转 */}
      <TransitionSeries.Transition
        presentation={flip({ direction: 'from-left' })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      {/* 场景 5 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SceneCard
          color="#fa709a"
          label="场景 5"
          nextTransition="Clock Wipe（时钟擦除）"
          description="3D 透视翻转效果"
        />
      </TransitionSeries.Sequence>

      {/* Clock Wipe 过渡 - 顺时针 */}
      <TransitionSeries.Transition
        presentation={clockWipe({ width, height })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      {/* 最后一个场景 - 总结 */}
      <TransitionSeries.Sequence durationInFrames={95}>
        <SummaryCard />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};

/**
 * 辅助函数：调整颜色亮度
 * @param color - 十六进制颜色值（如 "#667eea"）
 * @param amount - 调整量（-100 到 100）
 */
function adjustColor(color: string, amount: number): string {
  const hex = color.replace('#', '');
  const r = Math.max(
    0,
    Math.min(255, parseInt(hex.substring(0, 2), 16) + amount)
  );
  const g = Math.max(
    0,
    Math.min(255, parseInt(hex.substring(2, 4), 16) + amount)
  );
  const b = Math.max(
    0,
    Math.min(255, parseInt(hex.substring(4, 6), 16) + amount)
  );
  return `#${r.toString(16).padStart(2, '0')}${g
    .toString(16)
    .padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
