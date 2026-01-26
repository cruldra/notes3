import { interpolate, spring, Easing } from 'remotion';

export const fadeIn = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

export const fadeOut = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

export const slideInFromLeft = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [-100, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.ease),
  });
};

export const slideInFromRight = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [100, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.ease),
  });
};

export const slideInFromBottom = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [100, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.ease),
  });
};

export const springScale = (frame: number, fps: number, delay: number = 0) => {
  return spring({
    frame: frame - delay,
    fps,
    config: {
      damping: 12,
      stiffness: 100,
    },
  });
};

export const growHeight = (frame: number, startFrame: number, duration: number) => {
  return interpolate(frame, [startFrame, startFrame + duration], [0, 100], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.exp),
  });
};
