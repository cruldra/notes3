// @ts-nocheck
import React from 'react';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { slide } from '@remotion/transitions/slide';
import { Intro } from './scenes/Intro';
import { WhatIsRemotion } from './scenes/WhatIsRemotion';
import { CoreFeatures } from './scenes/CoreFeatures';
import { UseCases } from './scenes/UseCases';
import { Comparison } from './scenes/Comparison';
import { Outro } from './scenes/Outro';

export const IntroVideo: React.FC = () => {
  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={900}>
        <Intro />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      <TransitionSeries.Sequence durationInFrames={1800}>
        <WhatIsRemotion />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={slide({ direction: 'from-right' })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      <TransitionSeries.Sequence durationInFrames={1800}>
        <CoreFeatures />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      <TransitionSeries.Sequence durationInFrames={1800}>
        <UseCases />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={slide({ direction: 'from-left' })}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      <TransitionSeries.Sequence durationInFrames={1800}>
        <Comparison />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 30 })}
      />

      <TransitionSeries.Sequence durationInFrames={900}>
        <Outro />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
