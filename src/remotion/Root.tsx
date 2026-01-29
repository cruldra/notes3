import { Composition } from 'remotion';
import { IntroVideo } from './IntroVideo';
import { TransitionDemo } from './TransitionDemo';
import { MantineDemo } from './MantineDemo';
import '@mantine/core/styles.css';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="mantine-demo"
        component={MantineDemo}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="remotion-intro"

        component={IntroVideo}
        durationInFrames={9150} // Adjusted for transitions: 9000 + 5*30
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="transition-demo"
        component={TransitionDemo}
        durationInFrames={420} // 6 scenes * 60 frames + 5 transitions * 30 frames = 360 + 150 = 510 (adjusted to 420 for actual calculation)
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
