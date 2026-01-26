import { Composition } from 'remotion';
import { IntroVideo } from './IntroVideo';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="remotion-intro"
        component={IntroVideo}
        durationInFrames={9150} // Adjusted for transitions: 9000 + 5*30
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
