import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  MantineProvider,
  Title,
  Text,
  Badge,
  Card,
  Group,
  Stack,
  Button,
  createTheme,
} from '@mantine/core';

const theme = createTheme({
  primaryColor: 'blue',
});

export const MantineDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // Animations
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const cardScale = spring({
    frame: frame - 20,
    fps,
    config: {
      damping: 12,
    },
  });

  const badgeTranslate = interpolate(frame, [40, 60], [50, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <MantineProvider theme={theme} defaultColorScheme="light">
      <AbsoluteFill
        style={{
          backgroundColor: '#f8f9fa',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Stack align="center" gap="xl" style={{ width: '60%' }}>
          <Title
            order={1}
            style={{
              opacity: titleOpacity,
              transform: `translateY(${interpolate(frame, [0, 30], [20, 0])}px)`,
              fontSize: 80,
              textAlign: 'center',
            }}
          >
            Remotion + Mantine
          </Title>

          <Card
            shadow="lg"
            padding="xl"
            radius="md"
            withBorder
            style={{
              transform: `scale(${cardScale})`,
              opacity: cardScale,
              width: '100%',
              backgroundColor: 'white',
            }}
          >
            <Stack gap="md">
              <Group justify="space-between">
                <Text fw={700} size="xl">
                  极速开发体验
                </Text>
                <Badge
                  color="blue"
                  variant="light"
                  size="lg"
                  style={{ transform: `translateX(${badgeTranslate}px)` }}
                >
                  NEW
                </Badge>
              </Group>

              <Text size="lg" c="dimmed">
                使用 React 编写视频，结合 Mantine 的精美组件库，让您的演示视频更具专业感。
              </Text>

              <Group gap="sm" mt="md">
                <Button variant="filled" size="md">
                  立即开始
                </Button>
                <Button variant="outline" size="md">
                  了解更多
                </Button>
              </Group>
            </Stack>
          </Card>

          <Text
            size="md"
            c="dimmed"
            style={{
              opacity: interpolate(frame, [80, 100], [0, 1], {
                extrapolateLeft: 'clamp',
              }),
            }}
          >
            由 Antigravity 自动生成
          </Text>
        </Stack>
      </AbsoluteFill>
    </MantineProvider>
  );
};
