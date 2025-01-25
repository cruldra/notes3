import React from 'react';
import { Card, TextInput, Switch, Stack, Button, Text } from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import {FaCopy} from "react-icons/fa";

export default function PoetryEnvPathConfig() {
    const form = useForm({
        initialValues: {
            customPath: '',
            inProject: false
        },
        validate: {
            customPath: (value, values) => {
                if (!values.inProject && !value) {
                    return '请输入自定义路径或选择在项目目录下创建';
                }
                return null;
            }
        }
    });

    const buildCommand = (values) => {
        const commands = [];

        if (values.inProject) {
            commands.push('poetry config virtualenvs.in-project true');
        } else if (values.customPath) {
            commands.push(`poetry config virtualenvs.path "${values.customPath}"`);
        }

        return commands.join('\n');
    };

    const handleSubmit = (values) => {
        const command = buildCommand(values);
        if (!command) {
            notifications.show({
                message: '请至少选择一个配置选项',
                color: 'red',
                position: 'top-center'
            });
            return;
        }

        navigator.clipboard.writeText(command)
            .then(() => {
                notifications.show({
                    message: '命令已复制到剪贴板',
                    color: 'green',
                    position: 'top-center'
                });
            })
            .catch(() => {
                notifications.show({
                    message: '复制失败',
                    color: 'red',
                    position: 'top-center'
                });
            });
    };

    const command = buildCommand(form.values);

    return (
        <Card shadow="sm" p="lg" radius="md" withBorder>
            <form onSubmit={form.onSubmit(handleSubmit)}>
                <Stack gap="md">
                    <Switch
                        label="在项目目录下创建虚拟环境"
                        description="虚拟环境将保存在项目的 .venv 目录中"
                        {...form.getInputProps('inProject', { type: 'checkbox' })}
                    />

                    <TextInput
                        label="自定义虚拟环境路径"
                        placeholder="例如: /custom/path/to/venvs"
                        disabled={form.values.inProject}
                        {...form.getInputProps('customPath')}
                    />

                    {command && (
                        <Card withBorder p="sm" radius="md">
                            <Text size="sm" fw={500} mb="xs">
                                生成的命令:
                            </Text>
                            <Text
                                style={{
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-all'
                                }}
                                c="dimmed"
                            >
                                {command}
                            </Text>
                        </Card>
                    )}

                    <Button
                        type="submit"
                        leftSection={<FaCopy  size={16} />}
                        disabled={!form.values.inProject && !form.values.customPath}
                    >
                        生成命令并复制
                    </Button>

                    <Card withBorder p="sm" radius="md">
                        <Text size="sm" fw={500} mb="xs">
                            其他常用命令:
                        </Text>
                        <Stack gap="xs">
                            <Text size="sm" c="dimmed">
                                查看当前配置:
                                <br />
                                poetry config --list
                            </Text>
                            <Text size="sm" c="dimmed">
                                恢复默认设置:
                                <br />
                                poetry config virtualenvs.in-project --unset
                                <br />
                                poetry config virtualenvs.path --unset
                            </Text>
                        </Stack>
                    </Card>
                </Stack>
            </form>
        </Card>
    );
}
