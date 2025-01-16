import React, {useState} from 'react';
import {
    ActionIcon,
    Badge,
    Button,
    Card,
    Group,
    Stack,
    Switch,
    Text,
    TextInput,
} from '@mantine/core';
import {useForm} from '@mantine/form';
import {notifications} from '@mantine/notifications';
import {MdOutlineContentCopy} from "react-icons/md";
import {FaPlus, FaTrash} from "react-icons/fa";

interface DockerBuildFormValues {
    tags: string[];           // 镜像标签
    dockerfile?: string;      // Dockerfile路径
    buildArgs: string[];      // 构建参数
    platform?: string;        // 目标平台
    noCache?: boolean;        // 不使用缓存
    pull?: boolean;          // 总是拉取基础镜像
    quiet?: boolean;         // 安静模式
}

export default function DockerBuildCommandBuilder() {
    const [newTag, setNewTag] = useState('');
    const [newBuildArg, setNewBuildArg] = useState('');
    const [command, setCommand] = useState('');

    const form = useForm<DockerBuildFormValues>({
        initialValues: {
            tags: [],
            buildArgs: [],
            noCache: false,
            pull: false,
            quiet: false,
        },
        validate: {
            tags: (value) => (value.length === 0 ? '请至少添加一个镜像标签' : null),
        },
    });

    // 添加标签
    const addTag = () => {
        if (newTag && !form.values.tags.includes(newTag)) {
            form.setFieldValue('tags', [...form.values.tags, newTag]);
            setNewTag('');
        }
    };

    // 添加构建参数
    const addBuildArg = () => {
        if (newBuildArg && !form.values.buildArgs.includes(newBuildArg)) {
            form.setFieldValue('buildArgs', [...form.values.buildArgs, newBuildArg]);
            setNewBuildArg('');
        }
    };

    // 移除标签
    const removeTag = (tag: string) => {
        form.setFieldValue('tags', form.values.tags.filter(t => t !== tag));
    };

    // 移除构建参数
    const removeBuildArg = (arg: string) => {
        form.setFieldValue('buildArgs', form.values.buildArgs.filter(a => a !== arg));
    };

    // 生成命令
    const buildCommand = (values: DockerBuildFormValues): string => {
        const parts: string[] = ['docker build'];

        values.tags.forEach(tag => {
            parts.push(`-t ${tag}`);
        });

        if (values.dockerfile) {
            parts.push(`-f ${values.dockerfile}`);
        }

        values.buildArgs.forEach(arg => {
            parts.push(`--build-arg ${arg}`);
        });

        if (values.platform) {
            parts.push(`--platform ${values.platform}`);
        }

        if (values.noCache) {
            parts.push('--no-cache');
        }

        if (values.pull) {
            parts.push('--pull');
        }

        if (values.quiet) {
            parts.push('--quiet');
        }

        parts.push('.');

        return parts.join(' ');
    };

    const onSubmit = (values: DockerBuildFormValues) => {
        const generatedCommand = buildCommand(values);
        setCommand(generatedCommand);
        navigator.clipboard.writeText(generatedCommand)
            .then(() => notifications.show({
                message: '命令已复制到剪贴板',
                color: 'green',
                position: "top-center"
            }))
            .catch(() => notifications.show({
                message: '复制失败',
                color: 'red',
                position: "top-center"
            }));
    };

    return (
        <Card shadow="sm" p="lg">
            <form onSubmit={form.onSubmit(onSubmit)}>
                <Stack gap="md">
                    {/* 镜像标签输入 */}
                    <Group align="flex-start">
                        <TextInput
                            style={{flex: 1}}
                            label="添加镜像标签"
                            placeholder="例如: myapp:latest"
                            value={newTag}
                            onChange={(e) => setNewTag(e.currentTarget.value)}
                            onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                            rightSection={
                                <ActionIcon
                                    onClick={addTag}
                                    disabled={!newTag}
                                    variant="filled"
                                    color="blue"
                                >
                                    <FaPlus size={14}/>
                                </ActionIcon>
                            }
                        />
                    </Group>

                    {/* 显示已添加的标签 */}
                    {form.values.tags.length > 0 && (
                        <Stack gap="xs">
                            <Text size="sm" fw={500}>已添加的标签：</Text>
                            <Group gap="xs">
                                {form.values.tags.map((tag) => (
                                    <Badge
                                        key={tag}
                                        size="lg"
                                        rightSection={
                                            <ActionIcon
                                                size="xs"
                                                color="red"
                                                variant="transparent"
                                                onClick={() => removeTag(tag)}
                                            >
                                                <FaTrash size={10}/>
                                            </ActionIcon>
                                        }
                                    >
                                        {tag}
                                    </Badge>
                                ))}
                            </Group>
                        </Stack>
                    )}

                    {/* Dockerfile路径 */}
                    <TextInput
                        label="Dockerfile路径"
                        placeholder="例如: Dockerfile.prod"
                        {...form.getInputProps('dockerfile')}
                    />

                    {/* 构建参数输入 */}
                    <Group align="flex-start">
                        <TextInput
                            style={{flex: 1}}
                            label="添加构建参数"
                            placeholder="例如: VERSION=1.0"
                            value={newBuildArg}
                            onChange={(e) => setNewBuildArg(e.currentTarget.value)}
                            onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addBuildArg())}
                            rightSection={
                                <ActionIcon
                                    onClick={addBuildArg}
                                    disabled={!newBuildArg}
                                    variant="filled"
                                    color="blue"
                                >
                                    <FaPlus size={14}/>
                                </ActionIcon>
                            }
                        />
                    </Group>

                    {/* 显示已添加的构建参数 */}
                    {form.values.buildArgs.length > 0 && (
                        <Stack gap="xs">
                            <Text size="sm" fw={500}>已添加的构建参数：</Text>
                            <Group gap="xs">
                                {form.values.buildArgs.map((arg) => (
                                    <Badge
                                        key={arg}
                                        size="lg"
                                        rightSection={
                                            <ActionIcon
                                                size="xs"
                                                color="red"
                                                variant="transparent"
                                                onClick={() => removeBuildArg(arg)}
                                            >
                                                <FaTrash size={10}/>
                                            </ActionIcon>
                                        }
                                    >
                                        {arg}
                                    </Badge>
                                ))}
                            </Group>
                        </Stack>
                    )}

                    {/* 目标平台 */}
                    <TextInput
                        label="目标平台"
                        placeholder="例如: linux/amd64,linux/arm64"
                        {...form.getInputProps('platform')}
                    />

                    {/* 开关选项 */}
                    <Switch
                        label="不使用缓存"
                        description="构建时不使用缓存"
                        {...form.getInputProps('noCache', {type: 'checkbox'})}
                    />

                    <Switch
                        label="总是拉取基础镜像"
                        description="构建前拉取所有引用的镜像"
                        {...form.getInputProps('pull', {type: 'checkbox'})}
                    />

                    <Switch
                        label="安静模式"
                        description="只显示镜像ID"
                        {...form.getInputProps('quiet', {type: 'checkbox'})}
                    />

                    {/* 显示生成的命令 */}
                    {command && (
                        <Card withBorder>
                            <Text size="sm" fw={500}>生成的命令：</Text>
                            <Text style={{wordBreak: 'break-all'}} mt="xs">
                                {command}
                            </Text>
                        </Card>
                    )}

                    {/* 提交按钮 */}
                    <Button
                        type="submit"
                        leftSection={<MdOutlineContentCopy size={14}/>}
                        disabled={form.values.tags.length === 0}
                    >
                        生成命令并复制
                    </Button>
                </Stack>
            </form>
        </Card>
    );
}
