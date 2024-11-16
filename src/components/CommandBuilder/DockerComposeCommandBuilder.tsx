import React, { useState } from 'react';
import {
    TextInput,
    Switch,
    Button,
    Card,
    Stack,
    Group,
    Select,
    Badge,
    ActionIcon,
    Text,
    NumberInput,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { MdOutlineContentCopy } from "react-icons/md";
import { FaTrash } from "react-icons/fa";

interface DockerComposeFormValues {
    command: string;
    file?: string;
    projectName?: string;
    profile?: string;
    services: string[];
    // up options
    detach?: boolean;
    build?: boolean;
    noDeps?: boolean;
    forceRecreate?: boolean;
    removeOrphans?: boolean;
    timeout?: number;
    // down options
    volumes?: boolean;
    removeImages?: 'all' | 'local' | '';
    // logs options
    follow?: boolean;
    tail?: number;
    timestamps?: boolean;
    // run/exec options
    user?: string;
    workdir?: string;
    entrypoint?: string;
    // scale options
    scale?: Record<string, number>;
}

const COMMANDS = [
    { value: 'build', label: 'build - 构建或重新构建服务' },
    { value: 'config', label: 'config - 验证并查看 Compose 文件' },
    { value: 'create', label: 'create - 创建服务' },
    { value: 'down', label: 'down - 停止并移除资源' },
    { value: 'events', label: 'events - 接收容器实时事件' },
    { value: 'exec', label: 'exec - 在运行中的容器中执行命令' },
    { value: 'images', label: 'images - 列出镜像' },
    { value: 'kill', label: 'kill - 杀死容器' },
    { value: 'logs', label: 'logs - 查看容器输出' },
    { value: 'pause', label: 'pause - 暂停服务' },
    { value: 'port', label: 'port - 打印端口映射' },
    { value: 'ps', label: 'ps - 列出容器' },
    { value: 'pull', label: 'pull - 拉取服务镜像' },
    { value: 'push', label: 'push - 推送服务镜像' },
    { value: 'restart', label: 'restart - 重启服务' },
    { value: 'rm', label: 'rm - 移除停止的容器' },
    { value: 'run', label: 'run - 运行一次性命令' },
    { value: 'scale', label: 'scale - 设置服务的容器数量' },
    { value: 'start', label: 'start - 启动服务' },
    { value: 'stop', label: 'stop - 停止服务' },
    { value: 'top', label: 'top - 显示运行中的进程' },
    { value: 'unpause', label: 'unpause - 取消暂停服务' },
    { value: 'up', label: 'up - 创建并启动容器' },
    { value: 'version', label: 'version - 显示版本信息' },
];

export default function DockerComposeCommandBuilder() {
    const [command, setCommand] = useState('');
    const [scale, setScale] = useState<Record<string, number>>({});

    const form = useForm<DockerComposeFormValues>({
        initialValues: {
            command: 'up',
            services: [],
            detach: true,
            build: false,
            noDeps: false,
            forceRecreate: false,
            removeOrphans: false,
            timeout: 10,
            volumes: false,
            removeImages: '',
            follow: false,
            tail: 0,
            timestamps: false,
            user: '',
            workdir: '',
            entrypoint: '',
            scale: {},
        }
    });

    const buildCommand = (values: DockerComposeFormValues): string => {
        const parts: string[] = ['docker compose'];

        if (values.file) {
            parts.push(`-f ${values.file}`);
        }

        if (values.projectName) {
            parts.push(`-p ${values.projectName}`);
        }

        if (values.profile) {
            parts.push(`--profile ${values.profile}`);
        }

        parts.push(values.command);

        switch (values.command) {
            case 'up':
                if (values.detach) parts.push('-d');
                if (values.build) parts.push('--build');
                if (values.noDeps) parts.push('--no-deps');
                if (values.forceRecreate) parts.push('--force-recreate');
                if (values.removeOrphans) parts.push('--remove-orphans');
                if (values.timeout !== 10) parts.push(`--timeout ${values.timeout}`);
                break;
            case 'down':
                if (values.volumes) parts.push('-v');
                if (values.removeImages) parts.push(`--rmi ${values.removeImages}`);
                if (values.removeOrphans) parts.push('--remove-orphans');
                break;
            case 'logs':
                if (values.follow) parts.push('-f');
                if (values.timestamps) parts.push('-t');
                if (values.tail !== undefined && values.tail >= 0) parts.push(`--tail=${values.tail}`);
                break;
            case 'run':
            case 'exec':
                if (values.user) parts.push(`--user ${values.user}`);
                if (values.workdir) parts.push(`--workdir ${values.workdir}`);
                if (values.entrypoint) parts.push(`--entrypoint ${values.entrypoint}`);
                break;
            case 'scale':
                const scaleArgs = Object.entries(values.scale || {})
                    .map(([service, count]) => `${service}=${count}`);
                if (scaleArgs.length > 0) parts.push(scaleArgs.join(' '));
                break;
        }

        if (['up', 'run', 'exec', 'logs', 'rm', 'start', 'stop', 'restart'].includes(values.command) && values.services.length > 0) {
            parts.push(values.services.join(' '));
        }

        return parts.join(' ');
    };

    // ... removeService 和 addService 函数保持不变 ...

    const removeService = (service: string) => {
        form.setFieldValue('services', form.values.services.filter(s => s !== service));
    };

    const addService = (service: string) => {
        if (service && !form.values.services.includes(service)) {
            form.setFieldValue('services', [...form.values.services, service]);
            return true;
        }
        return false;
    };
    const renderCommandOptions = () => {
        switch (form.values.command) {
            case 'up':
                return (
                    <>
                        <Switch label="后台运行 (-d)" {...form.getInputProps('detach', { type: 'checkbox' })} />
                        <Switch label="构建镜像 (--build)" {...form.getInputProps('build', { type: 'checkbox' })} />
                        <Switch label="不启动依赖 (--no-deps)" {...form.getInputProps('noDeps', { type: 'checkbox' })} />
                        <Switch label="强制重新创建 (--force-recreate)" {...form.getInputProps('forceRecreate', { type: 'checkbox' })} />
                        <Switch label="移除孤立容器 (--remove-orphans)" {...form.getInputProps('removeOrphans', { type: 'checkbox' })} />
                        <NumberInput label="超时时间（秒）" {...form.getInputProps('timeout')} min={0} />
                    </>
                );
            case 'down':
                return (
                    <>
                        <Switch label="移除卷 (-v)" {...form.getInputProps('volumes', { type: 'checkbox' })} />
                        <Select
                            label="移除镜像"
                            placeholder="选择要移除的镜像类型"
                            data={[
                                { value: '', label: '不移除' },
                                { value: 'local', label: '仅本地镜像' },
                                { value: 'all', label: '所有镜像' },
                            ]}
                            {...form.getInputProps('removeImages')}
                        />
                    </>
                );
            case 'logs':
                return (
                    <>
                        <Switch label="跟踪日志 (-f)" {...form.getInputProps('follow', { type: 'checkbox' })} />
                        <Switch label="显示时间戳 (-t)" {...form.getInputProps('timestamps', { type: 'checkbox' })} />
                        <NumberInput label="显示最后几行" {...form.getInputProps('tail')} min={0} />
                    </>
                );
            case 'run':
            case 'exec':
                return (
                    <>
                        <TextInput label="用户" placeholder="username:group" {...form.getInputProps('user')} />
                        <TextInput label="工作目录" placeholder="/app" {...form.getInputProps('workdir')} />
                        <TextInput label="入口点" placeholder="custom-entrypoint" {...form.getInputProps('entrypoint')} />
                    </>
                );
            case 'scale':
                return (
                    <Stack gap="xs">
                        {form.values.services.map(service => (
                            <NumberInput
                                key={service}
                                label={`${service} 容器数量`}
                                min={0}
                                value={scale[service] || 0}
                                onChange={(value) => {
                                    const newScale = { ...scale, [service]: value || 0 };
                                    setScale(newScale);
                                    form.setFieldValue('scale', newScale);
                                }}
                            />
                        ))}
                    </Stack>
                );
            default:
                return null;
        }
    };
    const onSubmit = (values: DockerComposeFormValues) => {
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
                    <Select
                        label="Docker Compose 命令"
                        data={COMMANDS}
                        searchable
                        {...form.getInputProps('command')}
                    />

                    <TextInput
                        label="Compose 文件路径"
                        placeholder="docker-compose.yml"
                        {...form.getInputProps('file')}
                    />

                    <TextInput
                        label="项目名称"
                        placeholder="my-project"
                        {...form.getInputProps('projectName')}
                    />

                    <TextInput
                        label="Profile"
                        placeholder="开发环境、生产环境等"
                        {...form.getInputProps('profile')}
                    />

                    {['up', 'run', 'exec', 'logs', 'rm', 'start', 'stop', 'restart', 'scale'].includes(form.values.command) && (
                        <>
                            <Select
                                label="添加服务"
                                placeholder="选择要操作的服务"
                                data={[
                                    { value: 'web', label: 'Web 服务' },
                                    { value: 'db', label: '数据库' },
                                    { value: 'redis', label: 'Redis' },
                                    { value: 'nginx', label: 'Nginx' },
                                ]}
                                onChange={(value) => value && addService(value)}
                                clearable
                            />

                            {form.values.services.length > 0 && (
                                <Stack gap="xs">
                                    <Text size="sm" fw={500}>已选择的服务：</Text>
                                    <Group gap="xs">
                                        {form.values.services.map((service) => (
                                            <Badge
                                                key={service}
                                                size="lg"
                                                rightSection={
                                                    <ActionIcon
                                                        size="xs"
                                                        color="red"
                                                        variant="transparent"
                                                        onClick={() => removeService(service)}
                                                    >
                                                        <FaTrash size={10} />
                                                    </ActionIcon>
                                                }
                                            >
                                                {service}
                                            </Badge>
                                        ))}
                                    </Group>
                                </Stack>
                            )}
                        </>
                    )}

                    {renderCommandOptions()}

                    {command && (
                        <Card withBorder>
                            <Text size="sm" fw={500}>生成的命令：</Text>
                            <Text style={{ wordBreak: 'break-all' }} mt="xs">
                                {command}
                            </Text>
                        </Card>
                    )}

                    <Button
                        type="submit"
                        leftSection={<MdOutlineContentCopy size={14} />}
                    >
                        生成命令并复制
                    </Button>
                </Stack>
            </form>
        </Card>
    );
}
