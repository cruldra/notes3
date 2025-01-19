// src/components/UvicornCmdBuilder.tsx
import React from 'react';
import {
    TextInput,
    NumberInput,
    Switch,
    Button,
    Select,
    Card,
    Stack,
    Group,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { MdOutlineContentCopy } from "react-icons/md";

interface UvicornFormValues {
    app: string;
    host?: string;
    port?: number;
    reload?: boolean;
    reloadDir?: string;
    workers?: number;
    loop?: 'auto' | 'asyncio' | 'uvloop';
    http?: 'auto' | 'h11' | 'httptools';
    ws?: 'auto' | 'none' | 'websockets' | 'wsproto';
    wsMaxSize?: number;
    wsMaxQueue?: number;
    wsPingInterval?: number;
    wsPingTimeout?: number;
    wsPerMessageDeflate?: boolean;
    lifespan?: 'auto' | 'on' | 'off';
    interface?: 'auto' | 'asgi3' | 'asgi2' | 'wsgi';
    envFile?: string;
    logConfig?: string;
    logLevel?: 'critical' | 'error' | 'warning' | 'info' | 'debug' | 'trace';
    accessLog?: boolean;
    useColors?: boolean;
    proxyHeaders?: boolean;
    limitConcurrency?: number;
    timeout?: number;
    sslKeyfile?: string;
    sslCertfile?: string;
    sslKeyfilePassword?: string;
}

export default function UvicornCmdBuilder() {
    const form = useForm<UvicornFormValues>({
        initialValues: {
            app: 'main:app',
            host: '127.0.0.1',
            port: 8000,
            reload: false,
            workers: 1,
            loop: 'auto',
            http: 'auto',
            ws: 'auto',
            wsMaxSize: 16777216,
            wsMaxQueue: 32,
            wsPingInterval: 20.0,
            wsPingTimeout: 20.0,
            wsPerMessageDeflate: true,
            lifespan: 'auto',
            interface: 'auto',
            logLevel: 'info',
            accessLog: true,
            useColors: true,
            proxyHeaders: true,
        },
    });

    const buildCommand = (values: UvicornFormValues): string => {
        const parts: string[] = ['uvicorn'];

        // 必需的app参数
        parts.push(values.app);

        // 可选参数
        if (values.host !== '127.0.0.1') parts.push(`--host ${values.host}`);
        if (values.port !== 8000) parts.push(`--port ${values.port}`);
        if (values.reload) parts.push('--reload');
        if (values.reloadDir) parts.push(`--reload-dir "${values.reloadDir}"`);
        if (values.workers !== 1) parts.push(`--workers ${values.workers}`);
        if (values.loop !== 'auto') parts.push(`--loop ${values.loop}`);
        if (values.http !== 'auto') parts.push(`--http ${values.http}`);
        if (values.ws !== 'auto') parts.push(`--ws ${values.ws}`);
        if (values.wsMaxSize !== 16777216) parts.push(`--ws-max-size ${values.wsMaxSize}`);
        if (values.wsMaxQueue !== 32) parts.push(`--ws-max-queue ${values.wsMaxQueue}`);
        if (values.wsPingInterval !== 20.0) parts.push(`--ws-ping-interval ${values.wsPingInterval}`);
        if (values.wsPingTimeout !== 20.0) parts.push(`--ws-ping-timeout ${values.wsPingTimeout}`);
        if (!values.wsPerMessageDeflate) parts.push('--ws-per-message-deflate false');
        if (values.lifespan !== 'auto') parts.push(`--lifespan ${values.lifespan}`);
        if (values.interface !== 'auto') parts.push(`--interface ${values.interface}`);
        if (values.envFile) parts.push(`--env-file "${values.envFile}"`);
        if (values.logConfig) parts.push(`--log-config "${values.logConfig}"`);
        if (values.logLevel !== 'info') parts.push(`--log-level ${values.logLevel}`);
        if (!values.accessLog) parts.push('--no-access-log');
        if (!values.useColors) parts.push('--no-use-colors');
        if (!values.proxyHeaders) parts.push('--no-proxy-headers');
        if (values.limitConcurrency) parts.push(`--limit-concurrency ${values.limitConcurrency}`);
        if (values.timeout) parts.push(`--timeout-keep-alive ${values.timeout}`);
        if (values.sslKeyfile) parts.push(`--ssl-keyfile "${values.sslKeyfile}"`);
        if (values.sslCertfile) parts.push(`--ssl-certfile "${values.sslCertfile}"`);
        if (values.sslKeyfilePassword) parts.push(`--ssl-keyfile-password "${values.sslKeyfilePassword}"`);

        return parts.join(' ');
    };

    const onSubmit = (values: UvicornFormValues) => {
        const command = buildCommand(values);
        navigator.clipboard.writeText(command)
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
        <Card shadow="sm" padding="lg">
            <form onSubmit={form.onSubmit(onSubmit)}>
                <Stack>
                    <TextInput
                        required
                        label="应用模块路径"
                        placeholder="例如: main:app"
                        {...form.getInputProps('app')}
                    />

                    <TextInput
                        label="主机地址"
                        placeholder="127.0.0.1"
                        {...form.getInputProps('host')}
                    />

                    <NumberInput
                        label="端口"
                        min={0}
                        max={65535}
                        {...form.getInputProps('port')}
                    />

                    <Switch
                        label="启用自动重载"
                        {...form.getInputProps('reload', { type: 'checkbox' })}
                    />

                    <TextInput
                        label="重载目录"
                        placeholder="指定要监视的目录路径"
                        {...form.getInputProps('reloadDir')}
                    />

                    <NumberInput
                        label="工作进程数"
                        min={1}
                        {...form.getInputProps('workers')}
                    />

                    <Select
                        label="事件循环实现"
                        data={[
                            { value: 'auto', label: '自动' },
                            { value: 'asyncio', label: 'asyncio' },
                            { value: 'uvloop', label: 'uvloop' }
                        ]}
                        {...form.getInputProps('loop')}
                    />

                    <Select
                        label="HTTP协议实现"
                        data={[
                            { value: 'auto', label: '自动' },
                            { value: 'h11', label: 'h11' },
                            { value: 'httptools', label: 'httptools' }
                        ]}
                        {...form.getInputProps('http')}
                    />

                    <Select
                        label="WebSocket实现"
                        data={[
                            { value: 'auto', label: '自动' },
                            { value: 'none', label: '禁用' },
                            { value: 'websockets', label: 'websockets' },
                            { value: 'wsproto', label: 'wsproto' }
                        ]}
                        {...form.getInputProps('ws')}
                    />

                    <Select
                        label="日志级别"
                        data={[
                            { value: 'critical', label: 'Critical' },
                            { value: 'error', label: 'Error' },
                            { value: 'warning', label: 'Warning' },
                            { value: 'info', label: 'Info' },
                            { value: 'debug', label: 'Debug' },
                            { value: 'trace', label: 'Trace' }
                        ]}
                        {...form.getInputProps('logLevel')}
                    />

                    <Group>
                        <Switch
                            label="访问日志"
                            {...form.getInputProps('accessLog', { type: 'checkbox' })}
                        />
                        <Switch
                            label="彩色输出"
                            {...form.getInputProps('useColors', { type: 'checkbox' })}
                        />
                        <Switch
                            label="代理头"
                            {...form.getInputProps('proxyHeaders', { type: 'checkbox' })}
                        />
                    </Group>

                    <TextInput
                        label="SSL密钥文件"
                        {...form.getInputProps('sslKeyfile')}
                    />

                    <TextInput
                        label="SSL证书文件"
                        {...form.getInputProps('sslCertfile')}
                    />

                    <TextInput
                        label="SSL密钥密码"
                        type="password"
                        {...form.getInputProps('sslKeyfilePassword')}
                    />

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
