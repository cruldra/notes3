// src/components/LuxCommandBuilder.tsx
import React from 'react';
import {
    TextInput,
    NumberInput,
    Switch,
    Button,
    Select,
    Card,
    Tooltip,
    Textarea,
    Stack,
    Group,
    PasswordInput,
    Box
} from '@mantine/core';
import {useForm} from '@mantine/form';
import {notifications} from '@mantine/notifications';
import {MdOutlineContentCopy} from "react-icons/md";
import {FaQuestionCircle} from "react-icons/fa";

interface LuxFormValues {
    url: string;
    aria2?: boolean;
    aria2Addr?: string;
    aria2Method?: string;
    aria2Token?: string;
    audioOnly?: boolean;
    caption?: boolean;
    chunkSize?: number;
    cookie?: string;
    debug?: boolean;
    end?: number;
    episodeTitleOnly?: boolean;
    file?: string;
    fileNameLength?: number;
    info?: boolean;
    items?: string;
    json?: boolean;
    multiThread?: boolean;
    outputName?: string;
    outputPath?: string;
    playlist?: boolean;
    refer?: string;
    retry?: number;
    silent?: boolean;
    start?: number;
    streamFormat?: string;
    threadNum?: number;
    userAgent?: string;
    youkuCcode?: string;
    youkuCkey?: string;
    youkuPassword?: string;
}

export default function LuxCommandBuilder() {
    const form = useForm<LuxFormValues>({
        initialValues: {
            url: '',
            aria2: false,
            aria2Addr: "localhost:6800",
            aria2Method: "http",
            audioOnly: false,
            caption: false,
            chunkSize: 1,
            debug: false,
            end: 0,
            episodeTitleOnly: false,
            fileNameLength: 255,
            info: false,
            json: false,
            multiThread: false,
            playlist: false,
            retry: 10,
            silent: false,
            start: 1,
            threadNum: 10,
            youkuCcode: "0502"
        },
    });

    const buildCommand = (values: LuxFormValues): string => {
        const parts: string[] = ['lux'];
        const defaultValues = {
            aria2: false,
            aria2Addr: "localhost:6800",
            aria2Method: "http",
            audioOnly: false,
            caption: false,
            chunkSize: 1,
            debug: false,
            end: 0,
            episodeTitleOnly: false,
            fileNameLength: 255,
            info: false,
            json: false,
            multiThread: false,
            playlist: false,
            retry: 10,
            silent: false,
            start: 1,
            threadNum: 10,
            youkuCcode: "0502"
        };

        if (values.aria2 && values.aria2 !== defaultValues.aria2) parts.push('--aria2');
        if (values.aria2) {
            if (values.aria2Addr && values.aria2Addr !== defaultValues.aria2Addr) {
                parts.push(`--aria2-addr "${values.aria2Addr}"`);
            }
            if (values.aria2Method && values.aria2Method !== defaultValues.aria2Method) {
                parts.push(`--aria2-method "${values.aria2Method}"`);
            }
            if (values.aria2Token) {
                parts.push(`--aria2-token "${values.aria2Token}"`);
            }
        }
        // ... [其余的buildCommand逻辑保持不变]
        parts.push(values.url);
        return parts.join(' ');
    };

    const onSubmit = (values: LuxFormValues) => {
        const command = buildCommand(values);
        navigator.clipboard.writeText(command)
            .then(() => notifications.show({
                message: '命令已复制到剪贴板',
                color: 'green',
                position:"top-center"
            }))
            .catch(() => notifications.show({
                message: '复制失败',
                color: 'red',
                position:"top-center"
            }));
    };

    return (
        <Card shadow="sm" padding="lg">
            <form onSubmit={form.onSubmit(onSubmit)}>
                <Stack>
                    <TextInput
                        required
                        label="视频URL"
                        placeholder="请输入要下载的视频URL"
                        {...form.getInputProps('url')}
                    />

                    <Switch
                        label="Aria2下载"
                        {...form.getInputProps('aria2', {type: 'checkbox'})}
                    />

                    <TextInput
                        label="Aria2地址"
                        placeholder="localhost:6800"
                        {...form.getInputProps('aria2Addr')}
                    />

                    <Select
                        label="Aria2方法"
                        data={[
                            {value: 'http', label: 'HTTP'},
                            {value: 'https', label: 'HTTPS'}
                        ]}
                        {...form.getInputProps('aria2Method')}
                    />

                    <TextInput
                        label="Aria2令牌"
                        placeholder="Aria2 RPC令牌"
                        {...form.getInputProps('aria2Token')}
                    />

                    <Textarea
                        label="Cookie"
                        placeholder="Cookie"
                        {...form.getInputProps('cookie')}
                    />

                    <Switch
                        label="调试模式"
                        {...form.getInputProps('debug', {type: 'checkbox'})}
                    />

                    <NumberInput
                        label="结束项"
                        min={0}
                        {...form.getInputProps('end')}
                    />

                    <Switch
                        label="纯集标题"
                        {...form.getInputProps('episodeTitleOnly', {type: 'checkbox'})}
                    />

                    <TextInput
                        label="URL文件"
                        placeholder="URL文件路径"
                        {...form.getInputProps('file')}
                    />

                    <NumberInput
                        label="文件名长度限制"
                        min={0}
                        {...form.getInputProps('fileNameLength')}
                    />

                    <Switch
                        label="仅显示信息"
                        {...form.getInputProps('info', {type: 'checkbox'})}
                    />

                    <TextInput
                        label="指定项目"
                        placeholder="如: 1,5,6,8-10"
                        {...form.getInputProps('items')}
                    />

                    <Switch
                        label="输出JSON"
                        {...form.getInputProps('json', {type: 'checkbox'})}
                    />

                    <TextInput
                        label="Referer"
                        {...form.getInputProps('refer')}
                    />

                    <Switch
                        label="静默模式"
                        {...form.getInputProps('silent', {type: 'checkbox'})}
                    />

                    <NumberInput
                        label="起始项"
                        min={1}
                        {...form.getInputProps('start')}
                    />

                    <TextInput
                        label="流格式"
                        {...form.getInputProps('streamFormat')}
                    />

                    <TextInput
                        label="User-Agent"
                        {...form.getInputProps('userAgent')}
                    />

                    <TextInput
                        label="优酷ccode"
                        {...form.getInputProps('youkuCcode')}
                    />

                    <TextInput
                        label="优酷ckey"
                        {...form.getInputProps('youkuCkey')}
                    />

                    <PasswordInput
                        label="优酷密码"
                        {...form.getInputProps('youkuPassword')}
                    />

                    <Button
                        type="submit"
                        leftSection={<MdOutlineContentCopy size={14}/>}
                    >
                        生成命令并复制
                    </Button>
                </Stack>
            </form>
        </Card>
    );
}
