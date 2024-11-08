// src/components/LuxCommandBuilder.tsx
import React from 'react';
import {Form, Input, InputNumber, Switch, Button, message, Select, Card, Tooltip} from 'antd';
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
    const [form] = Form.useForm();

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
        if (values.audioOnly && values.audioOnly !== defaultValues.audioOnly) parts.push('--audio-only');
        if (values.caption && values.caption !== defaultValues.caption) parts.push('--caption');
        if (values.chunkSize && values.chunkSize !== defaultValues.chunkSize) parts.push(`--chunk-size ${values.chunkSize}`);
        if (values.cookie) parts.push(`--cookie "${values.cookie}"`);
        if (values.debug && values.debug !== defaultValues.debug) parts.push('--debug');
        if (values.end && values.end !== defaultValues.end) parts.push(`--end ${values.end}`);
        if (values.episodeTitleOnly && values.episodeTitleOnly !== defaultValues.episodeTitleOnly) parts.push('--episode-title-only');
        if (values.file) parts.push(`--file "${values.file}"`);
        if (values.fileNameLength && values.fileNameLength !== defaultValues.fileNameLength) parts.push(`--file-name-length ${values.fileNameLength}`);
        if (values.info && values.info !== defaultValues.info) parts.push('--info');
        if (values.items) parts.push(`--items "${values.items}"`);
        if (values.json && values.json !== defaultValues.json) parts.push('--json');
        if (values.multiThread && values.multiThread !== defaultValues.multiThread) parts.push('--multi-thread');
        if (values.outputName) parts.push(`-O "${values.outputName}"`);
        if (values.outputPath) parts.push(`-o "${values.outputPath}"`);
        if (values.playlist && values.playlist !== defaultValues.playlist) parts.push('--playlist');
        if (values.refer) parts.push(`--refer "${values.refer}"`);
        if (values.retry && values.retry !== defaultValues.retry) parts.push(`--retry ${values.retry}`);
        if (values.silent && values.silent !== defaultValues.silent) parts.push('--silent');
        if (values.start && values.start !== defaultValues.start) parts.push(`--start ${values.start}`);
        if (values.streamFormat) parts.push(`--stream-format "${values.streamFormat}"`);
        if (values.threadNum && values.threadNum !== defaultValues.threadNum) parts.push(`--thread ${values.threadNum}`);
        if (values.userAgent) parts.push(`--user-agent "${values.userAgent}"`);
        if (values.youkuCcode && values.youkuCcode !== defaultValues.youkuCcode) parts.push(`--youku-ccode "${values.youkuCcode}"`);
        if (values.youkuCkey) parts.push(`--youku-ckey "${values.youkuCkey}"`);
        if (values.youkuPassword) parts.push(`--youku-password "${values.youkuPassword}"`);
        parts.push(values.url);
        return parts.join(' ');
    };
    const onFinish = (values: LuxFormValues) => {
        const command = buildCommand(values);
        navigator.clipboard.writeText(command)
            .then(() => message.success('命令已复制到剪贴板'))
            .catch(() => message.error('复制失败'));
    };

    return (
        <Card
            title="Lux命令生成器"
            extra={
                <Tooltip title="Lux视频下载工具命令行生成器">
                    <FaQuestionCircle/>
                </Tooltip>
            }
        >

            <Form
                form={form}
                layout="vertical"
                onFinish={onFinish}
                initialValues={{
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
                }}
            >
                <Form.Item
                    label="视频URL"
                    name="url"
                    rules={[{required: true, message: '请输入视频URL'}]}
                >
                    <Input placeholder="请输入要下载的视频URL"/>
                </Form.Item>

                <Form.Item label="Aria2下载" name="aria2" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="Aria2地址" name="aria2Addr">
                    <Input placeholder="localhost:6800"/>
                </Form.Item>

                <Form.Item label="Aria2方法" name="aria2Method">
                    <Select>
                        <Select.Option value="http">HTTP</Select.Option>
                        <Select.Option value="https">HTTPS</Select.Option>
                    </Select>
                </Form.Item>

                <Form.Item label="Aria2令牌" name="aria2Token">
                    <Input placeholder="Aria2 RPC令牌"/>
                </Form.Item>

                {/* ... 以下是原有的表单项 ... */}

                <Form.Item label="Cookie" name="cookie">
                    <Input.TextArea placeholder="Cookie"/>
                </Form.Item>

                <Form.Item label="调试模式" name="debug" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="结束项" name="end">
                    <InputNumber min={0}/>
                </Form.Item>

                <Form.Item label="纯集标题" name="episodeTitleOnly" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="URL文件" name="file">
                    <Input placeholder="URL文件路径"/>
                </Form.Item>

                <Form.Item label="文件名长度限制" name="fileNameLength">
                    <InputNumber min={0}/>
                </Form.Item>

                <Form.Item label="仅显示信息" name="info" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="指定项目" name="items">
                    <Input placeholder="如: 1,5,6,8-10"/>
                </Form.Item>

                <Form.Item label="输出JSON" name="json" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="Referer" name="refer">
                    <Input/>
                </Form.Item>

                <Form.Item label="静默模式" name="silent" valuePropName="checked">
                    <Switch/>
                </Form.Item>

                <Form.Item label="起始项" name="start">
                    <InputNumber min={1}/>
                </Form.Item>

                <Form.Item label="流格式" name="streamFormat">
                    <Input/>
                </Form.Item>

                <Form.Item label="User-Agent" name="userAgent">
                    <Input/>
                </Form.Item>

                <Form.Item label="优酷ccode" name="youkuCcode">
                    <Input/>
                </Form.Item>

                <Form.Item label="优酷ckey" name="youkuCkey">
                    <Input/>
                </Form.Item>

                <Form.Item label="优酷密码" name="youkuPassword">
                    <Input.Password/>
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit" icon={<MdOutlineContentCopy/>}>
                        生成命令并复制
                    </Button>
                </Form.Item>
            </Form>
        </Card>
    );
}
