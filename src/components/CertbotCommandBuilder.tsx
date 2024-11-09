// src/components/CertbotCommandBuilder.tsx
import React, { useState } from 'react';
import {
    TextInput,
    Switch,
    Button,
    Card,
    Stack,
    Group,
    Select,
    MultiSelect,
    Tooltip,
    ActionIcon,
    Text,
    Badge,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { MdOutlineContentCopy } from "react-icons/md";
import { FaQuestionCircle, FaPlus, FaTrash } from "react-icons/fa";

interface CertbotFormValues {
    domains: string[];
    challengeType: string;
    staging?: boolean;
    agreeTos?: boolean;
    email?: string;
    serverType?: string;
    certPath?: string;
    keyPath?: string;
    quiet?: boolean;
    nonInteractive?: boolean;
}

export default function CertbotCommandBuilder() {
    const [newDomain, setNewDomain] = useState('');
    const [command, setCommand] = useState('');

    const form = useForm<CertbotFormValues>({
        initialValues: {
            domains: [],
            challengeType: 'dns-01',
            staging: false,
            agreeTos: true,
            quiet: false,
            nonInteractive: false,
        },
        validate: {
            domains: (value) => (value.length === 0 ? '请至少添加一个域名' : null),
            email: (value) => {
                if (!value) return null;
                return /^\S+@\S+$/.test(value) ? null : '请输入有效的邮箱地址';
            },
        },
    });

    const addDomain = () => {
        if (newDomain && !form.values.domains.includes(newDomain)) {
            form.setFieldValue('domains', [...form.values.domains, newDomain]);
            setNewDomain('');
        }
    };

    const removeDomain = (domain: string) => {
        form.setFieldValue('domains', form.values.domains.filter(d => d !== domain));
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addDomain();
        }
    };

    const buildCommand = (values: CertbotFormValues): string => {
        const parts: string[] = ['certbot', 'certonly'];

        values.domains.forEach(domain => {
            parts.push(`-d ${domain}`);
        });

        parts.push('--manual');
        parts.push(`--preferred-challenges ${values.challengeType}`);

        if (values.staging) {
            parts.push('--staging');
        }
        if (values.agreeTos) {
            parts.push('--agree-tos');
        }
        if (values.email) {
            parts.push(`--email ${values.email}`);
        }
        if (values.serverType) {
            parts.push(`--server-type ${values.serverType}`);
        }
        if (values.certPath) {
            parts.push(`--cert-path ${values.certPath}`);
        }
        if (values.keyPath) {
            parts.push(`--key-path ${values.keyPath}`);
        }
        if (values.quiet) {
            parts.push('--quiet');
        }
        if (values.nonInteractive) {
            parts.push('--non-interactive');
        }

        return parts.join(' ');
    };

    const onSubmit = (values: CertbotFormValues) => {
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
                    <Group align="flex-start">
                        <TextInput
                            style={{ flex: 1 }}
                            label="添加域名"
                            placeholder="输入域名后按回车或点击添加按钮"
                            value={newDomain}
                            onChange={(e) => setNewDomain(e.currentTarget.value)}
                            onKeyPress={handleKeyPress}
                            rightSection={
                                <ActionIcon
                                    onClick={addDomain}
                                    disabled={!newDomain}
                                    variant="filled"
                                    color="blue"
                                >
                                    <FaPlus size={14} />
                                </ActionIcon>
                            }
                        />
                    </Group>

                    {form.values.domains.length > 0 && (
                        <Stack gap="xs">
                            <Text size="sm" fw={500}>已添加的域名：</Text>
                            <Group gap="xs">
                                {form.values.domains.map((domain) => (
                                    <Badge
                                        key={domain}
                                        size="lg"
                                        rightSection={
                                            <ActionIcon
                                                size="xs"
                                                color="red"
                                                variant="transparent"
                                                onClick={() => removeDomain(domain)}
                                            >
                                                <FaTrash size={10} />
                                            </ActionIcon>
                                        }
                                    >
                                        {domain}
                                    </Badge>
                                ))}
                            </Group>
                        </Stack>
                    )}

                    <Select
                        label="验证方式"
                        data={[
                            { value: 'dns-01', label: 'DNS 验证' },
                            { value: 'http-01', label: 'HTTP 验证' },
                            { value: 'tls-alpn-01', label: 'TLS-ALPN 验证' },
                        ]}
                        {...form.getInputProps('challengeType')}
                    />

                    <TextInput
                        label="邮箱地址"
                        placeholder="用于接收证书过期通知"
                        {...form.getInputProps('email')}
                    />

                    <Switch
                        label="测试模式"
                        description="使用 Let's Encrypt 的测试环境"
                        {...form.getInputProps('staging', { type: 'checkbox' })}
                    />

                    <Switch
                        label="同意服务条款"
                        {...form.getInputProps('agreeTos', { type: 'checkbox' })}
                    />

                    <Switch
                        label="安静模式"
                        description="减少输出信息"
                        {...form.getInputProps('quiet', { type: 'checkbox' })}
                    />

                    <Switch
                        label="非交互式"
                        description="不需要用户输入"
                        {...form.getInputProps('nonInteractive', { type: 'checkbox' })}
                    />

                    <TextInput
                        label="证书路径"
                        placeholder="/etc/letsencrypt/live/domain/fullchain.pem"
                        {...form.getInputProps('certPath')}
                    />

                    <TextInput
                        label="私钥路径"
                        placeholder="/etc/letsencrypt/live/domain/privkey.pem"
                        {...form.getInputProps('keyPath')}
                    />

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
                        disabled={form.values.domains.length === 0}
                    >
                        生成命令并复制
                    </Button>
                </Stack>
            </form>
        </Card>
    );
}
