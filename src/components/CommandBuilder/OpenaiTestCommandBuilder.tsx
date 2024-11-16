import React from 'react';
import {
    ActionIcon,
    Button,
    Card,
    Select,
    Stack,
    TextInput,
    Text,
} from '@mantine/core';
import {useForm} from '@mantine/form';
import {notifications} from '@mantine/notifications';
import {MdOutlineContentCopy} from "react-icons/md";

interface ApiTestFormValues {
    apiEndpoint: string;
    apiKey: string;
    model: string;
    systemPrompt: string;
    userPrompt: string;
}

export default function OpenaiTestCommandBuilder() {
    const form = useForm<ApiTestFormValues>({
        initialValues: {
            apiEndpoint: 'https://api.openai.com/v1/chat/completions',
            apiKey: '',
            model: 'gpt-4',
            systemPrompt: 'You are a helpful assistant.',
            userPrompt: 'Hello!',
        },
        validate: {
            apiEndpoint: (value) => (!value ? '请输入 API 地址' : null),
            apiKey: (value) => (!value ? '请输入 API Key' : null),
        },
    });

    const buildCommand = (values: ApiTestFormValues): string => {
        const payload = {
            model: values.model,
            messages: [
                {
                    role: "system",
                    content: values.systemPrompt
                },
                {
                    role: "user",
                    content: values.userPrompt
                }
            ]
        };

        return `curl ${values.apiEndpoint} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${values.apiKey}" \\
  -d '${JSON.stringify(payload, null, 2)}'`;
    };

    const onSubmit = (values: ApiTestFormValues) => {
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
        <Card shadow="sm" p="lg">
            <form onSubmit={form.onSubmit(onSubmit)}>
                <Stack gap="md">
                    <TextInput
                        label="API 地址"
                        placeholder="https://api.openai.com/v1/chat/completions"
                        {...form.getInputProps('apiEndpoint')}
                    />

                    <TextInput
                        label="API Key"
                        type="password"
                        placeholder="sk-..."
                        {...form.getInputProps('apiKey')}
                    />

                    <Select
                        label="模型"
                        data={[
                            {value: 'gpt-4', label: 'GPT-4'},
                            {value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo'},
                        ]}
                        {...form.getInputProps('model')}
                    />

                    <TextInput
                        label="System Prompt"
                        placeholder="系统提示词"
                        {...form.getInputProps('systemPrompt')}
                    />

                    <TextInput
                        label="User Prompt"
                        placeholder="用户提示词"
                        {...form.getInputProps('userPrompt')}
                    />

                    <Button
                        type="submit"
                        leftSection={<MdOutlineContentCopy size={14}/>}
                    >
                        生成命令并复制
                    </Button>

                    {form.values.apiEndpoint && form.values.apiKey && (
                        <Card withBorder>
                            <Text size="sm" fw={500}>生成的 curl 命令：</Text>
                            <Text style={{wordBreak: 'break-all', whiteSpace: 'pre-wrap'}} mt="xs">
                                {buildCommand(form.values)}
                            </Text>
                        </Card>
                    )}
                </Stack>
            </form>
        </Card>
    );
}
