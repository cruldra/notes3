import React, { useState } from 'react';
import {
    ActionIcon,
    Button,
    Card,
    Group,
    Select,
    Stack,
    Text,
    TextInput,
    NumberInput,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { MdOutlineContentCopy } from "react-icons/md";

interface DatabaseFormValues {
    provider: string;
    username: string;
    password: string;
    host: string;
    port: number;
    database: string;
    schema?: string;
}

export default function DatabaseURLBuilder() {
    const [connectionUrl, setConnectionUrl] = useState('');

    const form = useForm<DatabaseFormValues>({
        initialValues: {
            provider: 'postgresql',
            username: '',
            password: '',
            host: 'localhost',
            port: 5432,
            database: '',
            schema: 'public'
        },
        validate: {
            username: (value) => (!value ? '请输入用户名' : null),
            password: (value) => (!value ? '请输入密码' : null),
            host: (value) => (!value ? '请输入主机地址' : null),
            port: (value) => (!value ? '请输入端口号' : null),
            database: (value) => (!value ? '请输入数据库名' : null),
        },
    });

    const buildConnectionUrl = (values: DatabaseFormValues): string => {
        const { provider, username, password, host, port, database, schema } = values;

        let url = `${provider}://`;

        // 添加认证信息
        if (username || password) {
            url += `${encodeURIComponent(username)}:${encodeURIComponent(password)}@`;
        }

        // 添加主机和端口
        url += `${host}:${port}`;

        // 添加数据库名
        url += `/${database}`;

        // 添加schema（如果是PostgreSQL）
        if (provider === 'postgresql' && schema) {
            url += `?schema=${schema}`;
        }

        return url;
    };

    const getDefaultPort = (provider: string): number => {
        switch (provider) {
            case 'postgresql':
                return 5432;
            case 'mysql':
                return 3306;
            case 'sqlserver':
                return 1433;
            case 'mongodb':
                return 27017;
            default:
                return 5432;
        }
    };

    const onSubmit = (values: DatabaseFormValues) => {
        const generatedUrl = buildConnectionUrl(values);
        setConnectionUrl(generatedUrl);
        navigator.clipboard.writeText(generatedUrl)
            .then(() => notifications.show({
                message: '连接URL已复制到剪贴板',
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
                        label="数据库类型"
                        data={[
                            { value: 'postgresql', label: 'PostgreSQL' },
                            { value: 'mysql', label: 'MySQL' },
                            { value: 'sqlserver', label: 'SQL Server' },
                            { value: 'mongodb', label: 'MongoDB' },
                            { value: 'sqlite', label: 'SQLite' },
                        ]}
                        onChange={(value) => {
                            form.setFieldValue('provider', value || 'postgresql');
                            form.setFieldValue('port', getDefaultPort(value || 'postgresql'));
                        }}
                        {...form.getInputProps('provider')}
                    />

                    <TextInput
                        label="用户名"
                        placeholder="database_user"
                        {...form.getInputProps('username')}
                    />

                    <TextInput
                        label="密码"
                        type="password"
                        placeholder="your_password"
                        {...form.getInputProps('password')}
                    />

                    <TextInput
                        label="主机地址"
                        placeholder="localhost"
                        {...form.getInputProps('host')}
                    />

                    <NumberInput
                        label="端口"
                        placeholder="5432"
                        {...form.getInputProps('port')}
                    />

                    <TextInput
                        label="数据库名"
                        placeholder="my_database"
                        {...form.getInputProps('database')}
                    />

                    {form.values.provider === 'postgresql' && (
                        <TextInput
                            label="Schema"
                            placeholder="public"
                            {...form.getInputProps('schema')}
                        />
                    )}

                    {connectionUrl && (
                        <Card withBorder>
                            <Text size="sm" fw={500}>数据库连接 URL：</Text>
                            <Text style={{ wordBreak: 'break-all' }} mt="xs">
                                {connectionUrl}
                            </Text>
                        </Card>
                    )}

                    <Button
                        type="submit"
                        leftSection={<MdOutlineContentCopy size={14}/>}
                    >
                        生成连接 URL 并复制
                    </Button>
                </Stack>
            </form>
        </Card>
    );
}
