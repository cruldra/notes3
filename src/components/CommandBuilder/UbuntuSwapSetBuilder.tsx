import React from 'react';
import { Stack, TextInput, NumberInput, Card, Button } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { MdOutlineContentCopy } from "react-icons/md";
import { useForm } from '@mantine/form';

export default function SwapFileBuilder() {
    const form = useForm({
        initialValues: {
            size: 4,
            swapfile: '/swapfile',
            swappiness: 10,
        },
        validate: {
            size: (value) => (value <= 0 ? '大小必须大于0' : null),
            swapfile: (value) => (!value ? '请输入交换文件路径' : null),
            swappiness: (value) => (
                value < 0 || value > 100 ? 'swappiness值必须在0-100之间' : null
            ),
        }
    });

    const buildCommand = (values) => {
        return [
            `sudo fallocate -l ${values.size}G ${values.swapfile}`,
            `sudo chmod 600 ${values.swapfile}`,
            `sudo mkswap ${values.swapfile}`,
            `sudo swapon ${values.swapfile}`,
            `sudo echo '${values.swapfile} none swap sw 0 0' | sudo tee -a /etc/fstab`,
            `sudo echo 'vm.swappiness=${values.swappiness}' | sudo tee -a /etc/sysctl.conf`
        ].join(' && ');
    };

    const handleSubmit = (values) => {
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
            <form onSubmit={form.onSubmit(handleSubmit)}>
                <Stack gap="md">
                    <NumberInput
                        label="交换文件大小 (GB)"
                        placeholder="输入交换文件大小"
                        min={1}
                        {...form.getInputProps('size')}
                    />

                    <TextInput
                        label="交换文件路径"
                        placeholder="输入交换文件路径"
                        {...form.getInputProps('swapfile')}
                    />

                    <NumberInput
                        label="Swappiness值"
                        description="设置系统对交换分区的使用程度（0-100）"
                        placeholder="输入0-100之间的值"
                        min={0}
                        max={100}
                        {...form.getInputProps('swappiness')}
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
