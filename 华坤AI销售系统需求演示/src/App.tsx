import React, { useState, Suspense } from 'react';
import { menuItems } from './config';
import { AppShell, Burger, NavLink, Title, LoadingOverlay, ScrollArea } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';

const App = () => {
    const [opened, { toggle }] = useDisclosure();
    const [activeId, setActiveId] = useState(menuItems[0].id);

    const activeItem = menuItems.find(item => item.id === activeId);

    return (
        <AppShell
            header={{ height: 60 }}
            navbar={{ width: 300, breakpoint: 'sm', collapsed: { mobile: !opened } }}
            padding="md"
        >
            <AppShell.Header className="flex items-center px-4">
                <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
                <Title order={3} className="ml-4">华坤 AI 销售系统 - 交互演示</Title>
            </AppShell.Header>

            <AppShell.Navbar p="md">
                <AppShell.Section grow component={ScrollArea}>
                    {menuItems.map((item) => (
                        <NavLink
                            key={item.id}
                            label={item.label}
                            active={activeId === item.id}
                            onClick={() => {
                                setActiveId(item.id);
                                toggle();
                            }}
                            className="rounded-md mb-1"
                        />
                    ))}
                </AppShell.Section>
            </AppShell.Navbar>

            <AppShell.Main className="bg-slate-50 min-h-screen">
                <Suspense fallback={<LoadingOverlay visible />}>
                    {activeItem && <activeItem.component />}
                </Suspense>
            </AppShell.Main>
        </AppShell>
    );
};

export default App;
