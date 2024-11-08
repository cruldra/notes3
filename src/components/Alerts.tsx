// src/components/Alerts/index.tsx
import React, {ReactNode} from 'react';
import {Alert} from '@mantine/core';
import {
    IoMdInformationCircleOutline,
    IoMdCheckmarkCircleOutline,
    IoMdWarning,
    IoMdCloseCircleOutline
} from "react-icons/io";

interface AlertProps {
    children: ReactNode;
    title?: string;
}

export function Info({children, title = "提示"}: AlertProps) {
    return (
        <Alert
            variant="light"
            color="blue"
            title={title}
            icon={<IoMdInformationCircleOutline size={18}/>}
        >
            {children}
        </Alert>
    );
}

export function Tip({children, title = "成功"}: AlertProps) {
    return (
        <Alert
            variant="light"
            color="green"
            title={title}
            icon={<IoMdCheckmarkCircleOutline size={18}/>}
        >
            {children}
        </Alert>
    );
}

export function Warn({children, title = "警告"}: AlertProps) {
    return (
        <Alert
            variant="light"
            color="yellow"
            title={title}
            icon={<IoMdWarning size={18}/>}
        >
            {children}
        </Alert>
    );
}

export function Error({children, title = "错误"}: AlertProps) {
    return (
        <Alert
            variant="light"
            color="red"
            title={title}
            icon={<IoMdCloseCircleOutline size={18}/>}
        >
            {children}
        </Alert>
    );
}
