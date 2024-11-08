import React from 'react';
import {ConfigProvider, theme} from "antd";

// Default implementation, that you can customize
export default function Root({children}) {
    return <ConfigProvider
        theme={{
            algorithm: theme.darkAlgorithm,
        }}
    >
        {children}
    </ConfigProvider>;
}
