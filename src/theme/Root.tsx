import React from 'react';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import {createTheme, MantineProvider} from '@mantine/core';
import {Notifications} from "@mantine/notifications";


// Default implementation, that you can customize
export default function Root({children}) {


    const theme = createTheme({
        /** Your theme override here */

    });
    return  <MantineProvider defaultColorScheme="auto" theme={theme}>
        <Notifications />
        {children}
    </MantineProvider>;
}
