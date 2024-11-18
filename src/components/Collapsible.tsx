// src/components/Collapsible.js
import React, { useState, useRef, useEffect } from 'react';
import styles from './Collapsible.module.css';

export default function Collapsible({
                                        children,
                                        title = "展开代码",
                                        defaultOpen = false,
                                        maxHeight = 500,
                                        showLineNumbers = true,
                                        language
                                    }) {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    const contentRef = useRef(null);
    const [height, setHeight] = useState(0);

    useEffect(() => {
        if (contentRef.current) {
            setHeight(contentRef.current.scrollHeight);
        }
    }, [children]);

    return (
        <div className={styles.collapsible}>
            <button
                className={styles.toggle}
                onClick={() => setIsOpen(!isOpen)}
                aria-expanded={isOpen}
            >
                <span className={styles.title}>{title}</span>
                <span className={styles.icon}>
          {isOpen ? '▼' : '▶'}
        </span>
            </button>
            <div
                ref={contentRef}
                className={`${styles.content} ${isOpen ? styles.open : ''}`}
                style={{
                    maxHeight: isOpen ? `${maxHeight}px` : 0
                }}
            >
                <div className={styles.codeWrapper}>
                    {React.Children.map(children, child => {
                        if (React.isValidElement(child) && child.type === 'pre') {
                            return React.cloneElement(child, {
                                className: `${child.props.className || ''} ${showLineNumbers ? 'line-numbers' : ''}`,
                                ...language && { 'data-language': language }
                            });
                        }
                        return child;
                    })}
                </div>
            </div>
        </div>
    );
}
