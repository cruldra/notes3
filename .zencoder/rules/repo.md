---
description: Repository Information Overview
alwaysApply: true
---

# Notes3 Documentation Site

## Summary
A Docusaurus-based documentation and blog website built with React and TypeScript. The project serves as a comprehensive knowledge base covering multiple technical domains including JVM, Frontend, Tools, AI, Python, Rust, SoftwareEngineering, and Games, with support for Mermaid diagrams and code syntax highlighting.

## Structure
- **`docs/`** - Main documentation content organized by topic (AI, CodeSnippets, FrontEnd, Games, Go, JVM, Personal, Python, Rust, SoftwareEngineering, Tools)
- **`blog/`** - Blog posts with metadata and tags configuration
- **`src/`** - React components, pages, CSS themes, and custom components
- **`static/`** - Static assets (images, robots configuration)
- **`.github/`** - GitHub workflows and CI/CD configurations
- **`.vscode/`** - VS Code settings and debug configurations

## Language & Runtime
**Language**: TypeScript  
**Node.js Version**: >= 18.0  
**TypeScript Version**: ~5.8.3  
**Package Manager**: pnpm  
**Build System**: Docusaurus (static site generator)

## Dependencies

**Main Dependencies**:
- `@docusaurus/core@3.9.2` - Core Docusaurus framework
- `@docusaurus/preset-classic@3.9.2` - Classic preset with docs and blog
- `@docusaurus/theme-mermaid@3.9.2` - Mermaid diagram support
- `react@19.1.0` - UI library
- `react-dom@19.1.0` - React DOM renderer
- `@mantine/core@8.0.0` - Component library
- `@mantine/form@8.0.0` - Form utilities
- `@mantine/hooks@8.0.0` - React hooks library
- `@mantine/notifications@8.0.0` - Notification system
- `@mdx-js/react@3.1.0` - MDX support for React
- `@vidstack/react@1.12.13` - Video player component
- `@saucelabs/theme-github-codeblock@0.3.0` - GitHub-style code blocks
- `prism-react-renderer@2.4.1` - Syntax highlighting
- `react-icons@5.5.0` - Icon library
- `clsx@2.1.1` - Utility for className management

**Development Dependencies**:
- `@docusaurus/module-type-aliases@3.9.2` - Type aliases
- `@docusaurus/tsconfig@3.9.2` - TypeScript configuration
- `@docusaurus/types@3.9.2` - Type definitions
- `cross-env@7.0.3` - Cross-platform environment variable support
- `typescript@~5.8.3` - TypeScript compiler

## Build & Installation

**Install dependencies**:
```bash
pnpm install
```

**Start development server** (port 3333):
```bash
pnpm start
```

**Build static site**:
```bash
pnpm build
```

**Serve production build locally**:
```bash
pnpm serve
```

**Deploy to GitHub Pages**:
```bash
cross-env GIT_USER=cruldra pnpm deploy
```

**Additional commands**:
- `pnpm docusaurus` - Direct Docusaurus CLI access
- `pnpm swizzle` - Customize Docusaurus components
- `pnpm clear` - Clear Docusaurus cache
- `pnpm write-translations` - Extract and write translation files
- `pnpm write-heading-ids` - Auto-generate heading IDs

## Configuration

**TypeScript**: Extends `@docusaurus/tsconfig` with base URL configuration for absolute imports

**Docusaurus Config** (`docusaurus.config.ts`):
- **Site Title**: Cruldra
- **Base URL**: `/notes3/` (GitHub Pages deployment)
- **Organization**: cruldra
- **Markdown**: Mermaid diagram support enabled
- **Theme**: Light mode default with GitHub/Dracula Prism themes
- **Documentation Sidebar**: 10 main sections (JVM, FrontEnd, Tools, Personal, AI, Python, Rust, SoftwareEngineering, Games, Go)
- **Blog**: RSS/Atom feed support with reading time display

**Sidebar Configuration** (`sidebars.ts`): Defines doc navigation structure and categories for each technical domain

## Browser Support
- **Production**: > 0.5% market share, not dead, not Opera Mini
- **Development**: Latest 3 versions of Chrome, Firefox; latest 5 versions of Safari

## Key Files
- **Entry Point**: `docusaurus.config.ts` - Main configuration
- **Theme**: `src/css/custom.css` - Custom styling
- **Navigation**: `sidebars.ts` - Documentation structure
- **Package Manager Config**: `pnpm-workspace.yaml` - Workspace settings for core-js dependencies
