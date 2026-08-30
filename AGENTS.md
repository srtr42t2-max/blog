# AGENTS.md — 项目记忆与改造经验

> 本文件是本博客项目的长期记忆，记录关键架构决策与修改经验。**之后修改本文涉及的任何区域时，必须同步更新本文件。**

## 项目概况

- 基于 **Mizuki / Fuwari 系主题** 的 Astro 静态博客：Astro 5 + Svelte 5（runes 语法）+ Tailwind CSS v4（`@theme` / `@utility` 新语法）+ Swup 页面过渡。
- 站点 `base: "/blog"`（astro.config.mjs），部署在子路径下，写链接时注意用 `url()` 工具函数。
- 常用命令：
  - 开发：`pnpm dev`
  - 构建：`pnpm build`（含 prebuild 同步内容 + pagefind 索引）
  - 类型检查：`pnpm check` / `pnpm type-check`
  - 测试：`pnpm test`
  - Lint/格式化：`pnpm lint` / `pnpm format`（Biome）

## 字体体系（2026-08 改造）

- **正文/标题使用系统字体栈**，定义在 `src/styles/main.css` 的 `@theme { --font-sans: ... }`（ui-sans-serif, system-ui, Segoe UI, PingFang SC, Microsoft YaHei 等）。
- 原先的圆润字体 **Zen Maru Gothic（Latin）+ loli.ttf（CJK）已弃用**（用户反馈太圆）。ttf 文件仍保留在 `src/assets/fonts/` 但未被引用；如需恢复，在 `astro.config.mjs` 的 `fonts` 数组中加回并在 `src/layouts/Layout.astro` 加 `<Font cssVariable>` 标签。
- **代码字体 JetBrains Mono** 仍由 Astro Font API 注入 `--font-jetbrains-mono`（astro.config.mjs `fonts` 数组），在 `markdown.css` 中引用。
- 全站只有一套 `--font-sans`，正文与标题共用，无独立标题字体。

## 字号调节机制（重要：双变量模型）

用户可在导航栏设置面板（调色板图标）中调节字号（80%–120%）。实现要点：

- **两个 CSS 变量相乘组合**，计算规则在 `src/styles/main.css` 的 `html` 选择器中：
  - `--font-size-scale`：用户设置的字号比例（默认 1），由设置面板写入；
  - `--page-scale`：宽屏自动缩放比例（pageScaling，默认 1），由 `HeadTags.astro` 内联脚本写入。
- 移动端基础字号 14px，桌面端（≥768px）16px：`calc(16px * var(--page-scale, 1) * var(--font-size-scale, 1))`。
- **不要直接写 `documentElement.style.fontSize`**——inline 样式会覆盖 CSS 变量计算。pageScaling 已改为设置 `--page-scale` 变量；新增任何根字号逻辑都应通过 CSS 变量。
- localStorage key：`fontSizeScale`（数值，如 1.05）。防闪烁读取在 `src/layouts/partials/HeadTags.astro` 的 `is:inline` 脚本中。
- 工具函数：`getFontSizeScale()` / `setFontSizeScale()` 在 `src/utils/setting-utils.ts`。

## 布局宽度

- **页面总宽**：`PAGE_WIDTH`（单位 rem）在 `src/constants/constants.ts`，当前 **100**（原 90，2026-08 加宽正文）。注入为 `--page-width` CSS 变量（Layout.astro），被 `MainGridLayout.astro` 和 `Navbar.astro` 的 `max-w` 使用——改这一个数字即全站生效。
- **网格列**：桌面端 `lg:grid-cols-[17.5rem_1fr_17.5rem]`（侧栏/正文/侧栏），在 `src/utils/grid-layout-utils.ts`。正文宽 ≈ PAGE_WIDTH − 35rem。
- 正文 prose 已 `!max-w-none`（`src/components/misc/Markdown.astro`），即正文宽度完全由网格中间列决定。

## 设置面板扩展模式（新增设置项的固定套路）

设置面板：`src/components/features/settings/SettingsPanel.svelte`（Svelte 5 runes）。新增一个设置项按 5 步走：

1. `src/utils/setting-utils.ts`：加 `getXxx()` / `setXxx()`（localStorage 持久化 + 写 CSS 变量/DOM + 必要时 dispatch CustomEvent）；
2. `src/layouts/partials/HeadTags.astro` 的 `is:inline` 防闪烁脚本：从 localStorage 提前读取并应用，避免页面闪烁；
3. `SettingsPanel.svelte`：`$state` 状态 + `$effect` 调用 set 函数 + UI（复用 `.slider` 样式与重置按钮模式，参考"字体大小"/"主题色"区块）；
4. i18n：`src/i18n/i18nKey.ts` 加枚举键，并在 `zh_CN / zh_TW / en / ja` 四个语言文件中补翻译（**四个都要，缺一不可**）；
5. 验证：`pnpm astro build` 通过 + 刷新后设置仍生效。

滑块填充进度由 `refreshAllRangeProgress()` 统一处理，新 range input 无需额外接线。

## 现有 localStorage keys 清单

`theme`、`hue`、`fontSizeScale`、`wallpaperMode`、`overlayOpacity`、`overlayBlur`、`overlayCardOpacity`、`fullscreenOpacity`、`fullscreenBlur`、`wavesEnabled`、`bannerTitleEnabled`、`bannerCarouselEnabled`、`sakuraEnabled`、`postListLayout`、`announcementClosed`。

新增 key 时请补充到此清单。

## 其他注意事项

- 暗色模式：`@custom-variant dark (&:where(.dark, .dark *))`（main.css），切换逻辑在 `applyThemeToDocument()`，同时维护 `data-theme="github-dark|github-light"` 供 Expressive Code 代码块用。
- Swup 页面切换后需要重新初始化的逻辑，挂在 `window.swup.hooks.on("content:replace", ...)` 或 `swup:page:view` 事件。
- 项目有根目录 `CLAUDE.md`（通用行为约定：简单优先、外科手术式修改、不重构无关代码），本文件补充项目特定知识。
