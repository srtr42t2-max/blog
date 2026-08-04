import type { PioConfig } from "../types/config";

// Pio 看板娘配置
export const pioConfig: PioConfig = {
	enable: false, // 暂时关闭模板看板娘
	models: ["/pio/models/NOIR/noir.model3.json"], // 默认模型路径
	position: "left", // 模型位置
	width: 280, // 默认宽度
	height: 250, // 默认高度
	mode: "draggable", // 默认为可拖拽模式
	hiddenOnMobile: true, // 默认在移动设备上隐藏
	hideAboutMenu: false, // 隐藏内置 About 菜单按钮
	dialog: {
		welcome: "KirakiraDokidoki today, too! ✨⭐", // 欢迎词
		touch: ["Kirakira!", "Dokidoki!"], // 触摸提示
		home: "Back to pplk_blog", // 首页提示
		skin: ["A new look!", "Looks great!"], // 换装提示
		close: "See you next time!", // 关闭提示
		link: "https://github.com/srtr42t2-max", // 关于链接
	},
};
