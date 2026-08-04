import type { AnnouncementConfig } from "../types/config";

// 公告栏配置
export const announcementConfig: AnnouncementConfig = {
	title: "", // 公告标题，填空使用i18n字符串Key.announcement
	content: "KirakiraDokidoki today, too! ✨⭐ 欢迎来到 pplk_blog。", // 公告内容
	closable: true, // 允许用户关闭公告
	link: {
		enable: false, // 暂无公告链接
		text: "",
		url: "",
		external: false, // 内部链接
	},
};
