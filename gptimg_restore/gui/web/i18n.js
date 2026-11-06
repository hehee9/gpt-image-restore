/**
 * @file gptimg_restore/gui/web/i18n.js
 * @description 화면 문구 4개 언어 사전, 언어 결정과 정적 문구 적용
 * @note 정적 문구: data-i18n(textContent), data-i18n-aria-label, data-i18n-title 속성에 키 지정
 * @note 복수형 문구: 값이 {one, other, ...} 객체면 params.count 기준 Intl.PluralRules 범주 선택
 */




/* =================================== 전역 상수/변수 =================================== */


/** @type {{code: string, name: string}[]} 지원 언어와 각 언어 고유 이름 */
export const LOCALES = [
    { code: "en", name: "English" },
    { code: "ko", name: "한국어" },
    { code: "ja", name: "日本語" },
    { code: "zh-CN", name: "简体中文" },
];

const DEFAULT_LOCALE = "en";
const STORAGE_KEY = "gptimg-restore.locale";
// 브라우저 언어 기본 태그 -> 지원 언어 코드
const PRIMARY_TAGS = { en: "en", ko: "ko", ja: "ja", zh: "zh-CN" };

/** @type {Object<string, Object<string, string | Object<string, string>>>} 언어별 문구 사전. 모든 언어의 키 구성 동일 */
export const MESSAGES = {
    en: {
        "app.title": "GPT Image Restore",
        "language.label": "Language",
        "viewer.label": "Before and after",
        "viewer.before": "Original",
        "viewer.after": "Restored",
        "viewer.split": "Split position",
        "zoom.label": "Zoom",
        "zoom.out": "Zoom out",
        "zoom.fit": "Fit",
        "zoom.in": "Zoom in",
        "empty.drop": "Drop images here",
        "empty.pick": "Choose files",
        "queue.label": "Images",
        "queue.add": "Add images",
        "queue.remove": "Remove from list",
        "queue.waiting": "Waiting",
        "drop.add": "Drop to add",
        "panel.label": "Settings",
        "preset.heading": "Preset",
        "preset.safe": "Safe",
        "preset.balanced": "Balanced",
        "preset.strong": "Strong",
        "dots.heading": "Dot detection",
        "dots.conservative": "Narrow",
        "dots.strict": "Wide",
        "stages.heading": "Stage strength",
        "stages.reset": "Reset to preset",
        "stage.dots": "Bright dots",
        "stage.sparkle": "Sparkle",
        "stage.periodic": "Periodic pattern",
        "stage.grime": "Crosshatch",
        "stage.mottle": "Blotches",
        "stage.lattice": "Fine grid",
        "stage.surface": "Surface grain",
        "stage.contour": "Edge ripple",
        "stage.defocus": "Blurred background",
        "stage.cel_fill": "Cel fill",
        "stage.dither": "Dithering",
        "strength.off": "Off",
        "output.heading": "Output",
        "output.bitDepth": "Bit depth",
        "output.quality": "Quality",
        "progress.load": "Loading",
        "progress.encode": "Finishing",
        "progress.working": "Processing",
        "export.idle": "Export",
        "export.stop": "Stop",
        "export.saving": "Exporting",
        "toast.processFailed": "Couldn't restore {name}",
        "toast.exportFailed": "Couldn't export.",
        "toast.nothingToExport": "Nothing to export.",
        "toast.saved": { one: "Saved {count} file.", other: "Saved {count} files." },
        "toast.tooLarge": "{names}: Images larger than {side}×{side} can be restored in the desktop app.",
        "toast.desktopLink": "Get the desktop app",
        "toast.unreadable": { one: "{names}: Unsupported or unreadable file.", other: "{names}: Unsupported or unreadable files." },
        "names.more": { one: "{first} and {count} more", other: "{first} and {count} more" },
    },
    ko: {
        "app.title": "GPT Image Restore",
        "language.label": "언어",
        "viewer.label": "전후 비교",
        "viewer.before": "원본",
        "viewer.after": "보정",
        "viewer.split": "비교 위치",
        "zoom.label": "배율",
        "zoom.out": "축소",
        "zoom.fit": "맞춤",
        "zoom.in": "확대",
        "empty.drop": "이미지를 여기에 끌어다 놓으세요",
        "empty.pick": "파일 선택",
        "queue.label": "이미지 목록",
        "queue.add": "이미지 추가",
        "queue.remove": "목록에서 제거",
        "queue.waiting": "대기",
        "drop.add": "놓아서 추가",
        "panel.label": "설정",
        "preset.heading": "프리셋",
        "preset.safe": "최소",
        "preset.balanced": "균형",
        "preset.strong": "강력",
        "dots.heading": "점 감지 범위",
        "dots.conservative": "좁게",
        "dots.strict": "넓게",
        "stages.heading": "단계별 강도",
        "stages.reset": "프리셋 값으로",
        "stage.dots": "점묘 현상",
        "stage.sparkle": "반짝임",
        "stage.periodic": "주기 패턴",
        "stage.grime": "대각선 노이즈",
        "stage.mottle": "얼룩",
        "stage.lattice": "미세 격자",
        "stage.surface": "표면 입자",
        "stage.contour": "윤곽 물결",
        "stage.defocus": "흐린 배경",
        "stage.cel_fill": "셀 채색",
        "stage.dither": "디더링",
        "strength.off": "끔",
        "output.heading": "출력",
        "output.bitDepth": "비트 깊이",
        "output.quality": "품질",
        "progress.load": "불러오기",
        "progress.encode": "마무리",
        "progress.working": "처리 중",
        "export.idle": "내보내기",
        "export.stop": "중지",
        "export.saving": "내보내는 중",
        "toast.processFailed": "보정에 실패했습니다: {name}",
        "toast.exportFailed": "내보내지 못했습니다.",
        "toast.nothingToExport": "내보낼 결과가 없습니다.",
        "toast.saved": { other: "{count}개 파일을 저장했습니다." },
        "toast.tooLarge": "{names}: {side}×{side}보다 큰 이미지는 데스크톱 앱에서 보정할 수 있습니다.",
        "toast.desktopLink": "데스크톱 앱 받기",
        "toast.unreadable": { other: "{names}: 지원하지 않거나 열 수 없는 파일입니다." },
        "names.more": { other: "{first} 외 {count}개" },
    },
    ja: {
        "app.title": "GPT Image Restore",
        "language.label": "言語",
        "viewer.label": "補正前後の比較",
        "viewer.before": "元画像",
        "viewer.after": "補正後",
        "viewer.split": "比較位置",
        "zoom.label": "表示倍率",
        "zoom.out": "縮小",
        "zoom.fit": "全体表示",
        "zoom.in": "拡大",
        "empty.drop": "ここに画像をドロップしてください",
        "empty.pick": "ファイルを選択",
        "queue.label": "画像リスト",
        "queue.add": "画像を追加",
        "queue.remove": "リストから削除",
        "queue.waiting": "待機中",
        "drop.add": "ドロップして追加",
        "panel.label": "設定",
        "preset.heading": "プリセット",
        "preset.safe": "控えめ",
        "preset.balanced": "バランス",
        "preset.strong": "強め",
        "dots.heading": "点の検出範囲",
        "dots.conservative": "狭め",
        "dots.strict": "広め",
        "stages.heading": "処理別の強度",
        "stages.reset": "プリセット値に戻す",
        "stage.dots": "輝点",
        "stage.sparkle": "きらめき",
        "stage.periodic": "周期パターン",
        "stage.grime": "斜線ノイズ",
        "stage.mottle": "ムラ",
        "stage.lattice": "微細格子",
        "stage.surface": "表面の粒状感",
        "stage.contour": "輪郭の波打ち",
        "stage.defocus": "ぼけた背景",
        "stage.cel_fill": "セル塗り",
        "stage.dither": "ディザリング",
        "strength.off": "オフ",
        "output.heading": "出力",
        "output.bitDepth": "ビット深度",
        "output.quality": "品質",
        "progress.load": "読み込み",
        "progress.encode": "仕上げ",
        "progress.working": "処理中",
        "export.idle": "エクスポート",
        "export.stop": "停止",
        "export.saving": "エクスポート中",
        "toast.processFailed": "補正に失敗しました：{name}",
        "toast.exportFailed": "エクスポートできませんでした。",
        "toast.nothingToExport": "エクスポートできる画像がありません。",
        "toast.saved": { other: "{count} 件のファイルを保存しました。" },
        "toast.tooLarge": "{names}：{side}×{side} より大きい画像はデスクトップアプリで補正できます。",
        "toast.desktopLink": "デスクトップアプリを入手",
        "toast.unreadable": { other: "{names}：対応していないか、開けないファイルです。" },
        "names.more": { other: "{first} ほか {count} 件" },
    },
    "zh-CN": {
        "app.title": "GPT Image Restore",
        "language.label": "语言",
        "viewer.label": "修复前后对比",
        "viewer.before": "原图",
        "viewer.after": "修复后",
        "viewer.split": "对比位置",
        "zoom.label": "缩放",
        "zoom.out": "缩小",
        "zoom.fit": "适应窗口",
        "zoom.in": "放大",
        "empty.drop": "将图像拖放到此处",
        "empty.pick": "选择文件",
        "queue.label": "图像列表",
        "queue.add": "添加图像",
        "queue.remove": "从列表中移除",
        "queue.waiting": "等待中",
        "drop.add": "松开以添加",
        "panel.label": "设置",
        "preset.heading": "预设",
        "preset.safe": "轻度",
        "preset.balanced": "均衡",
        "preset.strong": "强力",
        "dots.heading": "点检测范围",
        "dots.conservative": "较窄",
        "dots.strict": "较宽",
        "stages.heading": "各阶段强度",
        "stages.reset": "恢复预设值",
        "stage.dots": "亮点",
        "stage.sparkle": "闪光点",
        "stage.periodic": "周期图案",
        "stage.grime": "斜纹噪点",
        "stage.mottle": "斑块",
        "stage.lattice": "细网格",
        "stage.surface": "表面颗粒",
        "stage.contour": "轮廓波纹",
        "stage.defocus": "虚化背景",
        "stage.cel_fill": "平涂",
        "stage.dither": "抖动",
        "strength.off": "关",
        "output.heading": "输出",
        "output.bitDepth": "位深度",
        "output.quality": "质量",
        "progress.load": "加载",
        "progress.encode": "收尾",
        "progress.working": "处理中",
        "export.idle": "导出",
        "export.stop": "停止",
        "export.saving": "正在导出",
        "toast.processFailed": "修复失败：{name}",
        "toast.exportFailed": "无法导出。",
        "toast.nothingToExport": "没有可导出的结果。",
        "toast.saved": { other: "已保存 {count} 个文件。" },
        "toast.tooLarge": "{names}：大于 {side}×{side} 的图像可在桌面应用中修复。",
        "toast.desktopLink": "获取桌面应用",
        "toast.unreadable": { other: "{names}：文件不受支持或无法打开。" },
        "names.more": { other: "{first} 及另外 {count} 个" },
    },
};

let current = DEFAULT_LOCALE;
let pluralRules = new Intl.PluralRules(DEFAULT_LOCALE);




/* =================================== 메인 로직 =================================== */


/**
 * 현재 언어 문구 반환. {이름} 자리에 params 값 대입
 * @param {string} key 사전 키
 * @param {Object<string, string | number>} [params] 대입 값. 복수형 문구는 count 필수
 * @returns {string}
 */
export function t(key, params = {}) {
    let message = MESSAGES[current][key];
    if (typeof message === "object") {
        message = message[pluralRules.select(params.count)] ?? message.other;
    }
    return message.replace(/\{(\w+)\}/g, (match, name) => String(params[name]));
}


/** @description 사전에 있는 키인지 판정 */
export function hasMessage(key) {
    return key in MESSAGES[current];
}


/** @description 현재 언어 코드 */
export function getLocale() {
    return current;
}


/**
 * 저장한 선택 -> 브라우저 선호 언어 중 첫 지원 언어 -> 영어 순으로 결정
 * @returns {string} LOCALES의 code
 */
export function detectLocale() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (LOCALES.some((locale) => locale.code === saved)) {
        return saved;
    }
    for (const tag of navigator.languages) {
        const code = PRIMARY_TAGS[tag.split("-")[0].toLowerCase()];
        if (code) {
            return code;
        }
    }
    return DEFAULT_LOCALE;
}


/** @description 사용자가 고른 언어 저장 */
export function saveLocale(code) {
    localStorage.setItem(STORAGE_KEY, code);
}


/** @description 문서 안 data-i18n 계열 속성의 정적 문구 갱신 */
export function applyTranslations(root = document) {
    for (const element of root.querySelectorAll("[data-i18n]")) {
        element.textContent = t(element.dataset.i18n);
    }
    for (const element of root.querySelectorAll("[data-i18n-aria-label]")) {
        element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
    }
    for (const element of root.querySelectorAll("[data-i18n-title]")) {
        element.title = t(element.dataset.i18nTitle);
    }
}


/**
 * 언어 전환: 문서 언어, 제목, 정적 문구 갱신. 동적 문구 재표시는 호출 측 담당
 * @param {string} code LOCALES의 code
 */
export function setLocale(code) {
    current = code;
    pluralRules = new Intl.PluralRules(code);
    document.documentElement.lang = code;
    document.title = t("app.title");
    applyTranslations();
}
