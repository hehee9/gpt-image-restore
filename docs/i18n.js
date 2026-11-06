/**
 * @file docs/i18n.js
 * @description 쇼케이스 페이지 4개 언어 문구, 언어 감지/저장, 화면 문구 적용
 */


const I18n = (() => {




/* =================================== 전역 상수/변수 =================================== */


const STORAGE_KEY = "gptimg-restore.locale";
const FALLBACK = "en";

/** 언어 코드별 고유 이름, 메뉴 표시 순서 */
const LOCALES = Object.freeze({
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
    "zh-CN": "简体中文",
});

/** navigator.languages 주 언어 태그 -> 지원 언어 코드 */
const PRIMARY_TAGS = { en: "en", ko: "ko", ja: "ja", zh: "zh-CN" };

/**
 * 언어별 문구, 모든 언어가 같은 키 집합을 가짐
 * item.<id>: showcase.json 항목 제목 번역 (한국어 제목 기준)
 */
const MESSAGES = {
    "en": {
        "page.title": "gpt-image-restore: before and after",
        "page.description": "Before-and-after comparisons of GPT Image 2.6 noise removal",
        "page.heading": "GPT Image 2.6 noise removal: before and after",
        "nav.index": "Comparisons",
        "locale.label": "Language",
        "toolbar.versions": "Result shown on the right",
        "version.original": "Original",
        "version.balanced": "Balanced",
        "version.strong": "Strong",
        "tool.boost": "Contrast boost",
        "image.alt": "{title}, {version}",
        "divider.label": "Divider between Original and {version}",
        "divider.valuetext": "Original {left}%, {version} {right}%",
        "status.loadFailed": "Couldn't load the comparison list (showcase.json).",
        "status.loadFailedHttp": "Couldn't load the comparison list (showcase.json). (HTTP {status})",
        "status.empty": "No comparison images yet.",
        "footer.notice": "The sample images are AI-generated with GPT Image 2.6 and are only for demonstrating before-and-after results. They are not covered by the project's AGPL license, and reuse is not permitted.",
        "footer.repo": "GitHub repository",
        "item.handbag": "Glossy leather handbag",
        "item.poster": "English notice poster",
        "item.cel-illustration": "Cel-shaded illustration",
        "item.portrait": "Portrait with a blurred forest background",
        "item.teapot": "Matte teapot on a dark background",
        "item.soup": "Pumpkin soup on a white plate",
    },
    "ko": {
        "page.title": "gpt-image-restore 전후 비교",
        "page.description": "GPT Image 2.6 생성 노이즈 제거 전후 비교",
        "page.heading": "GPT Image 2.6 노이즈 제거 전후 비교",
        "nav.index": "비교 항목",
        "locale.label": "언어",
        "toolbar.versions": "오른쪽에 표시할 결과",
        "version.original": "원본",
        "version.balanced": "균형",
        "version.strong": "강력",
        "tool.boost": "대비 강조",
        "image.alt": "{title} {version}",
        "divider.label": "원본과 {version} 경계 위치",
        "divider.valuetext": "원본 {left}%, {version} {right}%",
        "status.loadFailed": "비교 목록(showcase.json)을 불러오지 못했습니다.",
        "status.loadFailedHttp": "비교 목록(showcase.json)을 불러오지 못했습니다. (HTTP {status})",
        "status.empty": "아직 등록된 비교 이미지가 없습니다.",
        "footer.notice": "예시 이미지는 GPT Image 2.6으로 생성한 AI 이미지이며, 보정 전후 비교 시연용입니다. 이 이미지는 프로젝트의 AGPL 라이선스 적용 대상이 아니며, 재사용을 허락하지 않습니다.",
        "footer.repo": "GitHub 저장소",
        "item.handbag": "광택 가죽 핸드백",
        "item.poster": "영어 안내 포스터",
        "item.cel-illustration": "셀 채색 일러스트",
        "item.portrait": "흐린 숲 배경 인물 사진",
        "item.teapot": "어두운 배경의 무광 찻주전자",
        "item.soup": "흰 접시의 호박 수프",
    },
    "ja": {
        "page.title": "gpt-image-restore 前後比較",
        "page.description": "GPT Image 2.6 で生成した画像のノイズ除去前後を比較",
        "page.heading": "GPT Image 2.6 ノイズ除去の前後比較",
        "nav.index": "比較項目",
        "locale.label": "言語",
        "toolbar.versions": "右側に表示する結果",
        "version.original": "元画像",
        "version.balanced": "バランス",
        "version.strong": "強め",
        "tool.boost": "コントラスト強調",
        "image.alt": "{title}（{version}）",
        "divider.label": "元画像と{version}の境界位置",
        "divider.valuetext": "元画像 {left}%、{version} {right}%",
        "status.loadFailed": "比較リスト（showcase.json）を読み込めませんでした。",
        "status.loadFailedHttp": "比較リスト（showcase.json）を読み込めませんでした。（HTTP {status}）",
        "status.empty": "登録された比較画像はまだありません。",
        "footer.notice": "サンプル画像は GPT Image 2.6 で生成した AI 画像で、補正前後の比較デモ用です。これらの画像は本プロジェクトの AGPL ライセンスの対象外であり、再利用は許可していません。",
        "footer.repo": "GitHub リポジトリ",
        "item.handbag": "光沢のある革のハンドバッグ",
        "item.poster": "英語の案内ポスター",
        "item.cel-illustration": "セル塗りのイラスト",
        "item.portrait": "ぼけた森を背景にした人物写真",
        "item.teapot": "暗い背景のマットなティーポット",
        "item.soup": "白い皿のかぼちゃスープ",
    },
    "zh-CN": {
        "page.title": "gpt-image-restore 前后对比",
        "page.description": "GPT Image 2.6 生成图像的降噪前后对比",
        "page.heading": "GPT Image 2.6 噪点去除前后对比",
        "nav.index": "对比项目",
        "locale.label": "语言",
        "toolbar.versions": "右侧显示的结果",
        "version.original": "原图",
        "version.balanced": "均衡",
        "version.strong": "强力",
        "tool.boost": "增强对比度",
        "image.alt": "{title}（{version}）",
        "divider.label": "原图与{version}的分界位置",
        "divider.valuetext": "原图 {left}%，{version} {right}%",
        "status.loadFailed": "无法加载对比列表（showcase.json）。",
        "status.loadFailedHttp": "无法加载对比列表（showcase.json）。（HTTP {status}）",
        "status.empty": "暂无对比图片。",
        "footer.notice": "示例图片是使用 GPT Image 2.6 生成的 AI 图片，仅用于演示修复前后的对比效果。这些图片不在本项目 AGPL 许可证的适用范围内，不允许二次使用。",
        "footer.repo": "GitHub 仓库",
        "item.handbag": "亮面皮革手提包",
        "item.poster": "英文告示海报",
        "item.cel-illustration": "平涂插画",
        "item.portrait": "虚化森林背景人像",
        "item.teapot": "深色背景下的哑光茶壶",
        "item.soup": "白盘中的南瓜汤",
    },
};

let current = FALLBACK;
const listeners = new Set();




/* =================================== 헬퍼/유틸 =================================== */


/** @description 언어 태그를 지원 언어 코드로 변환, 지원하지 않으면 null */
function _resolve(tag) {
    if (typeof tag !== "string") return null;
    if (tag in LOCALES) return tag;
    return PRIMARY_TAGS[tag.split("-")[0].toLowerCase()] ?? null;
}


/** @description 저장된 언어 읽기, 저장소 접근이 막힌 브라우저에서는 null */
function _readSaved() {
    try {
        return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
        console.error(e.name + "\n" + e.message + "\n" + e.stack);
        return null;
    }
}


/** @description 선택한 언어 저장, 저장소 접근이 막힌 브라우저에서는 이번 방문에만 적용 */
function _save(code) {
    try {
        localStorage.setItem(STORAGE_KEY, code);
    } catch (e) {
        console.error(e.name + "\n" + e.message + "\n" + e.stack);
    }
}


/** @description 언어 적용: html lang, 문서 제목, 정적 문구 갱신 후 변경 알림 */
function _apply(code) {
    current = code;
    document.documentElement.lang = code;
    document.title = t("page.title");
    translate(document);
    listeners.forEach((listener) => listener(code));
}




/* =================================== 메인 로직 =================================== */


/**
 * @description 현재 언어 문구 조회, 없으면 영어, 영어에도 없으면 키 그대로
 * @param {string} key 문구 키
 * @param {Object<string, string|number>} [params] {name} 자리에 넣을 값
 * @returns {string} 문구
 */
function t(key, params) {
    const template = MESSAGES[current][key] ?? MESSAGES[FALLBACK][key] ?? key;
    if (!params) return template;
    return template.replace(/\{(\w+)\}/g, (match, name) => name in params ? String(params[name]) : match);
}


/** @description 현재 언어나 영어에 해당 문구 키가 있는지 여부 */
function has(key) {
    return key in MESSAGES[current] || key in MESSAGES[FALLBACK];
}


/**
 * @description 표시 언어 결정: 저장된 선택 -> 브라우저 언어 목록 중 첫 지원 언어 -> 영어
 * @returns {string} 언어 코드
 */
function detectLocale() {
    const saved = _resolve(_readSaved());
    if (saved) return saved;
    const tags = navigator.languages?.length ? navigator.languages : [navigator.language];
    for (const tag of tags) {
        const code = _resolve(tag);
        if (code) return code;
    }
    return FALLBACK;
}


/**
 * @description root 아래 data-i18n(본문), data-i18n-label(aria-label), data-i18n-content(content) 요소에 현재 언어 적용
 * @param {ParentNode} root 적용 범위
 */
function translate(root) {
    for (const element of root.querySelectorAll("[data-i18n]")) element.textContent = t(element.dataset.i18n);
    for (const element of root.querySelectorAll("[data-i18n-label]")) element.setAttribute("aria-label", t(element.dataset.i18nLabel));
    for (const element of root.querySelectorAll("[data-i18n-content]")) element.setAttribute("content", t(element.dataset.i18nContent));
}


/**
 * @description 사용자가 고른 언어로 바꾸고 저장
 * @param {string} code 언어 코드
 */
function setLocale(code) {
    const resolved = _resolve(code) ?? FALLBACK;
    _save(resolved);
    _apply(resolved);
}




/* =================================== 이벤트/내보내기 =================================== */


_apply(detectLocale());

return Object.freeze({
    LOCALES,
    t,
    has,
    detectLocale,
    setLocale,
    translate,
    getLocale: () => current,
    onChange: (listener) => listeners.add(listener),
});

})();
