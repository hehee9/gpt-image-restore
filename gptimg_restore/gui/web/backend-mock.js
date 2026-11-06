/**
 * @file gptimg_restore/gui/web/backend-mock.js
 * @description 화면 개발용 가짜 백엔드: 단계별 가짜 진행률과 약간 변형된 결과 이미지 생성
 * @note 실제 복원 없음. 계약(backend.js)의 동작과 오류 형태만 재현
 */

import { downloadFiles } from "./download.js";




/* =================================== 전역 상수/변수 =================================== */


const STAGES = ["dots", "sparkle", "periodic", "grime", "mottle", "lattice", "surface", "contour", "defocus", "cel_fill", "dither"];
const PRESETS = {
    safe: { dots: 0, sparkle: 0, periodic: 1, grime: 0.5, mottle: 0.35, lattice: 0, surface: 0, contour: 0, defocus: 0, cel_fill: 0, dither: 0 },
    balanced: { dots: 1, sparkle: 0, periodic: 1, grime: 1, mottle: 0.7, lattice: 0, surface: 0, contour: 0, defocus: 0, cel_fill: 0, dither: 0 },
    strong: { dots: 1, sparkle: 1, periodic: 1, grime: 1.1, mottle: 1, lattice: 1, surface: 0, contour: 1, defocus: 1, cel_fill: 1, dither: 1 },
};
const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"];
const WEB_MAX_PIXELS = 16_777_216;
const THUMBNAIL_SHORT_SIDE = 160;
const MIME_TYPES = { png: "image/png", jpg: "image/jpeg", webp: "image/webp" };
const MOCK_FOLDER = "C:\\Users\\mock\\Pictures\\restored";
const TICK_MS = 40;




/* =================================== 헬퍼/유틸 =================================== */


/** @description 중단 신호를 따르는 대기 */
function _sleep(ms, signal) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        signal.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(signal.reason);
        }, { once: true });
    });
}

/** @description 확장자 제외 파일 이름 */
function _stem(name) {
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(0, dot) : name;
}

/** @description 짧은 변 기준 축소 이미지 주소 생성 */
async function _thumbnailUrl(bitmap) {
    const scale = Math.min(1, THUMBNAIL_SHORT_SIDE / Math.min(bitmap.width, bitmap.height));
    const canvas = new OffscreenCanvas(Math.round(bitmap.width * scale), Math.round(bitmap.height * scale));
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return URL.createObjectURL(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 }));
}

/** @description 강도 평균에 비례한 흐림으로 "복원" 흉내 */
function _renderRestored(bitmap, settings) {
    const values = Object.values(settings.strengths);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const blur = mean * (settings.dotsMode === "strict" ? 1.1 : 0.8);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    context.filter = `blur(${blur.toFixed(2)}px) saturate(${1 - mean * 0.12})`;
    context.drawImage(bitmap, 0, 0);
    return canvas.convertToBlob({ type: "image/png" });
}

/** @description 결과 PNG를 출력 형식으로 재인코딩 */
async function _encode(blob, output) {
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    bitmap.close();
    return canvas.convertToBlob({ type: MIME_TYPES[output.format], quality: output.quality / 100 });
}

/** @description 숨은 파일 입력으로 네이티브 대화상자 흉내 */
function _chooseFiles() {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = IMAGE_EXTENSIONS.join(",");
        input.addEventListener("change", () => resolve([...input.files]));
        input.addEventListener("cancel", () => resolve([]));
        input.click();
    });
}




/* =================================== 메인 로직 =================================== */


/**
 * 가짜 백엔드 생성
 * @param {"desktop" | "web"} kind 흉내 낼 환경
 * @returns {import("./backend.js").Backend}
 */
export function createMockBackend(kind) {
    const entries = new Map();
    const isWeb = kind === "web";

    /** @description 형식, 해독 가능 여부, 웹 화소 제한 검사 후 항목 등록 */
    async function addFiles(files) {
        const added = [];
        const rejected = [];
        for (const file of files) {
            const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
            if (!IMAGE_EXTENSIONS.includes(extension)) {
                rejected.push({ name: file.name, reason: "unreadable" });
                continue;
            }
            let bitmap;
            try {
                bitmap = await createImageBitmap(file);
            } catch (e) {
                console.warn(`${e.name}\n${e.message}\n${e.stack}`);
                rejected.push({ name: file.name, reason: "unreadable" });
                continue;
            }
            if (isWeb && bitmap.width * bitmap.height > WEB_MAX_PIXELS) {
                bitmap.close();
                rejected.push({ name: file.name, reason: "too_large" });
                continue;
            }
            const item = {
                id: crypto.randomUUID(),
                name: file.name,
                width: bitmap.width,
                height: bitmap.height,
                originalUrl: URL.createObjectURL(file),
                thumbnailUrl: await _thumbnailUrl(bitmap),
            };
            entries.set(item.id, { ...item, bitmap, restoredBlob: null, restoredUrl: null, controller: null });
            added.push(item);
        }
        return { added, rejected };
    }

    return {
        async init() {
            return {
                kind,
                stages: STAGES,
                presets: PRESETS,
                maxPixels: isWeb ? WEB_MAX_PIXELS : null,
                nativePicker: !isWeb,
                desktopAppUrl: isWeb ? "https://github.com/" : null,
            };
        },

        async pickFiles() {
            return addFiles(await _chooseFiles());
        },

        addFiles,

        async process(id, settings, onProgress) {
            const entry = entries.get(id);
            const controller = new AbortController();
            entry.controller = controller;
            const steps = ["load", ...STAGES.filter((stage) => settings.strengths[stage] > 0), "encode"];
            const stepMs = 160 + 50 * (entry.width * entry.height / 1e6);

            for (const [index, stage] of steps.entries()) {
                for (let elapsed = 0; elapsed < stepMs; elapsed += TICK_MS) {
                    onProgress({ fraction: (index + elapsed / stepMs) / steps.length, stage });
                    await _sleep(TICK_MS, controller.signal);
                }
            }
            const blob = await _renderRestored(entry.bitmap, settings);
            controller.signal.throwIfAborted();
            onProgress({ fraction: 1, stage: "encode" });

            if (entry.restoredUrl) {
                URL.revokeObjectURL(entry.restoredUrl);
            }
            entry.restoredBlob = blob;
            entry.restoredBitDepth = settings.bitDepth;
            entry.restoredUrl = URL.createObjectURL(blob);
            entry.controller = null;
            return { restoredUrl: entry.restoredUrl };
        },

        cancel(id) {
            entries.get(id).controller?.abort();
        },

        remove(id) {
            const entry = entries.get(id);
            entry.controller?.abort();
            entry.bitmap.close();
            URL.revokeObjectURL(entry.originalUrl);
            URL.revokeObjectURL(entry.thumbnailUrl);
            if (entry.restoredUrl) {
                URL.revokeObjectURL(entry.restoredUrl);
            }
            entries.delete(id);
        },

        async pickExportTarget() {
            if (isWeb) {
                return { kind: "download" };
            }
            await _sleep(250, new AbortController().signal);
            return { kind: "folder", path: MOCK_FOLDER };
        },

        async export(ids, output, target) {
            const bitDepth = output.format === "png" && output.bitDepth === 16 ? 16 : 8;
            const files = [];
            for (const id of ids) {
                const entry = entries.get(id);
                // 화면 보장 확인: 처리 비트 깊이와 출력 설정 일치
                if (entry.restoredBitDepth !== bitDepth) {
                    throw new Error(`Processing bit depth mismatch: ${entry.name} ${entry.restoredBitDepth} != ${bitDepth}`);
                }
                files.push({ name: `${_stem(entry.name)}_restored.${output.format}`, blob: await _encode(entry.restoredBlob, output) });
            }
            if (target.kind === "folder") {
                await _sleep(300, new AbortController().signal);
                return { count: files.length, location: target.path };
            }
            await downloadFiles(files);
            return { count: files.length, location: null };
        },

        async setLocale() {},
    };
}

