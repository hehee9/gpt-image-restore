/**
 * @file gptimg_restore/gui/web/app.js
 * @description 복원 작업 화면: 이미지 목록, 전후 비교 뷰어, 설정 패널, 일괄 처리 및 내보내기
 * @note 처리 순서: 선택한 이미지 우선, 내보내기 중이면 목록 순서대로 나머지 처리 후 저장
 */

import { createBackend } from "./backend.js";
import { LOCALES, detectLocale, getLocale, hasMessage, saveLocale, setLocale, t } from "./i18n.js";




/* =================================== 전역 상수/변수 =================================== */


const STRENGTH_MAX = 2;
const STRENGTH_STEP = 0.05;
const ZOOM_MAX = 32;
const ZOOM_STEP = 1.25;
const FIT_PADDING = 24;
const PIXELATED_FROM = 2;
const TOAST_MS = 6000;

const els = {
    body: document.body,
    viewer: document.getElementById("viewer"),
    imgBefore: document.getElementById("img-before"),
    imgAfter: document.getElementById("img-after"),
    layerAfter: document.getElementById("layer-after"),
    split: document.getElementById("split"),
    viewerProgress: document.getElementById("viewer-progress"),
    queueList: document.getElementById("queue-list"),
    toasts: document.getElementById("toasts"),
    stages: document.getElementById("stages"),
    stageList: document.getElementById("stage-list"),
    stageReset: document.getElementById("stage-reset"),
    bitDepthRow: document.getElementById("bit-depth-row"),
    qualityRow: document.getElementById("quality-row"),
    quality: document.getElementById("quality"),
    qualityValue: document.getElementById("quality-value"),
    exportButton: document.getElementById("export"),
    exportFill: document.getElementById("export-fill"),
    exportLabel: document.getElementById("export-label"),
    dropVeil: document.getElementById("drop-veil"),
    fileInput: document.getElementById("file-input"),
    localeSelect: document.getElementById("locale-select"),
};

const state = {
    backend: null,
    info: null,
    preset: "strong",
    overrides: new Map(),
    dotsMode: "conservative",
    output: { format: "png", bitDepth: 8, quality: 90 },
    items: [],
    selectedId: null,
    job: null,
    exportRun: null,
    saving: false,
};

const view = { scale: 1, x: 0, y: 0, fitted: true, split: 0.5 };
const pointers = new Map();
let gesture = null;
let dragDepth = 0;




/* =================================== 헬퍼/유틸 =================================== */


/** @description 프리셋 값에 개별 조정을 덮어쓴 단계별 강도 */
function _strengths() {
    const strengths = { ...state.info.presets[state.preset] };
    for (const [stage, value] of state.overrides) {
        strengths[stage] = value;
    }
    return strengths;
}


/** @description 처리 비트 깊이. PNG 16비트 출력일 때만 16 */
function _processBitDepth() {
    return state.output.format === "png" && state.output.bitDepth === 16 ? 16 : 8;
}


/** @description 결과 재사용 판단용 복원 설정 키(처리 비트 깊이 포함) */
function _settingsKey() {
    return JSON.stringify([_strengths(), state.dotsMode, _processBitDepth()]);
}


/** @description 현재 설정으로 처리가 필요한 항목인지 판정. 같은 설정에서 실패한 항목은 제외 */
function _needsWork(item, key) {
    return item.doneKey !== key && item.errorKey !== key;
}


/** @description 내보내기 진행 여부(일괄 처리와 저장 단계 모두) */
function _isExporting() {
    return state.exportRun !== null || state.saving;
}


/** @description 선택 항목 조회 */
function _selectedItem() {
    return state.items.find((item) => item.id === state.selectedId);
}


/** @description 범위 제한 */
function _clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}


/** @description 강도 표시 문자열. 0은 끔 */
function _formatStrength(value) {
    return value === 0 ? t("strength.off") : value.toFixed(2);
}


/** @description 파일 이름 목록 요약: "a.png" 또는 "a.png and 2 more" 형태 */
function _namesLabel(names) {
    return names.length === 1 ? names[0] : t("names.more", { first: names[0], count: names.length - 1 });
}


/** @description 단계 표시 이름. 사전에 없는 단계는 id 그대로 */
function _stageLabel(stage) {
    return hasMessage(`stage.${stage}`) ? t(`stage.${stage}`) : stage;
}


/** @description 진행 단계 표시 이름. 알 수 없는 단계는 일반 처리 문구 */
function _progressLabel(stage) {
    if (stage === "load" || stage === "encode") {
        return t(`progress.${stage}`);
    }
    return state.info.stages.includes(stage) ? _stageLabel(stage) : t("progress.working");
}


/** @description 범위 입력 채움 비율 갱신 */
function _paintRange(input) {
    const ratio = (input.valueAsNumber - Number(input.min)) / (Number(input.max) - Number(input.min));
    input.style.setProperty("--fill", `${ratio * 100}%`);
}


/**
 * 알림 표시
 * @param {string} message
 * @param {{error?: boolean, link?: {href: string, text: string}}} [options]
 */
function _toast(message, options = {}) {
    const toast = document.createElement("div");
    toast.className = options.error ? "toast is-error" : "toast";
    toast.textContent = message;
    if (options.link) {
        const anchor = document.createElement("a");
        anchor.href = options.link.href;
        anchor.target = "_blank";
        anchor.rel = "noopener";
        anchor.textContent = options.link.text;
        toast.append(anchor);
    }
    els.toasts.append(toast);
    setTimeout(() => toast.remove(), TOAST_MS);
}




/* =================================== 뷰어 =================================== */


/** @description 뷰어 크기 */
function _viewerSize() {
    return { width: els.viewer.clientWidth, height: els.viewer.clientHeight };
}


/** @description 여백을 둔 화면 맞춤 배율 */
function _fitScale(item) {
    const { width, height } = _viewerSize();
    return Math.min((width - FIT_PADDING * 2) / item.width, (height - FIT_PADDING * 2) / item.height);
}


/** @description 이미지가 화면 밖으로 완전히 벗어나지 않도록 이동량 제한 */
function _clampPan(item) {
    const { width, height } = _viewerSize();
    const w = item.width * view.scale;
    const h = item.height * view.scale;
    view.x = w <= width ? _clamp(view.x, 0, width - w) : _clamp(view.x, width - w, 0);
    view.y = h <= height ? _clamp(view.y, 0, height - h) : _clamp(view.y, height - h, 0);
}


/** @description 원본과 결과 이미지에 같은 변환 적용 */
function _applyTransform() {
    const transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
    els.imgBefore.style.transform = transform;
    els.imgAfter.style.transform = transform;
    els.viewer.classList.toggle("is-pixelated", view.scale >= PIXELATED_FROM);
}


/** @description 분할선 위치 반영 */
function _applySplit() {
    const percent = view.split * 100;
    els.viewer.style.setProperty("--split", `${percent}%`);
    els.split.setAttribute("aria-valuenow", String(Math.round(percent)));
}


/** @description 화면 맞춤 보기 */
function fitView() {
    const item = _selectedItem();
    const { width, height } = _viewerSize();
    view.scale = _fitScale(item);
    view.x = (width - item.width * view.scale) / 2;
    view.y = (height - item.height * view.scale) / 2;
    view.fitted = true;
    _applyTransform();
}


/**
 * 화면 좌표 (cx, cy)를 고정점으로 확대/축소
 * @param {number} scale 목표 배율
 * @param {number} cx 뷰어 기준 x
 * @param {number} cy 뷰어 기준 y
 */
function zoomTo(scale, cx, cy) {
    const item = _selectedItem();
    const next = _clamp(scale, Math.min(_fitScale(item), 1) / 2, ZOOM_MAX);
    view.x = cx - (cx - view.x) * (next / view.scale);
    view.y = cy - (cy - view.y) * (next / view.scale);
    view.scale = next;
    view.fitted = false;
    _clampPan(item);
    _applyTransform();
}


/** @description 뷰어 중앙 기준 확대/축소 */
function _zoomAtCenter(scale) {
    const { width, height } = _viewerSize();
    zoomTo(scale, width / 2, height / 2);
}


/** @description 선택 항목의 원본과 결과를 뷰어에 표시 */
function _showSelected() {
    const item = _selectedItem();
    els.body.classList.toggle("is-empty", !item);
    if (!item) {
        els.imgBefore.removeAttribute("src");
        els.imgAfter.removeAttribute("src");
        return;
    }
    for (const img of [els.imgBefore, els.imgAfter]) {
        img.style.width = `${item.width}px`;
        img.style.height = `${item.height}px`;
    }
    els.imgBefore.src = item.originalUrl;
    els.imgAfter.classList.remove("is-fresh");
    els.imgAfter.src = item.restoredUrl ?? item.originalUrl;
    els.layerAfter.classList.toggle("is-pending", item.restoredUrl === null);
    fitView();
    _renderViewerProgress();
}


/** @description 새 결과를 해독한 뒤 교체하고 현상 효과 적용 */
async function _revealRestored(item) {
    const probe = new Image();
    probe.src = item.restoredUrl;
    await probe.decode();
    if (item.id !== state.selectedId) {
        return;
    }
    const wasPending = els.layerAfter.classList.contains("is-pending");
    els.imgAfter.src = item.restoredUrl;
    els.layerAfter.classList.remove("is-pending");
    if (wasPending) {
        els.imgAfter.classList.remove("is-fresh");
        void els.imgAfter.offsetWidth;
        els.imgAfter.classList.add("is-fresh");
    }
}


/** @description 선택 항목 처리 진행선 갱신 */
function _renderViewerProgress() {
    const job = state.job;
    const active = job !== null && job.item.id === state.selectedId;
    els.viewerProgress.classList.toggle("is-active", active);
    els.viewerProgress.firstElementChild.style.transform = `scaleX(${active ? job.item.progress : 0})`;
}




/* =================================== 이미지 목록 =================================== */


/** @description 목록 항목 요소 생성 */
function _createThumb(item) {
    const li = document.createElement("li");
    li.className = "thumb";
    li.innerHTML = `
        <button type="button" class="thumb-select">
            <img alt="" decoding="async" draggable="false">
            <span class="thumb-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100"><span></span></span>
            <span class="thumb-error" aria-hidden="true">!</span>
        </button>
        <button type="button" class="thumb-remove" data-i18n-aria-label="queue.remove">
            <svg viewBox="0 0 20 20" width="12" height="12" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </button>`;
    const [select, remove] = li.querySelectorAll("button");
    select.title = item.name;
    select.setAttribute("aria-label", item.name);
    remove.setAttribute("aria-label", t("queue.remove"));
    li.querySelector("img").src = item.thumbnailUrl;
    select.addEventListener("click", () => selectItem(item.id));
    remove.disabled = _isExporting();
    remove.addEventListener("click", () => removeItem(item.id));
    li.addEventListener("keydown", (event) => {
        if (event.key === "Delete" && !_isExporting()) {
            removeItem(item.id);
        }
    });
    item.el = li;
    return li;
}


/** @description 목록 항목 상태 표시 갱신 */
function _renderItem(item, key) {
    const li = item.el;
    const running = state.job !== null && state.job.item === item;
    const waiting = state.exportRun !== null && _needsWork(item, key);
    li.classList.toggle("is-selected", item.id === state.selectedId);
    li.classList.toggle("is-error", item.errorKey === key);
    li.classList.toggle("is-busy", running || waiting);
    li.querySelector(".thumb-select").setAttribute("aria-current", String(item.id === state.selectedId));
    _renderItemProgress(item);
}


/** @description 목록 항목 진행 막대 갱신 */
function _renderItemProgress(item) {
    const running = state.job !== null && state.job.item === item;
    const fraction = running ? item.progress : 0;
    const bar = item.el.querySelector(".thumb-bar");
    bar.firstElementChild.style.transform = `scaleX(${fraction})`;
    bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
    bar.setAttribute("aria-valuetext", running ? _progressLabel(item.stage) : t("queue.waiting"));
}


/** @description 전체 목록 상태 갱신 */
function renderQueue() {
    const key = _settingsKey();
    for (const item of state.items) {
        _renderItem(item, key);
    }
}




/* =================================== 처리 일정 =================================== */


/** @description 다음 처리 대상 선택 및 실행. 대상이 없고 내보내기 중이면 저장. 저장 중에는 실행 안 함 */
function pump() {
    if (state.job || state.saving) {
        return;
    }
    const key = _settingsKey();
    const selected = _selectedItem();
    let next = selected && _needsWork(selected, key) ? selected : undefined;
    if (!next && state.exportRun) {
        next = state.items.find((item) => _needsWork(item, key));
    }
    if (next) {
        _runJob(next, key);
    } else if (state.exportRun) {
        _finishExport(key);
    }
}


/** @description 항목 복원 실행 및 결과 반영 */
function _runJob(item, key) {
    state.job = { item, key };
    item.progress = 0;
    item.stage = "load";
    _renderItem(item, key);
    _renderViewerProgress();

    const settings = { strengths: _strengths(), dotsMode: state.dotsMode, bitDepth: _processBitDepth() };
    const onProgress = (progress) => {
        item.progress = progress.fraction;
        item.stage = progress.stage;
        _renderItemProgress(item);
        if (item.id === state.selectedId) {
            _renderViewerProgress();
        }
        if (state.exportRun) {
            _renderExport();
        }
    };

    state.backend.process(item.id, settings, onProgress)
        .then((result) => {
            item.restoredUrl = result.restoredUrl;
            item.doneKey = key;
            if (item.id === state.selectedId) {
                _revealRestored(item);
            }
        })
        .catch((e) => {
            if (e.name === "AbortError") {
                return;
            }
            console.error(`${e.name}\n${e.message}\n${e.stack}`);
            item.errorKey = key;
            _toast(t("toast.processFailed", { name: item.name }), { error: true });
        })
        .finally(() => {
            state.job = null;
            if (item.el.isConnected) {
                _renderItem(item, _settingsKey());
            }
            _renderViewerProgress();
            _renderExport();
            pump();
        });
}


/** @description 설정 변경 반영: 이전 설정으로 진행 중인 처리 중단 후 재예약 */
function onRestoreSettingsChanged() {
    const key = _settingsKey();
    if (state.job && state.job.key !== key) {
        state.backend.cancel(state.job.item.id);
    }
    renderQueue();
    pump();
}




/* =================================== 목록 조작 =================================== */


/** @description 항목 선택. 내보내기 중이 아니면 다른 항목 처리 중단 */
function selectItem(id) {
    if (id === state.selectedId) {
        return;
    }
    if (state.job && !state.exportRun && state.job.item.id !== id) {
        state.backend.cancel(state.job.item.id);
    }
    state.selectedId = id;
    renderQueue();
    _showSelected();
    pump();
}


/** @description 항목 제거 및 이웃 항목 선택 */
function removeItem(id) {
    const index = state.items.findIndex((item) => item.id === id);
    const item = state.items[index];
    if (state.job && state.job.item === item) {
        state.backend.cancel(id);
    }
    state.backend.remove(id);
    state.items.splice(index, 1);
    item.el.remove();
    if (state.selectedId === id) {
        const neighbor = state.items[Math.min(index, state.items.length - 1)];
        state.selectedId = null;
        if (neighbor) {
            selectItem(neighbor.id);
        } else {
            _showSelected();
        }
    }
    _renderExport();
    pump();
}


/** @description 추가 결과 반영 및 거부 사유 알림 */
function _acceptAdded(result) {
    for (const source of result.added) {
        const item = { ...source, restoredUrl: null, doneKey: null, errorKey: null, progress: 0, stage: null, el: null };
        state.items.push(item);
        els.queueList.append(_createThumb(item));
    }
    _reportRejected(result.rejected);
    if (result.added.length > 0 && !_selectedItem()) {
        selectItem(result.added[0].id);
    } else {
        renderQueue();
    }
    _renderExport();
    pump();
}


/** @description 거부 사유별 알림 */
function _reportRejected(rejected) {
    const tooLarge = rejected.filter((entry) => entry.reason === "too_large").map((entry) => entry.name);
    const unreadable = rejected.filter((entry) => entry.reason === "unreadable").map((entry) => entry.name);
    if (tooLarge.length > 0) {
        const side = Math.sqrt(state.info.maxPixels);
        const link = state.info.desktopAppUrl ? { href: state.info.desktopAppUrl, text: t("toast.desktopLink") } : undefined;
        _toast(t("toast.tooLarge", { names: _namesLabel(tooLarge), side }), { error: true, link });
    }
    if (unreadable.length > 0) {
        _toast(t("toast.unreadable", { names: _namesLabel(unreadable), count: unreadable.length }), { error: true });
    }
}


/** @description File 목록 추가 */
async function addFiles(files) {
    if (files.length === 0) {
        return;
    }
    _acceptAdded(await state.backend.addFiles(files));
}


/** @description 환경별 파일 선택 열기 */
async function openFilePicker() {
    if (state.info.nativePicker) {
        _acceptAdded(await state.backend.pickFiles());
        return;
    }
    els.fileInput.click();
}




/* =================================== 내보내기 =================================== */


/** @description 내보내기 버튼 상태와 일괄 진행률 갱신 */
function _renderExport() {
    const button = els.exportButton;
    const running = state.exportRun !== null;
    button.classList.toggle("is-running", running);
    button.disabled = state.saving || (!running && state.items.length === 0);

    if (state.saving) {
        els.exportLabel.textContent = t("export.saving");
    } else {
        els.exportLabel.textContent = t(running ? "export.stop" : "export.idle");
    }

    if (running || state.saving) {
        const key = _settingsKey();
        const settled = state.items.filter((item) => !_needsWork(item, key)).length;
        const partial = state.job ? state.job.item.progress : 0;
        const ratio = state.items.length === 0 ? 1 : (settled + partial) / state.items.length;
        els.exportFill.style.transform = `scaleX(${state.saving ? 1 : ratio})`;
    }
}


/** @description 저장 위치 확인 후 일괄 처리 시작 */
async function startExport() {
    const target = await state.backend.pickExportTarget();
    if (!target) {
        return;
    }
    state.exportRun = { target };
    renderQueue();
    _renderExport();
    _renderLock();
    pump();
}


/** @description 일괄 처리 중지. 선택 항목 처리는 유지 */
function stopExport() {
    state.exportRun = null;
    if (state.job && state.job.item.id !== state.selectedId) {
        state.backend.cancel(state.job.item.id);
    }
    renderQueue();
    _renderExport();
    _renderLock();
}


/** @description 현재 설정 결과 저장 및 알림. 끝나면 잠금 해제 후 일반 처리 재개 */
async function _finishExport(key) {
    const { target } = state.exportRun;
    state.exportRun = null;
    const ids = state.items.filter((item) => item.doneKey === key).map((item) => item.id);
    renderQueue();
    if (ids.length === 0) {
        _toast(t("toast.nothingToExport"), { error: true });
    } else {
        state.saving = true;
        _renderExport();
        try {
            const result = await state.backend.export(ids, { ...state.output }, target);
            if (result.location !== null) {
                _toast(t("toast.saved", { count: result.count }));
            }
        } catch (e) {
            console.error(`${e.name}\n${e.message}\n${e.stack}`);
            _toast(t("toast.exportFailed"), { error: true });
        }
        state.saving = false;
    }
    _renderExport();
    _renderLock();
    pump();
}


/** @description 내보내기 동안 복원 설정과 목록 추가, 제거 잠금 */
function _renderLock() {
    const locked = _isExporting();
    for (const control of document.querySelectorAll('.panel-body input, .panel-body button, [data-action="add"], .thumb-remove')) {
        control.disabled = locked;
    }
}




/* =================================== 설정 패널 =================================== */


/** @description 단계별 강도 슬라이더 생성 */
function _buildStageSliders() {
    for (const stage of state.info.stages) {
        const row = document.createElement("div");
        row.className = "slider-row";
        row.dataset.stage = stage;
        row.innerHTML = `
            <label for="stage-${stage}"></label>
            <output for="stage-${stage}"></output>
            <input type="range" id="stage-${stage}" min="0" max="${STRENGTH_MAX}" step="${STRENGTH_STEP}">`;
        els.stageList.append(row);
    }
}


/** @description 설정 패널 표시 갱신 */
function renderSettings() {
    const strengths = _strengths();
    for (const row of els.stageList.children) {
        const stage = row.dataset.stage;
        const input = row.querySelector("input");
        row.querySelector("label").textContent = _stageLabel(stage);
        input.value = String(strengths[stage]);
        _paintRange(input);
        row.querySelector("output").textContent = _formatStrength(strengths[stage]);
        row.classList.toggle("is-override", state.overrides.has(stage));
    }
    els.stages.classList.toggle("has-override", state.overrides.size > 0);
    els.stageReset.hidden = state.overrides.size === 0;
    document.querySelector(`input[name="preset"][value="${state.preset}"]`).checked = true;

    const isPng = state.output.format === "png";
    els.bitDepthRow.hidden = !isPng;
    els.qualityRow.hidden = isPng;
    els.qualityValue.textContent = String(state.output.quality);
    _paintRange(els.quality);
}


/** @description 단계 슬라이더 값 반영. 프리셋 값과 같으면 개별 조정 해제 */
function _setStageValue(stage, value) {
    if (value === state.info.presets[state.preset][stage]) {
        state.overrides.delete(stage);
    } else {
        state.overrides.set(stage, value);
    }
    renderSettings();
}


/** @description 프리셋 적용 및 개별 조정 해제 */
function _applyPreset(preset) {
    state.preset = preset;
    state.overrides.clear();
    renderSettings();
    onRestoreSettingsChanged();
}




/* =================================== 이벤트/내보내기 =================================== */


/** @description 설정 패널 이벤트 연결 */
function _bindSettings() {
    for (const input of document.querySelectorAll('input[name="preset"]')) {
        // 이미 선택된 프리셋을 다시 누르면 개별 조정 해제
        input.addEventListener("click", () => _applyPreset(input.value));
    }
    for (const input of document.querySelectorAll('input[name="dots-mode"]')) {
        input.addEventListener("change", () => {
            state.dotsMode = input.value;
            onRestoreSettingsChanged();
        });
    }
    els.stageList.addEventListener("input", (event) => {
        _setStageValue(event.target.closest(".slider-row").dataset.stage, event.target.valueAsNumber);
    });
    els.stageList.addEventListener("change", onRestoreSettingsChanged);
    els.stageList.addEventListener("dblclick", (event) => {
        const row = event.target.closest(".slider-row");
        // 비활성 슬라이더 옆 라벨에서도 발생하므로 잠금 확인
        if (row && !_isExporting() && state.overrides.delete(row.dataset.stage)) {
            renderSettings();
            onRestoreSettingsChanged();
        }
    });
    els.stageReset.addEventListener("click", () => _applyPreset(state.preset));

    for (const input of document.querySelectorAll('input[name="format"]')) {
        input.addEventListener("change", () => {
            state.output.format = input.value;
            renderSettings();
            onRestoreSettingsChanged();
        });
    }
    for (const input of document.querySelectorAll('input[name="bit-depth"]')) {
        input.addEventListener("change", () => {
            state.output.bitDepth = Number(input.value);
            onRestoreSettingsChanged();
        });
    }
    els.quality.addEventListener("input", () => {
        state.output.quality = els.quality.valueAsNumber;
        renderSettings();
    });

    els.exportButton.addEventListener("click", () => {
        if (state.exportRun) {
            stopExport();
        } else {
            startExport();
        }
    });
}


/** @description 뷰어 확대, 이동, 분할선 이벤트 연결 */
function _bindViewer() {
    const viewer = els.viewer;

    viewer.addEventListener("pointerdown", (event) => {
        if (!_selectedItem() || event.target.closest(".zoom") || event.button !== 0) {
            return;
        }
        viewer.setPointerCapture(event.pointerId);
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (pointers.size === 1 && event.target.closest(".split")) {
            gesture = { kind: "split" };
            els.split.classList.add("is-active");
            return;
        }
        _beginGesture();
    });

    viewer.addEventListener("pointermove", (event) => {
        const previous = pointers.get(event.pointerId);
        if (!previous) {
            return;
        }
        const rect = viewer.getBoundingClientRect();
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

        if (gesture.kind === "split") {
            view.split = _clamp((event.clientX - rect.left) / rect.width, 0, 1);
            _applySplit();
        } else if (gesture.kind === "pan") {
            view.x += event.clientX - previous.x;
            view.y += event.clientY - previous.y;
            view.fitted = false;
            _clampPan(_selectedItem());
            _applyTransform();
        } else {
            const [a, b] = [...pointers.values()];
            const distance = Math.hypot(a.x - b.x, a.y - b.y);
            const midX = (a.x + b.x) / 2 - rect.left;
            const midY = (a.y + b.y) / 2 - rect.top;
            view.x += midX - gesture.midX;
            view.y += midY - gesture.midY;
            zoomTo(view.scale * (distance / gesture.distance), midX, midY);
            Object.assign(gesture, { distance, midX, midY });
        }
    });

    const release = (event) => {
        if (!pointers.delete(event.pointerId)) {
            return;
        }
        if (pointers.size > 0) {
            _beginGesture();
            return;
        }
        gesture = null;
        viewer.classList.remove("is-panning");
        els.split.classList.remove("is-active");
    };
    viewer.addEventListener("pointerup", release);
    viewer.addEventListener("pointercancel", release);

    viewer.addEventListener("wheel", (event) => {
        if (!_selectedItem()) {
            return;
        }
        event.preventDefault();
        const rect = viewer.getBoundingClientRect();
        const delta = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? event.deltaY * 16 : event.deltaY;
        zoomTo(view.scale * Math.exp(-delta * 0.0015), event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    viewer.addEventListener("dblclick", (event) => {
        const item = _selectedItem();
        if (!item || event.target.closest(".zoom, .split")) {
            return;
        }
        if (!view.fitted) {
            fitView();
            return;
        }
        const rect = viewer.getBoundingClientRect();
        const fit = _fitScale(item);
        zoomTo(fit >= 1 ? fit * 2 : 1, event.clientX - rect.left, event.clientY - rect.top);
    });

    viewer.querySelector(".zoom").addEventListener("click", (event) => {
        const action = event.target.closest("button")?.dataset.zoom;
        if (action === "fit") {
            fitView();
        } else if (action === "actual") {
            _zoomAtCenter(1);
        } else if (action === "in") {
            _zoomAtCenter(view.scale * ZOOM_STEP);
        } else if (action === "out") {
            _zoomAtCenter(view.scale / ZOOM_STEP);
        }
    });

    els.split.addEventListener("keydown", (event) => {
        const step = event.shiftKey ? 0.1 : 0.02;
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault();
            view.split = _clamp(view.split + (event.key === "ArrowLeft" ? -step : step), 0, 1);
            _applySplit();
        }
    });

    new ResizeObserver(() => {
        const item = _selectedItem();
        if (!item) {
            return;
        }
        if (view.fitted) {
            fitView();
        } else {
            _clampPan(item);
            _applyTransform();
        }
    }).observe(viewer);
}


/** @description 현재 누른 포인터 수에 맞춰 이동 또는 핀치 시작 */
function _beginGesture() {
    const rect = els.viewer.getBoundingClientRect();
    const points = [...pointers.values()];
    if (points.length === 1) {
        gesture = { kind: "pan" };
        els.viewer.classList.add("is-panning");
        return;
    }
    const [a, b] = points;
    gesture = {
        kind: "pinch",
        distance: Math.hypot(a.x - b.x, a.y - b.y),
        midX: (a.x + b.x) / 2 - rect.left,
        midY: (a.y + b.y) / 2 - rect.top,
    };
}


/** @description 파일 선택과 끌어다 놓기 연결 */
function _bindFiles() {
    for (const button of document.querySelectorAll('[data-action="add"]')) {
        button.addEventListener("click", openFilePicker);
    }
    els.fileInput.addEventListener("change", () => {
        addFiles([...els.fileInput.files]);
        els.fileInput.value = "";
    });

    const hasFiles = (event) => event.dataTransfer.types.includes("Files");
    window.addEventListener("dragenter", (event) => {
        if (!hasFiles(event)) {
            return;
        }
        dragDepth++;
        els.dropVeil.classList.toggle("is-active", !_isExporting());
    });
    window.addEventListener("dragover", (event) => {
        if (hasFiles(event)) {
            event.preventDefault();
            event.dataTransfer.dropEffect = _isExporting() ? "none" : "copy";
        }
    });
    window.addEventListener("dragleave", (event) => {
        if (hasFiles(event) && --dragDepth === 0) {
            els.dropVeil.classList.remove("is-active");
        }
    });
    window.addEventListener("drop", (event) => {
        if (!hasFiles(event)) {
            return;
        }
        event.preventDefault();
        dragDepth = 0;
        els.dropVeil.classList.remove("is-active");
        if (!_isExporting()) {
            addFiles([...event.dataTransfer.files]);
        }
    });
}


/** @description 언어 선택 목록 생성. 각 항목은 해당 언어 고유 이름 */
function _buildLocaleMenu() {
    for (const locale of LOCALES) {
        const option = new Option(locale.name, locale.code);
        option.lang = locale.code;
        els.localeSelect.append(option);
    }
    els.localeSelect.value = getLocale();
}


/** @description 언어 전환과 저장, 동적 문구 재표시, 백엔드 전달 */
function _changeLocale(code) {
    setLocale(code);
    saveLocale(code);
    renderSettings();
    renderQueue();
    _renderExport();
    state.backend.setLocale?.(code);
}


setLocale(detectLocale());
_buildLocaleMenu();
state.backend = await createBackend();
state.info = await state.backend.init();
state.backend.setLocale?.(getLocale());
_buildStageSliders();
_bindSettings();
_bindViewer();
_bindFiles();
els.localeSelect.addEventListener("change", () => _changeLocale(els.localeSelect.value));
renderSettings();
_applySplit();
_renderExport();


