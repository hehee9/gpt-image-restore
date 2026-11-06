/**
 * @file docs/site.js
 * @description showcase.json 기반 전후 비교 항목 렌더링, 슬라이더, 확대/이동, 대비 강조, 표시 언어 전환 (i18n.js의 I18n 사용)
 */




/* =================================== 전역 상수/변수 =================================== */


const MANIFEST_URL = "showcase.json";
const KEY_STEP = 2;
const KEY_STEP_LARGE = 10;
const STAGE_MAX_VIEWPORT = 0.78;
const STAGE_MIN_HEIGHT = 160;
const TOUCH_SLOP = 6;

const VERSIONS = ["original", "balanced", "strong"];
const DEFAULT_RIGHT = "strong";

const MAX_SCALE = 16;
const DEVICE_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64];
const SHRINK_STEP = 1.3;
const WHEEL_NOTCH = 50;
const WHEEL_LATCH_MS = 250;
const PINCH_WHEEL_SPEED = 0.01;
const PINCH_WHEEL_LIMIT = 50;
const SETTLE_MS = 160;
const BOOST_MAX_GAIN = 4;
const BOOST_MIN_GAIN = 0.3;
const BOOST_SMOOTH_SIGMA = 3;
const BOOST_TAIL_PERCENT = 0.2;

/**
 * 항목별 추가 강조 구간: 원본 전체에서 비중이 작아 기울기가 낮게 잡히는 주요 영역의 V 범위와 최소 기울기
 * cel-illustration: 머리카락 청록색 중심 밝기 (기본 보기 영역 청록 화소의 V 25~90 백분위)
 */
const BOOST_BANDS = {
    "cel-illustration": [{ from: 160, to: 178, gain: 4 }],
};

/** 페이지 스크롤로 흘려보낸 마지막 휠 이벤트 시각 */
let pageWheelAt = -Infinity;

/** 언어 변경 시 다시 그릴 문구 갱신 함수 목록 */
const relabelers = [];

/** 표시 중인 상태 문구의 키와 값, 언어 변경 시 다시 그림 */
let statusMessage = null;




/* =================================== 헬퍼/유틸 =================================== */


/** @description 결과 키의 현재 언어 이름 */
function _versionLabel(key) {
    return I18n.t("version." + key);
}


/** @description 항목 제목, 번역이 없으면 showcase.json 제목 */
function _itemTitle(item) {
    const key = "item." + item.id;
    return I18n.has(key) ? I18n.t(key) : item.title;
}


/** @description 현재 상태 문구를 현재 언어로 표시 */
function _renderStatus() {
    if (!statusMessage) return;
    const status = document.getElementById("status");
    status.textContent = I18n.t(statusMessage.key, statusMessage.params);
    status.hidden = false;
}


/** @description 목록 대신 상태 문구 표시 */
function _showStatus(key, params) {
    statusMessage = { key, params };
    _renderStatus();
}


/** @description 주소 해시의 항목으로 즉시 이동 */
function _jumpToHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id && document.getElementById(id);
    if (target) target.scrollIntoView({ block: "start", behavior: "instant" });
}


/** @description 색인 링크 생성 */
function _createIndexLink(item) {
    const li = document.createElement("li");
    const link = document.createElement("a");
    const image = document.createElement("img");
    const label = document.createElement("span");
    link.href = "#" + item.id;
    image.src = item.preview;
    image.width = 136;
    image.height = 67;
    image.alt = "";
    image.decoding = "async";
    const relabel = () => { label.textContent = _itemTitle(item); };
    relabel();
    relabelers.push(relabel);
    link.append(image, label);
    li.append(link);
    return li;
}


/** @description 언어 메뉴 구성, 고른 언어 저장과 화면 문구 갱신 */
function _initLocaleMenu() {
    const menu = document.getElementById("locale");
    for (const [code, name] of Object.entries(I18n.LOCALES)) {
        const option = new Option(name, code);
        option.lang = code;
        menu.append(option);
    }
    menu.value = I18n.getLocale();
    menu.addEventListener("change", () => I18n.setLocale(menu.value));
    I18n.onChange((code) => {
        menu.value = code;
        relabelers.forEach((relabel) => relabel());
        _renderStatus();
    });
}


/** @description 화면 배율(devicePixelRatio) 변경 감시, 모니터 이동 시 배율 단계 재계산용 */
function _watchPixelRatio(callback) {
    const query = matchMedia("(resolution: " + window.devicePixelRatio + "dppx)");
    query.addEventListener("change", () => {
        callback();
        _watchPixelRatio(callback);
    }, { once: true });
}


/**
 * @description 배율 단계 목록, 확대 구간은 이미지 1px이 화면 픽셀 정수 개에 대응하도록 구성
 * @param {number} base 화면 맞춤 배율 (CSS px / 이미지 px)
 * @returns {number[]} 오름차순 배율 단계
 */
function _zoomSteps(base) {
    const ratio = window.devicePixelRatio;
    const unit = 1 / ratio;
    const steps = [base];
    if (base < unit) {
        const count = Math.ceil(Math.log(unit / base) / Math.log(SHRINK_STEP));
        for (let i = 1; i < count; i++) steps.push(base * Math.pow(unit / base, i / count));
    }
    for (const multiple of DEVICE_STEPS) {
        const scale = multiple / ratio;
        if (scale > base * 1.05 && scale <= MAX_SCALE) steps.push(scale);
    }
    return steps;
}


/** @description 로그 거리 기준 가장 가까운 배율 단계 */
function _nearestStep(steps, scale) {
    return steps.reduce((best, step) =>
        Math.abs(Math.log(step / scale)) < Math.abs(Math.log(best / scale)) ? step : best);
}


/** @description 두 손가락 중심점과 거리 */
function _pinchOf(pointers) {
    const [a, b] = pointers.values();
    return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
    };
}


/**
 * @description 이미지를 읽어 화소별 V = max(R, G, B) 계산
 * @returns {Promise<{width: number, height: number, values: Uint8Array}>} V 값
 */
async function _loadValues(url) {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const values = new Uint8Array(canvas.width * canvas.height);
    for (let i = 0, j = 0; j < values.length; i += 4, j++) values[j] = Math.max(data[i], data[i + 1], data[i + 2]);
    return { width: canvas.width, height: canvas.height, values };
}


/**
 * @description 원본 전체의 V 분포로 항목별 대비 강조 곡선 계산
 * 화소가 많이 모인 밝기일수록 기울기를 키워 주된 밝기 영역의 미세 노이즈를 드러내고,
 * 기울기를 최대/최소로 제한해 나머지 밝기가 날아가거나 뭉개지지 않게 함
 * 검은점과 흰점은 양 끝 BOOST_TAIL_PERCENT 백분위
 * @param {Uint8Array} values 원본 V 값
 * @param {{from: number, to: number, gain: number}[]} [bands] 최소 기울기를 높일 V 구간
 * @returns {Uint8ClampedArray} V 값별 출력 밝기
 */
function _boostCurve(values, bands = []) {
    const histogram = new Float64Array(256);
    for (const value of values) histogram[value]++;
    const tail = values.length * BOOST_TAIL_PERCENT / 100;
    let low = 0;
    let high = 255;
    for (let value = 0, seen = 0; value < 256; value++) {
        seen += histogram[value];
        if (seen > tail) {
            low = value;
            break;
        }
    }
    for (let value = 255, seen = 0; value >= 0; value--) {
        seen += histogram[value];
        if (seen > tail) {
            high = value;
            break;
        }
    }

    const radius = Math.ceil(BOOST_SMOOTH_SIGMA * 3);
    const density = new Float64Array(256);
    for (let value = 0; value < 256; value++) {
        for (let offset = -radius; offset <= radius; offset++) {
            const source = Math.min(255, Math.max(0, value + offset));
            density[value] += histogram[source] * Math.exp(-offset * offset / (2 * BOOST_SMOOTH_SIGMA * BOOST_SMOOTH_SIGMA));
        }
    }

    const floors = new Float64Array(256).fill(BOOST_MIN_GAIN);
    for (const band of bands) {
        for (let value = band.from; value <= band.to; value++) floors[value] = Math.max(floors[value], band.gain);
    }

    const slopes = new Float64Array(256);
    const fillSlopes = (scale) => {
        let total = 0;
        for (let value = low; value <= high; value++) {
            slopes[value] = Math.min(BOOST_MAX_GAIN, Math.max(floors[value], scale * density[value]));
            total += slopes[value];
        }
        return total;
    };
    // 기울기 합이 출력 범위(255)와 같아지는 배율을 이분 탐색, 범위가 좁아 최대 기울기로도 모자라면 가운데 정렬
    let lower = 0;
    let upper = 1;
    while (fillSlopes(upper) < 255 && upper < 1e12) upper *= 2;
    for (let i = 0; i < 60; i++) {
        const middle = (lower + upper) / 2;
        if (fillSlopes(middle) < 255) lower = middle;
        else upper = middle;
    }
    const total = fillSlopes(upper);
    const shift = Math.max(0, (255 - total) / 2);
    const curve = new Uint8ClampedArray(256);
    for (let value = 0, sum = 0; value < 256; value++) {
        curve[value] = Math.round(shift + sum + slopes[value] / 2);
        sum += slopes[value];
    }
    return curve;
}


/** @description V 값에 곡선을 적용해 회색조로 캔버스에 그리기 */
function _paintBoosted(source, curve, canvas) {
    canvas.width = source.width;
    canvas.height = source.height;
    const context = canvas.getContext("2d");
    const output = context.createImageData(source.width, source.height);
    const data = output.data;
    for (let i = 0, j = 0; j < source.values.length; i += 4, j++) {
        const value = curve[source.values[j]];
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
        data[i + 3] = 255;
    }
    context.putImageData(output, 0, 0);
}




/* =================================== 메인 로직 =================================== */


/**
 * @description 항목 하나의 비교 뷰어 구성
 * @param {object} item showcase.json 항목
 * @param {HTMLTemplateElement} template 항목 템플릿
 * @returns {{section: HTMLElement, wrap: HTMLElement, fit: Function}} 섹션과 크기 맞춤 함수
 */
function createItem(item, template) {
    const section = template.content.firstElementChild.cloneNode(true);
    const stage = section.querySelector(".stage");
    const wrap = section.querySelector(".stage-wrap");
    const planes = stage.querySelectorAll(".plane");
    const before = stage.querySelector("img.before");
    const after = stage.querySelector("img.after");
    const boostBefore = stage.querySelector("canvas.before");
    const boostAfter = stage.querySelector("canvas.after");
    const tagAfter = stage.querySelector(".tag-after");
    const divider = stage.querySelector(".divider");
    const boost = section.querySelector(".boost");
    const state = { right: null, pos: 50, boost: false };
    const view = { base: 0, steps: [], scale: 0, x: 0, y: 0, width: 0, height: 0, wheelAt: -Infinity, wheelRest: 0, touched: false };
    const slider = { pointer: null, dragging: false, startX: 0, startY: 0 };
    const gesture = { pointers: new Map(), pinch: null, settleTimer: 0 };
    const pixels = { wanted: DEFAULT_RIGHT, original: null, curve: null, painted: null, token: 0 };

    section.id = item.id;
    const link = section.querySelector(".item-link");
    link.href = "#" + item.id;
    before.src = item.images.original;
    for (const image of [before, after]) {
        image.width = item.width;
        image.height = item.height;
    }
    I18n.translate(section);

    const versionButtons = VERSIONS.map((key) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "view";
        button.addEventListener("click", () => selectRight(key));
        section.querySelector(".versions").append(button);
        return button;
    });

    /** @description 화면 맞춤보다 확대된 상태 여부 */
    function isZoomed() {
        return view.scale > view.base * 1.001;
    }

    /** @description 배율, 위치를 화면에 반영, 양쪽 면은 같은 크기와 위치를 공유 */
    function render() {
        const ratio = window.devicePixelRatio;
        const width = item.width * view.scale + "px";
        const height = item.height * view.scale + "px";
        const offset = Math.round(view.x * ratio) / ratio + "px " + Math.round(view.y * ratio) / ratio + "px";
        for (const plane of planes) {
            plane.style.width = width;
            plane.style.height = height;
            plane.style.translate = offset;
        }
        const zoomed = isZoomed();
        stage.classList.toggle("zoomed", zoomed);
        stage.classList.toggle("pixel", zoomed && view.scale * ratio > 1);
    }

    /** @description 배율, 위치 지정 후 화면 밖으로 벗어나지 않게 제한 */
    function setView(scale, x, y) {
        view.scale = Math.min(MAX_SCALE, Math.max(view.base, scale));
        view.x = Math.min(0, Math.max(view.width - item.width * view.scale, x));
        view.y = Math.min(0, Math.max(view.height - item.height * view.scale, y));
        render();
    }

    /** @description 무대 좌표 (fx, fy) 아래의 이미지 지점을 고정한 채 배율 변경, dx/dy만큼 함께 이동 */
    function zoomAt(fx, fy, scale, dx = 0, dy = 0) {
        view.touched = true;
        const next = Math.min(MAX_SCALE, Math.max(view.base, scale));
        const imageX = (fx - view.x) / view.scale;
        const imageY = (fy - view.y) / view.scale;
        setView(next, fx + dx - imageX * next, fy + dy - imageY * next);
    }

    /** @description 배율 단계 한 칸 이동 */
    function stepZoom(direction, fx, fy) {
        const target = direction > 0
            ? view.steps.find((step) => step > view.scale * 1.001)
            : view.steps.findLast((step) => step < view.scale * 0.999);
        if (target !== undefined) zoomAt(fx, fy, target);
    }

    /** @description 연속 확대가 끝나면 가장 가까운 정수 배율 단계로 맞춤 */
    function settle(fx, fy) {
        clearTimeout(gesture.settleTimer);
        zoomAt(fx, fy, _nearestStep(view.steps, view.scale));
    }

    /** @description 페이지 좌표를 무대 기준 좌표로 변환 */
    function toStage(clientX, clientY) {
        const rect = stage.getBoundingClientRect();
        return { x: clientX - rect.left, y: clientY - rect.top };
    }

    /**
     * @description 기본 확대 영역(item.view)을 화면 중앙에 표시
     * 영역 전체가 들어가는 배율 중 화면 픽셀 정수 배수인 가장 큰 값을 써서 픽셀 그대로 보이게 함
     */
    function showDefaultView() {
        const [x, y, width, height] = item.view;
        const ratio = window.devicePixelRatio;
        const fill = Math.min(view.width / width, view.height / height);
        const scale = fill * ratio > 1 ? Math.floor(fill * ratio + 1e-6) / ratio : fill;
        setView(scale, view.width / 2 - (x + width / 2) * scale, view.height / 2 - (y + height / 2) * scale);
    }

    /** @description 비교 화면 크기 계산, 이미지 전체가 들어가는 크기를 맞춤 배율로 사용, 사용자가 확대/이동하기 전에는 기본 확대 영역 표시 */
    function fit() {
        const style = getComputedStyle(wrap);
        const availWidth = wrap.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const availHeight = Math.max(STAGE_MIN_HEIGHT, window.innerHeight * STAGE_MAX_VIEWPORT);
        const zoomed = isZoomed();
        const centerX = (view.width / 2 - view.x) / view.scale;
        const centerY = (view.height / 2 - view.y) / view.scale;
        view.base = Math.min(1, availWidth / item.width, availHeight / item.height);
        view.steps = _zoomSteps(view.base);
        view.width = item.width * view.base;
        view.height = item.height * view.base;
        stage.style.width = view.width + "px";
        stage.style.height = view.height + "px";
        if (!view.touched && item.view) {
            showDefaultView();
        } else if (zoomed) {
            const scale = Math.max(view.base, view.scale);
            setView(scale, view.width / 2 - centerX * scale, view.height / 2 - centerY * scale);
        } else {
            setView(view.base, 0, 0);
        }
    }

    /** @description 오른쪽 결과에 딸린 대체 텍스트, 이름표, 경계 설명을 현재 언어로 갱신 */
    function labelRight() {
        const label = _versionLabel(state.right);
        after.alt = I18n.t("image.alt", { title: _itemTitle(item), version: label });
        tagAfter.textContent = label;
        divider.setAttribute("aria-label", I18n.t("divider.label", { version: label }));
        setPos(state.pos);
    }

    /** @description 항목 문구 전체를 현재 언어로 갱신, 보기 상태는 유지 */
    function relabel() {
        const title = _itemTitle(item);
        link.textContent = title;
        before.alt = I18n.t("image.alt", { title, version: _versionLabel("original") });
        versionButtons.forEach((button, i) => { button.textContent = _versionLabel(VERSIONS[i]); });
        if (state.right) labelRight();
    }

    /** @description 오른쪽 면 결과를 버튼, 이미지, 이름표에 한꺼번에 반영 */
    function commitRight(key) {
        state.right = key;
        versionButtons.forEach((button, i) => button.setAttribute("aria-pressed", String(VERSIONS[i] === key)));
        after.src = item.images[key];
        labelRight();
    }

    /**
     * @description 대비 강조 없이 오른쪽 결과 표시, 새 이미지 디코딩이 끝난 뒤 반영해 이전 이미지가 새 이름표 아래 보이지 않게 함
     * @param {string} key 표시할 결과 키
     */
    async function showPlain(key) {
        const token = ++pixels.token;
        const image = new Image();
        image.src = item.images[key];
        try {
            await image.decode();
        } catch (e) {
            console.error(e.name + "\n" + e.message + "\n" + e.stack);
            return;
        }
        if (token === pixels.token) commitRight(key);
    }

    /** @description 오른쪽 면에 표시할 결과 선택 */
    function selectRight(key) {
        pixels.wanted = key;
        if (state.boost) refreshBoost();
        else showPlain(key);
    }

    /**
     * @description 대비 강조 이미지 준비, 원본 전체에서 항목마다 한 번 구한 곡선을 모든 결과에 같게 적용해 이미지마다 한 번만 그림
     * 오른쪽 캔버스와 선택 결과는 같은 시점에 반영해 이전 결과가 새 이름표 아래 보이지 않게 함
     */
    async function refreshBoost() {
        const token = ++pixels.token;
        const right = pixels.wanted;
        let original = pixels.original;
        let rightValues = null;
        try {
            if (!original) original = await _loadValues(item.images.original);
            if (pixels.painted !== right) rightValues = right === "original" ? original : await _loadValues(item.images[right]);
        } catch (e) {
            console.error(e.name + "\n" + e.message + "\n" + e.stack);
            if (token === pixels.token) setBoost(false);
            return;
        }
        if (token !== pixels.token) return;
        if (!pixels.original) {
            pixels.original = original;
            pixels.curve = _boostCurve(original.values, BOOST_BANDS[item.id]);
            _paintBoosted(original, pixels.curve, boostBefore);
        }
        if (rightValues) {
            _paintBoosted(rightValues, pixels.curve, boostAfter);
            pixels.painted = right;
        }
        stage.classList.add("boosted");
        if (right !== state.right) commitRight(right);
    }

    /** @description 대비 강조 켜기/끄기 */
    function setBoost(on) {
        state.boost = on;
        boost.setAttribute("aria-pressed", String(on));
        if (on) {
            refreshBoost();
            return;
        }
        pixels.token++;
        stage.classList.remove("boosted");
        if (pixels.wanted !== state.right) showPlain(pixels.wanted);
    }

    /** @description 경계 위치 지정 (0~100) */
    function setPos(value) {
        state.pos = Math.min(100, Math.max(0, value));
        const rounded = Math.round(state.pos);
        stage.style.setProperty("--pos", state.pos + "%");
        divider.setAttribute("aria-valuenow", String(rounded));
        divider.setAttribute("aria-valuetext", I18n.t("divider.valuetext", { left: rounded, version: _versionLabel(state.right), right: 100 - rounded }));
    }

    /** @description 포인터 좌표로 경계 이동 */
    function moveToPointer(event) {
        const rect = stage.getBoundingClientRect();
        setPos((event.clientX - rect.left) / rect.width * 100);
    }

    /** @description 경계 드래그 시작, 터치는 가로 이동이 확인될 때까지 보류 */
    function startSlide(event) {
        slider.pointer = event.pointerId;
        slider.startX = event.clientX;
        slider.startY = event.clientY;
        slider.dragging = event.pointerType !== "touch";
        if (slider.dragging) {
            stage.classList.add("dragging");
            moveToPointer(event);
        }
    }

    /** @description 경계 드래그 중 이동, 터치는 세로 이동이 크면 드래그를 포기 */
    function moveSlide(event) {
        if (!slider.dragging) {
            const dx = Math.abs(event.clientX - slider.startX);
            const dy = Math.abs(event.clientY - slider.startY);
            if (dy > dx && dy > TOUCH_SLOP) {
                slider.pointer = null;
                return;
            }
            if (dx <= TOUCH_SLOP) return;
            slider.dragging = true;
        }
        stage.classList.add("dragging");
        moveToPointer(event);
    }

    /** @description 경계 드래그 종료 */
    function endSlide() {
        slider.pointer = null;
        slider.dragging = false;
        stage.classList.remove("dragging");
    }

    divider.addEventListener("keydown", (event) => {
        const step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
        const moves = { ArrowLeft: -step, ArrowDown: -step, ArrowRight: step, ArrowUp: step, PageDown: -KEY_STEP_LARGE, PageUp: KEY_STEP_LARGE };
        if (event.key in moves) setPos(state.pos + moves[event.key]);
        else if (event.key === "Home") setPos(0);
        else if (event.key === "End") setPos(100);
        else return;
        event.preventDefault();
    });

    // 포인터 처리: 손잡이 띠에서 시작한 한 포인터는 경계 이동, 그 밖의 한 포인터는 확대 상태에서 이미지 이동
    // 두 번째 손가락이 닿으면 어디서 시작했든 경계 이동을 멈추고 두 손가락 확대/축소로 전환
    stage.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        gesture.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        stage.setPointerCapture(event.pointerId);
        if (gesture.pointers.size === 2) {
            endSlide();
            clearTimeout(gesture.settleTimer);
            gesture.pinch = _pinchOf(gesture.pointers);
            stage.classList.remove("panning");
            return;
        }
        if (gesture.pointers.size === 1 && divider.contains(event.target)) startSlide(event);
    });
    stage.addEventListener("pointermove", (event) => {
        const last = gesture.pointers.get(event.pointerId);
        if (!last) return;
        const point = { x: event.clientX, y: event.clientY };
        gesture.pointers.set(event.pointerId, point);
        if (gesture.pinch) {
            const pinch = _pinchOf(gesture.pointers);
            const focus = toStage(gesture.pinch.x, gesture.pinch.y);
            zoomAt(focus.x, focus.y, view.scale * pinch.distance / gesture.pinch.distance,
                pinch.x - gesture.pinch.x, pinch.y - gesture.pinch.y);
            gesture.pinch = pinch;
            return;
        }
        if (event.pointerId === slider.pointer) {
            moveSlide(event);
            return;
        }
        if (!isZoomed()) return;
        view.touched = true;
        stage.classList.add("panning");
        setView(view.scale, view.x + point.x - last.x, view.y + point.y - last.y);
    });

    /** @description 포인터 해제, 두 손가락 확대가 끝나면 정수 배율로 맞춤 */
    function endPointer(event) {
        if (!gesture.pointers.delete(event.pointerId)) return;
        if (event.pointerId === slider.pointer) endSlide();
        if (gesture.pinch && gesture.pointers.size < 2) {
            const focus = toStage(gesture.pinch.x, gesture.pinch.y);
            gesture.pinch = null;
            settle(focus.x, focus.y);
        }
        if (gesture.pointers.size === 0) stage.classList.remove("panning");
    }
    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);

    // 맞춤 배율의 한 손가락 세로 이동은 페이지 스크롤, 두 손가락은 브라우저 대신 이미지 확대에 사용
    stage.addEventListener("touchmove", (event) => {
        if (event.touches.length > 1) event.preventDefault();
    }, { passive: false });

    // 휠: 위로 굴리면 확대, 맞춤 배율에서 아래로 굴리면 연속 입력 중이어도 항상 페이지 스크롤로 통과
    // 페이지 스크롤 중 이미지가 커서 아래로 들어와도 스크롤을 가로채지 않음
    // Ctrl+휠(트랙패드 핀치)은 연속 확대 후 정수 배율로 맞춤
    stage.addEventListener("wheel", (event) => {
        const delta = event.deltaY * (event.deltaMode === 1 ? 40 : event.deltaMode === 2 ? 800 : 1);
        if (delta === 0) return;
        if (!event.ctrlKey && delta > 0 && !isZoomed()) {
            view.wheelAt = -Infinity;
            view.wheelRest = 0;
            return;
        }
        const now = event.timeStamp;
        const engaged = now - view.wheelAt < WHEEL_LATCH_MS;
        if (!event.ctrlKey && !engaged && now - pageWheelAt < WHEEL_LATCH_MS) return;
        event.preventDefault();
        view.wheelAt = now;
        const focus = toStage(event.clientX, event.clientY);
        if (event.ctrlKey) {
            const limited = Math.max(-PINCH_WHEEL_LIMIT, Math.min(PINCH_WHEEL_LIMIT, delta));
            zoomAt(focus.x, focus.y, view.scale * Math.exp(-limited * PINCH_WHEEL_SPEED));
            clearTimeout(gesture.settleTimer);
            gesture.settleTimer = setTimeout(() => settle(focus.x, focus.y), SETTLE_MS);
            return;
        }
        if (Math.abs(delta) >= WHEEL_NOTCH) {
            view.wheelRest = 0;
            stepZoom(-Math.sign(delta), focus.x, focus.y);
            return;
        }
        if (Math.sign(view.wheelRest) !== Math.sign(delta)) view.wheelRest = 0;
        view.wheelRest += delta;
        if (Math.abs(view.wheelRest) < WHEEL_NOTCH) return;
        stepZoom(-Math.sign(view.wheelRest), focus.x, focus.y);
        view.wheelRest = 0;
    }, { passive: false });

    boost.addEventListener("click", () => setBoost(!state.boost));

    relabel();
    relabelers.push(relabel);
    commitRight(DEFAULT_RIGHT);
    return { section, wrap, fit };
}


/** @description 목록 로드 후 항목, 색인 렌더링 */
async function init() {
    _initLocaleMenu();
    let response;
    try {
        response = await fetch(MANIFEST_URL, { cache: "no-cache" });
    } catch (e) {
        console.error(e.name + "\n" + e.message + "\n" + e.stack);
        _showStatus("status.loadFailed");
        return;
    }
    if (!response.ok) {
        _showStatus("status.loadFailedHttp", { status: response.status });
        return;
    }
    const manifest = await response.json();
    if (manifest.items.length === 0) {
        _showStatus("status.empty");
        return;
    }

    const template = document.getElementById("item-template");
    const container = document.getElementById("items");
    const indexList = document.getElementById("index-list");
    const viewers = manifest.items.map((item) => createItem(item, template));
    container.append(...viewers.map((viewer) => viewer.section));
    indexList.append(...manifest.items.map(_createIndexLink));
    indexList.parentElement.hidden = false;
    viewers.forEach((viewer) => viewer.fit());

    const fitters = new Map(viewers.map((viewer) => [viewer.wrap, viewer.fit]));
    const widths = new Map();
    const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
            const width = entry.contentRect.width;
            if (widths.get(entry.target) === width) continue;
            widths.set(entry.target, width);
            fitters.get(entry.target)();
        }
    });
    viewers.forEach((viewer) => observer.observe(viewer.wrap));
    _watchPixelRatio(() => viewers.forEach((viewer) => viewer.fit()));
    _jumpToHash();
}




/* =================================== 이벤트/내보내기 =================================== */


// 이미지 밖에서 시작한 페이지 스크롤 기록, 스크롤 도중 이미지 위를 지나도 확대로 바뀌지 않게 함
window.addEventListener("wheel", (event) => {
    if (!event.defaultPrevented) pageWheelAt = event.timeStamp;
}, { passive: true });

init();
