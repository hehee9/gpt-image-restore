/**
 * @file gptimg_restore/gui/web/backend.js
 * @description 화면과 처리 백엔드 사이의 계약 정의 및 시작 시 백엔드 선택
 *
 * @note 백엔드 선택: 페이지 주소의 backend 쿼리 값
 *   - desktop: pywebview 데스크톱 앱. 앱은 창을 index.html?backend=desktop 으로 열어야 함
 *     (window.pywebview는 비동기로 주입되므로 존재 여부 감지 대신 쿼리 사용)
 *   - web 또는 값 없음: 온라인 변환기(Web Worker 안의 Pyodide)
 *   - mock-desktop, mock-web: 화면 개발용 가짜 백엔드
 *
 * @note 구현 파일: 각 모듈은 createBackend(): Backend 를 export
 *   - ./backend-desktop.js (데스크톱 구현 담당)
 *   - ./backend-web.js (웹 구현 담당, Worker 파일 이름은 자유)
 *   - ./backend-mock.js (현재 제공)
 *
 * @note 화면 쪽 호출 보장
 *   - init()은 다른 메서드보다 먼저 한 번만 호출
 *   - process()는 전체에서 동시에 하나만 실행. 다음 호출은 이전 Promise 정산 후
 *   - export()에 넘기는 id는 모두 현재 설정으로 process()가 성공한 항목.
 *     그 처리의 settings.bitDepth는 output이 PNG 16비트면 16, 그 밖에는 8
 *   - 내보내기 진행 중(pickExportTarget() 이후 export() 정산까지) 설정 고정.
 *     process()는 내보낼 항목 처리에만 사용하고 export() 호출 중에는 호출하지 않음.
 *     remove(), pickFiles(), addFiles()는 새로 호출하지 않음
 *   - remove()된 id는 이후 어떤 메서드에도 다시 전달하지 않음
 *   - setLocale()은 init() 직후 한 번, 이후 화면 언어를 바꿀 때마다 호출
 */




/* =================================== 계약 타입 =================================== */


/**
 * @typedef {"desktop" | "web"} BackendKind
 */

/**
 * 화면 언어 코드
 * @typedef {"en" | "ko" | "ja" | "zh-CN"} LocaleCode
 */

/**
 * init() 결과
 * @typedef {Object} BackendInfo
 * @property {BackendKind} kind
 * @property {string[]} stages 단계 id 목록(화면 표시 순서). 현재 값:
 *   dots, sparkle, periodic, grime, mottle, lattice, surface, contour, defocus, cel_fill, dither
 * @property {Object<string, Object<string, number>>} presets
 *   safe, balanced, strong 각각의 단계별 강도. 모든 단계 키 포함. 값은 gptimg_restore.cli.PRESET_STRENGTHS와 동일
 * @property {number | null} maxPixels 입력 최대 화소 수. 웹 16777216(4096×4096), 데스크톱 null(제한 없음)
 * @property {boolean} nativePicker
 *   true: 파일 선택 시 화면이 pickFiles() 호출(데스크톱 네이티브 대화상자)
 *   false: 화면이 <input type="file">로 고른 File을 addFiles()에 전달(웹)
 * @property {string | null} desktopAppUrl 웹의 크기 초과 안내에 연결할 데스크톱 앱 주소. 없거나 데스크톱이면 null
 */

/**
 * 대기열에 추가된 원본 이미지
 * @typedef {Object} SourceItem
 * @property {string} id 백엔드가 발급한 고유 id
 * @property {string} name 원본 파일 이름(확장자 포함)
 * @property {number} width 원본 픽셀 폭
 * @property {number} height 원본 픽셀 높이
 * @property {string} originalUrl 원본 전체 해상도 이미지 주소(<img src>로 사용, blob: 또는 http:)
 * @property {string} thumbnailUrl 짧은 변 160px 내외의 축소 이미지 주소
 */

/**
 * 추가하지 못한 파일
 * @typedef {Object} Rejection
 * @property {string} name 파일 이름
 * @property {"too_large" | "unreadable"} reason
 *   too_large: maxPixels 초과(웹 전용). unreadable: 지원하지 않는 형식(png, jpg, jpeg, webp 외) 또는 해독 실패
 */

/**
 * pickFiles()/addFiles() 결과
 * @typedef {Object} AddResult
 * @property {SourceItem[]} added 입력 순서 유지
 * @property {Rejection[]} rejected
 */

/**
 * 복원 설정
 * @typedef {Object} RestoreSettings
 * @property {Object<string, number>} strengths 모든 단계 키 포함. 0이면 해당 단계 비활성. 범위 0~2
 * @property {"conservative" | "strict"} dotsMode CLI --dots-mode 값
 * @property {8 | 16} bitDepth 처리 비트 깊이. 출력 형식이 png이고 PNG 비트 깊이가 16일 때만 16, 그 밖에는 8.
 *   디더링과 마지막 반짝임 단계가 이 값에 따라 달라지므로 내보내기 인코딩이 아닌 처리 단계에 적용
 */

/**
 * 진행 보고
 * @typedef {Object} Progress
 * @property {number} fraction 0~1, 단조 증가
 * @property {string} stage 현재 단계 id, 또는 "load"(런타임 준비, 이미지 로드), "encode"(미리보기 생성)
 */

/**
 * @typedef {Object} ProcessResult
 * @property {string} restoredUrl 복원 결과 전체 해상도 미리보기 주소.
 *   같은 id의 다음 결과가 나오거나 remove()될 때 이전 주소 해제는 백엔드 책임
 */

/**
 * 출력 설정
 * @typedef {Object} OutputSettings
 * @property {"png" | "jpg" | "webp"} format
 * @property {8 | 16} bitDepth PNG에만 적용. 보관 결과는 이미 이 값에 맞는 처리 비트 깊이로 처리됨(png가 아니면 8)
 * @property {number} quality 0~100 정수, jpg/webp에만 적용
 */

/**
 * 내보내기 대상
 * @typedef {{kind: "folder", path: string} | {kind: "download"}} ExportTarget
 */

/**
 * @typedef {Object} ExportResult
 * @property {number} count 저장 또는 다운로드한 파일 수
 * @property {string | null} location 저장 폴더 경로. 다운로드면 null
 */

/**
 * 처리 백엔드 인터페이스
 * @typedef {Object} Backend
 *
 * @property {() => Promise<BackendInfo>} init
 *   메타데이터 반환. 무거운 런타임(Pyodide, numpy 등) 준비를 기다리지 않고 바로 반환.
 *   준비는 백그라운드에서 시작하고 process()가 준비 완료를 대기(그동안 stage "load" 보고)
 *
 * @property {() => Promise<AddResult>} pickFiles
 *   nativePicker가 true일 때만 호출. 네이티브 다중 선택 대화상자(png, jpg, jpeg, webp).
 *   대화상자 취소 시 {added: [], rejected: []}
 *
 * @property {(files: File[]) => Promise<AddResult>} addFiles
 *   끌어다 놓기(양쪽 모두)와 웹 파일 선택에서 받은 File 전달.
 *   데스크톱은 File에 경로가 없으므로 바이트를 읽어 Python으로 넘기는 방식 사용
 *
 * @property {(id: string, settings: RestoreSettings, onProgress: (progress: Progress) => void) => Promise<ProcessResult>} process
 *   복원 실행. settings.bitDepth로 파이프라인 실행.
 *   export()를 위해 id별 마지막 결과 하나만 보관(16비트 저장에 필요한 고정밀 결과 포함).
 *   cancel()되면 name이 "AbortError"인 DOMException으로 reject. 그 밖의 실패는 Error로 reject(message는 로그용)
 *
 * @property {(id: string) => void} cancel
 *   해당 id의 진행 중 process() 중단. 진행 중이 아니면 무시.
 *   현재 단계가 끝난 뒤 멈춰도 되지만 결과는 반환하지 않고 AbortError로 reject
 *
 * @property {(id: string) => void} remove
 *   대기열에서 제거. 보관 중인 결과와 originalUrl, thumbnailUrl, restoredUrl 해제
 *
 * @property {() => Promise<ExportTarget | null>} pickExportTarget
 *   내보내기 시작 시 일괄 처리 전에 한 번 호출.
 *   데스크톱: 폴더 선택 대화상자, 취소 시 null. 웹: 바로 {kind: "download"}
 *
 * @property {(ids: string[], output: OutputSettings, target: ExportTarget) => Promise<ExportResult>} export
 *   보관 중인 결과를 재처리 없이 output 형식으로 인코딩해 저장. 파일 이름은 CLI와 같은 "<원본 stem>_restored.<format>".
 *   데스크톱: target.path에 기록(같은 이름은 덮어쓰기).
 *   웹: download.js의 downloadFiles() 사용(1개면 파일 하나, 2개 이상이면 ZIP 하나).
 *   저장 실패 시 Error로 reject
 *
 * @property {(code: LocaleCode) => Promise<void>} [setLocale]
 *   화면 언어 전달(선택 구현). 데스크톱: pywebview 브리지 set_locale(code)로 Python 쪽에 전달.
 *   웹, mock: 동작 없음
 */




/* =================================== 백엔드 선택 =================================== */


const BACKEND_LOADERS = {
    desktop: () => import("./backend-desktop.js").then((module) => module.createBackend()),
    web: () => import("./backend-web.js").then((module) => module.createBackend()),
    "mock-desktop": () => import("./backend-mock.js").then((module) => module.createMockBackend("desktop")),
    "mock-web": () => import("./backend-mock.js").then((module) => module.createMockBackend("web")),
};


/**
 * @description 페이지 주소의 backend 쿼리에 맞는 백엔드 생성
 * @returns {Promise<Backend>}
 */
export async function createBackend() {
    const name = new URLSearchParams(location.search).get("backend") ?? "web";
    const loader = BACKEND_LOADERS[name];
    if (!loader) {
        throw new Error(`Unknown backend value: ${name}`);
    }
    return loader();
}

