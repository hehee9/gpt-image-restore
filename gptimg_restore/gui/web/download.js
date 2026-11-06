/**
 * @file gptimg_restore/gui/web/download.js
 * @description 웹 내보내기 다운로드: 파일 하나는 그대로, 여러 개는 무압축 ZIP 하나로 묶음
 */




/* =================================== 전역 상수/변수 =================================== */


const ZIP_NAME = "gpt-image-restore.zip";
const UTF8_FLAG = 0x0800;
const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        }
        table[n] = c >>> 0;
    }
    return table;
})();




/* =================================== 헬퍼/유틸 =================================== */


/** @description CRC-32 계산 */
function _crc32(bytes) {
    let c = 0xffffffff;
    for (const byte of bytes) {
        c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xffffffff) >>> 0;
}

/** @description ZIP 헤더용 MS-DOS 날짜와 시각 */
function _dosDateTime(date) {
    return {
        time: (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1),
        date: ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
    };
}

/** @description 겹치는 이름을 "이름 (2).ext" 형태로 구분 */
function _uniqueNames(names) {
    const used = new Set();
    return names.map((name) => {
        const dot = name.lastIndexOf(".");
        let candidate = name;
        for (let n = 2; used.has(candidate); n++) {
            candidate = `${name.slice(0, dot)} (${n})${name.slice(dot)}`;
        }
        used.add(candidate);
        return candidate;
    });
}

/** @description Blob을 파일로 다운로드 */
function _downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
}




/* =================================== 메인 로직 =================================== */


/**
 * 무압축(STORE) ZIP 생성. 이미 압축된 이미지 형식이라 압축 이득이 작음
 * @param {{name: string, blob: Blob}[]} files
 * @returns {Promise<Blob>}
 */
export async function createZip(files) {
    const encoder = new TextEncoder();
    const { time, date } = _dosDateTime(new Date());
    const names = _uniqueNames(files.map((file) => file.name));
    const localParts = [];
    const centralParts = [];
    let offset = 0;
    let centralSize = 0;

    for (const [index, file] of files.entries()) {
        const data = new Uint8Array(await file.blob.arrayBuffer());
        const name = encoder.encode(names[index]);
        const crc = _crc32(data);

        const local = new DataView(new ArrayBuffer(30));
        local.setUint32(0, 0x04034b50, true);
        local.setUint16(4, 20, true);
        local.setUint16(6, UTF8_FLAG, true);
        local.setUint16(10, time, true);
        local.setUint16(12, date, true);
        local.setUint32(14, crc, true);
        local.setUint32(18, data.length, true);
        local.setUint32(22, data.length, true);
        local.setUint16(26, name.length, true);
        localParts.push(local, name, data);

        const central = new DataView(new ArrayBuffer(46));
        central.setUint32(0, 0x02014b50, true);
        central.setUint16(4, 20, true);
        central.setUint16(6, 20, true);
        central.setUint16(8, UTF8_FLAG, true);
        central.setUint16(12, time, true);
        central.setUint16(14, date, true);
        central.setUint32(16, crc, true);
        central.setUint32(20, data.length, true);
        central.setUint32(24, data.length, true);
        central.setUint16(28, name.length, true);
        central.setUint32(42, offset, true);
        centralParts.push(central, name);

        offset += 30 + name.length + data.length;
        centralSize += 46 + name.length;
    }

    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, centralSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...localParts, ...centralParts, end], { type: "application/zip" });
}


/**
 * 내보낸 파일 다운로드. 1개면 그 파일, 2개 이상이면 gpt-image-restore.zip
 * @param {{name: string, blob: Blob}[]} files
 */
export async function downloadFiles(files) {
    if (files.length === 1) {
        _downloadBlob(files[0].blob, files[0].name);
        return;
    }
    _downloadBlob(await createZip(files), ZIP_NAME);
}

