/** @description pywebview 데스크톱 복원 백엔드 */

const STAGES = ["dots", "sparkle", "periodic", "grime", "mottle", "lattice", "surface", "contour", "defocus", "cel_fill", "dither"];
const PRESETS = {
    safe: { dots: 0, sparkle: 0, periodic: 1, grime: 0.5, mottle: 0.35, lattice: 0, surface: 0, contour: 0, defocus: 0, cel_fill: 0, dither: 0 },
    balanced: { dots: 1, sparkle: 0, periodic: 1, grime: 1, mottle: 0.7, lattice: 0, surface: 0, contour: 0, defocus: 0, cel_fill: 0, dither: 0 },
    strong: { dots: 1, sparkle: 1, periodic: 1, grime: 1.1, mottle: 1, lattice: 1, surface: 0, contour: 1, defocus: 1, cel_fill: 1, dither: 1 },
};

/** @description 데스크톱 백엔드 생성 */
export function createBackend() {
    const ready = new Promise((resolve) => {
        if (window.pywebview) resolve(window.pywebview.api);
        else window.addEventListener("pywebviewready", () => resolve(window.pywebview.api), { once: true });
    });
    const jobs = new Map();
    return {
        async init() {
            return { kind: "desktop", stages: STAGES, presets: PRESETS, maxPixels: null, nativePicker: true, desktopAppUrl: null };
        },
        async pickFiles() {
            return (await ready).pick_files();
        },
        async addFiles(files) {
            const api = await ready;
            const result = { added: [], rejected: [] };
            for (const file of files) {
                const data = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
                    reader.onerror = () => reject(reader.error);
                    reader.readAsDataURL(file);
                });
                const part = await api.add_bytes(file.name, data);
                result.added.push(...part.added);
                result.rejected.push(...part.rejected);
            }
            return result;
        },
        async process(id, settings, onProgress) {
            const job = { cancelled: false, active: true, token: crypto.randomUUID() };
            job.done = new Promise((resolve) => { job.resolve = resolve; });
            jobs.set(id, job);
            const api = await ready;
            if (job.cancelled) {
                jobs.delete(id);
                job.resolve();
                throw new DOMException("Cancelled", "AbortError");
            }
            let polling = false;
            const timer = setInterval(async () => {
                if (polling) return;
                polling = true;
                try {
                    if (job.cancelled) await api.cancel(id, job.token);
                    const progress = await api.get_progress(id);
                    if (job.active && progress) onProgress(progress);
                } finally {
                    polling = false;
                }
            }, 100);
            try {
                const prepared = await api.process(id, settings, job.token);
                if (job.cancelled && prepared.ready) await api.cancel(id, job.token);
                if (prepared.cancelled) throw new DOMException("Cancelled", "AbortError");
                const result = await api.finish_process(id);
                if (job.cancelled) await api.cancel(id, job.token);
                if (job.cancelled || result.cancelled) throw new DOMException("Cancelled", "AbortError");
                job.active = false;
                onProgress({ fraction: 1, stage: "encode" });
                return result;
            } finally {
                clearInterval(timer);
                job.active = false;
                jobs.delete(id);
                job.resolve();
                api.ack_process(id, job.token);
            }
        },
        cancel(id) {
            const job = jobs.get(id);
            if (job) {
                job.cancelled = true;
                ready.then((api) => api.cancel(id, job.token));
            }
        },
        remove(id) {
            const job = jobs.get(id);
            if (job) {
                job.cancelled = true;
                ready.then((api) => api.cancel(id, job.token));
            }
            Promise.all([ready, job ? job.done : Promise.resolve()]).then(([api]) => api.remove(id));
        },
        async pickExportTarget() {
            return (await ready).pick_export_target();
        },
        async export(ids, output, target) {
            return (await ready).export(ids, output, target);
        },
        async setLocale(code) {
            return (await ready).set_locale(code);
        },
    };
}
