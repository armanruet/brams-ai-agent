() => {
    // Client-only layout preference; never reads prompts, keys, or model state.
    let bootstrap;
    const watched = new WeakSet();
    const mount = () => {
        // Gradio may render directly into the document or inside its web component.
        const roots = [document, ...Array.from(document.querySelectorAll("gradio-app"))
            .map((app) => app.shadowRoot).filter(Boolean)];
        const root = roots.find((item) => item.getElementById("workspace-layout")
            && item.getElementById("workspace-split-handle"));
        if (!root) {
            for (const item of roots) {
                const target = item === document ? document.body : item;
                if (target && !watched.has(target)) {
                    bootstrap.observe(target, {childList: true, subtree: true});
                    watched.add(target);
                }
            }
            return false;
        }
        const layout = root.getElementById("workspace-layout");
        const handle = root.getElementById("workspace-split-handle");
        if (!layout || !handle) return false;
        if (handle.dataset.splitReady) return true;
        handle.dataset.splitReady = "true";
        if (bootstrap) bootstrap.disconnect();

        const storageKey = "brams.monitor-width-percent.v1";
        const separatorWidth = 18;
        let preferred = 40;
        let pointer = null;
        let previousSelection = "";
        try {
            const stored = Number(localStorage.getItem(storageKey));
            if (Number.isFinite(stored) && stored >= 20 && stored <= 65) preferred = stored;
        } catch (_) { /* Storage can be unavailable in private/restricted browsers. */ }

        const bounds = () => {
            const width = Math.max(1, layout.getBoundingClientRect().width - separatorWidth);
            return {min: Math.max(20, 300 / width * 100), max: Math.min(65, 100 - 320 / width * 100)};
        };
        const stacked = () => window.matchMedia("(max-width: 900px)").matches
            || layout.getBoundingClientRect().width < 680;
        const render = () => {
            const narrow = stacked();
            layout.classList.toggle("workspace-stacked", narrow);
            handle.tabIndex = narrow ? -1 : 0;
            handle.setAttribute("aria-disabled", String(narrow));
            if (narrow) return;
            const range = bounds();
            const actual = Math.min(range.max, Math.max(range.min, preferred));
            layout.style.setProperty("--chat-fr", `${100 - actual}fr`);
            layout.style.setProperty("--monitor-fr", `${actual}fr`);
            handle.setAttribute("aria-valuemin", String(Math.round(range.min)));
            handle.setAttribute("aria-valuemax", String(Math.round(range.max)));
            handle.setAttribute("aria-valuenow", String(Math.round(actual)));
            handle.setAttribute("aria-valuetext", `Request monitor ${Math.round(actual)} percent`);
        };
        const save = () => {
            try { localStorage.setItem(storageKey, String(preferred)); } catch (_) {}
        };
        const adjust = (percent, persist) => {
            if (stacked()) return;
            const range = bounds();
            preferred = Math.min(range.max, Math.max(range.min, percent));
            render();
            if (persist) save();
        };
        const endDrag = () => {
            if (pointer === null) return;
            pointer = null;
            document.body.style.userSelect = previousSelection;
            layout.classList.remove("is-resizing");
            save();
        };
        handle.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || stacked()) return;
            pointer = event.pointerId;
            previousSelection = document.body.style.userSelect;
            handle.setPointerCapture(pointer);
            layout.classList.add("is-resizing");
            document.body.style.userSelect = "none";
            event.preventDefault();
        });
        handle.addEventListener("pointermove", (event) => {
            if (event.pointerId !== pointer) return;
            const rect = layout.getBoundingClientRect();
            adjust((rect.right - event.clientX - separatorWidth / 2)
                / (rect.width - separatorWidth) * 100, false);
        });
        handle.addEventListener("pointerup", endDrag);
        handle.addEventListener("pointercancel", endDrag);
        handle.addEventListener("lostpointercapture", endDrag);
        handle.addEventListener("dblclick", () => adjust(40, true));
        handle.addEventListener("keydown", (event) => {
            if (stacked()) return;
            const current = Number(handle.getAttribute("aria-valuenow"));
            const actions = {ArrowLeft: current + 2, ArrowRight: current - 2,
                Home: bounds().min, End: bounds().max, Enter: 40};
            if (Object.prototype.hasOwnProperty.call(actions, event.key)) {
                event.preventDefault();
                adjust(actions[event.key], true);
            }
        });
        const observer = new ResizeObserver(render);
        observer.observe(layout);
        window.addEventListener("resize", render);
        window.addEventListener("blur", endDrag);
        window.addEventListener("pagehide", () => {
            endDrag();
            observer.disconnect();
            window.removeEventListener("resize", render);
            window.removeEventListener("blur", endDrag);
        }, {once: true});
        render();
        return true;
    };
    bootstrap = new MutationObserver(mount);
    if (!mount()) bootstrap.observe(document.body, {childList: true, subtree: true});
}
