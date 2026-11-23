() => {
    const find = selector => document.querySelector(selector)
        || document.querySelector("gradio-app")?.shadowRoot?.querySelector(selector);
    let bootstrap;
    const mount = () => {
        const conversation = find("#conversation");
        const manifest = find("#message-action-manifest");
        const request = find("#delete-message-request");
        const send = find("#send-message");
        if (!conversation || !manifest || !request || !send) return false;
        if (conversation.dataset.messageActionsBound) return true;
        conversation.dataset.messageActionsBound = "true";
        bootstrap?.disconnect();
        let frame = null;
        const sync = () => {
            frame = null;
            let ids;
            try {
                ids = JSON.parse(manifest.querySelector("[data-message-ids]")?.getAttribute("data-message-ids") || "[]");
            } catch (_) { ids = []; }
            const rows = Array.from(conversation.querySelectorAll(".message-row"))
                .filter(row => !row.parentElement?.closest(".message-row"));
            if (!Array.isArray(ids) || rows.length !== ids.length) {
                conversation.querySelectorAll(".brams-message-actions").forEach(node => node.remove());
                return;
            }
            const busy = send.matches(":disabled") || !!send.querySelector("button:disabled");
            rows.forEach((row, index) => {
                if (typeof ids[index] !== "string" || !/^[0-9a-f]{32}$/.test(ids[index])) return;
                let actions = row.querySelector(".brams-message-actions");
                if (!actions) {
                    actions = document.createElement("span");
                    actions.className = "brams-message-actions";
                    const button = document.createElement("button");
                    button.type = "button";
                    button.className = "brams-delete-message";
                    button.title = "Delete this message";
                    button.setAttribute("aria-label", "Delete this message");
                    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                    svg.setAttribute("viewBox", "0 0 24 24");
                    svg.setAttribute("aria-hidden", "true");
                    svg.setAttribute("fill", "none");
                    svg.setAttribute("stroke", "currentColor");
                    svg.setAttribute("stroke-width", "1.7");
                    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    path.setAttribute("d", "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7M14 10v7");
                    svg.append(path);
                    button.append(svg);
                    button.addEventListener("click", event => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (button.disabled || send.matches(":disabled") || send.querySelector("button:disabled")) return;
                        const field = request.querySelector("textarea, input");
                        if (!field) return;
                        const prototype = field.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                        const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
                        if (!setter) return;
                        setter.call(field, JSON.stringify({message_id: button.dataset.messageId, nonce: Date.now()}));
                        field.dispatchEvent(new Event("input", {bubbles: true}));
                        button.disabled = true;
                    });
                    actions.append(button);
                    const nativeActions = row.querySelector(".message-buttons");
                    if (!nativeActions) actions.classList.add("standalone");
                    (nativeActions || row).append(actions);
                }
                const button = actions.querySelector("button");
                button.dataset.messageId = ids[index];
                button.disabled = busy;
            });
        };
        const schedule = () => {
            if (frame === null) frame = requestAnimationFrame(sync);
        };
        const observer = new MutationObserver(schedule);
        observer.observe(conversation, {childList: true, subtree: true});
        observer.observe(manifest, {childList: true, subtree: true, attributes: true, attributeFilter: ["data-message-ids"]});
        observer.observe(send, {childList: true, subtree: true, attributes: true, attributeFilter: ["disabled"]});
        window.addEventListener("pagehide", () => {
            observer.disconnect();
            if (frame !== null) cancelAnimationFrame(frame);
        }, {once: true});
        sync();
        return true;
    };
    bootstrap = new MutationObserver(mount);
    if (!mount()) {
        bootstrap.observe(document.body, {childList: true, subtree: true});
        const shadow = document.querySelector("gradio-app")?.shadowRoot;
        if (shadow) bootstrap.observe(shadow, {childList: true, subtree: true});
    }
}
