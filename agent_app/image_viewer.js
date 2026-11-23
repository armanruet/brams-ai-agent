() => {
    const find = selector => document.querySelector(selector)
        || document.querySelector("gradio-app")?.shadowRoot?.querySelector(selector);
    let bootstrap;
    const mount = () => {
        const conversation = find("#conversation");
        if (!conversation) return false;
        if (conversation.dataset.imageViewerBound) return true;
        conversation.dataset.imageViewerBound = "true";
        bootstrap?.disconnect();

        const dialog = document.createElement("dialog");
        dialog.className = "brams-image-viewer";
        dialog.setAttribute("aria-label", "Spectrogram image viewer");
        const toolbar = document.createElement("div");
        toolbar.className = "brams-image-viewer-toolbar";
        const caption = document.createElement("span");
        caption.className = "brams-image-viewer-caption";
        const zoom = document.createElement("button");
        zoom.type = "button";
        zoom.textContent = "Actual size";
        zoom.setAttribute("aria-pressed", "false");
        const close = document.createElement("button");
        close.type = "button";
        close.textContent = "Close";
        close.setAttribute("aria-label", "Close image viewer");
        toolbar.append(caption, zoom, close);
        const viewport = document.createElement("div");
        viewport.className = "brams-image-viewer-viewport";
        const fullImage = document.createElement("img");
        viewport.append(fullImage);
        dialog.append(toolbar, viewport);
        conversation.append(dialog);

        let opener = null;
        const show = image => {
            if (!image.currentSrc && !image.src) return;
            opener = image.closest("button") || image;
            fullImage.src = image.currentSrc || image.src;
            fullImage.alt = image.alt || "Enlarged spectrogram";
            const row = image.closest(".message-row");
            caption.textContent = row?.querySelector('[data-testid="block-label"]')?.textContent?.trim()
                || image.alt || "Spectrogram";
            viewport.classList.remove("is-actual-size");
            zoom.textContent = "Actual size";
            zoom.setAttribute("aria-pressed", "false");
            dialog.showModal();
            close.focus();
        };
        zoom.addEventListener("click", () => {
            const actual = viewport.classList.toggle("is-actual-size");
            zoom.textContent = actual ? "Fit to screen" : "Actual size";
            zoom.setAttribute("aria-pressed", String(actual));
        });
        close.addEventListener("click", () => dialog.close());
        dialog.addEventListener("click", event => {
            if (event.target === dialog) dialog.close();
        });
        dialog.addEventListener("close", () => {
            fullImage.removeAttribute("src");
            if (opener?.isConnected) opener.focus();
            opener = null;
        });
        const prepareImages = () => {
            conversation.querySelectorAll(".message-row .message .image-frame img").forEach(image => {
                const trigger = image.closest("button");
                if (!trigger || trigger.dataset.imageViewerReady) return;
                trigger.dataset.imageViewerReady = "true";
                trigger.classList.add("brams-viewable-image");
                trigger.setAttribute("aria-label", "Open spectrogram image viewer");
                trigger.title = "Click to enlarge";
            });
        };
        conversation.addEventListener("click", event => {
            const trigger = event.target.closest?.(".message-row .message button");
            const image = trigger?.querySelector(".image-frame img");
            if (image && conversation.contains(image)) show(image);
        });
        const observer = new MutationObserver(prepareImages);
        observer.observe(conversation, {childList: true, subtree: true});
        window.addEventListener("pagehide", () => observer.disconnect(), {once: true});
        prepareImages();
        return true;
    };
    bootstrap = new MutationObserver(mount);
    if (!mount()) {
        bootstrap.observe(document.body, {childList: true, subtree: true});
        const shadow = document.querySelector("gradio-app")?.shadowRoot;
        if (shadow) bootstrap.observe(shadow, {childList: true, subtree: true});
    }
}
