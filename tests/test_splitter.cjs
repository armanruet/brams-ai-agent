/* Execute the actual browser initializer against a small DOM fixture. No browser,
 * server, model, or third-party test dependency is needed for these interactions. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "../agent_app/splitter.js"), "utf8");

function environment({width = 1400, narrow = false, stored = null, blocked = false, shadow = false} = {}) {
    const listeners = {};
    const classes = new Set();
    const styles = {};
    const attrs = {};
    const storage = new Map(stored === null ? [] : [["brams.monitor-width-percent.v1", String(stored)]]);
    const layout = {style: {setProperty: (k, v) => { styles[k] = v; }},
        getBoundingClientRect: () => ({width, left: 0, right: width}),
        classList: {toggle: (k, on) => on ? classes.add(k) : classes.delete(k),
            add: (k) => classes.add(k), remove: (k) => classes.delete(k)}};
    const handle = {dataset: {}, tabIndex: 0,
        setAttribute: (k, v) => { attrs[k] = v; }, getAttribute: (k) => attrs[k],
        setPointerCapture: () => {}, addEventListener: (k, fn) => { listeners[k] = fn; }};
    const body = {style: {userSelect: "text"}};
    const windowEvents = {};
    const lookup = (id) => id === "workspace-layout" ? layout : handle;
    const context = {document: {body, getElementById: (id) => shadow ? null : lookup(id),
            querySelectorAll: () => shadow ? [{shadowRoot: {getElementById: lookup}}] : []},
        window: {matchMedia: () => ({matches: narrow}), addEventListener: (k, fn) => {windowEvents[k] = fn;},
            removeEventListener: (k) => {delete windowEvents[k];}},
        localStorage: {getItem: (k) => {if (blocked) throw Error("Storage blocked"); return storage.get(k) ?? null;},
            setItem: (k, v) => {if (blocked) throw Error("Storage blocked"); storage.set(k, v);}},
        MutationObserver: class {disconnect() {} observe() {}},
        ResizeObserver: class {constructor(fn) {this.fn = fn;} observe() {} disconnect() {}}};
    vm.runInNewContext(`(${source})()`, context);
    return {listeners, attrs, classes, styles, storage, body, handle, layout, windowEvents,
        resize: (newWidth) => {width = newWidth; windowEvents.resize();},
        mobile: (value) => {narrow = value; windowEvents.resize();},
        key: (key) => listeners.keydown({key, preventDefault() {}})};
}

// Keyboard operation grows/shrinks the monitor and remembers only the split.
const desktop = environment();
assert.equal(desktop.attrs["aria-valuenow"], "40");
desktop.key("ArrowLeft");
assert.equal(desktop.attrs["aria-valuenow"], "42");
desktop.key("ArrowRight");
assert.equal(desktop.attrs["aria-valuenow"], "40");
desktop.key("End");
assert.equal(desktop.attrs["aria-valuenow"], "65");
desktop.listeners.dblclick();
assert.equal(desktop.attrs["aria-valuenow"], "40");
assert.equal(desktop.storage.size, 1);

// Dragging is bounded so neither panel becomes unreadably small.
const drag = environment({width: 1000});
drag.listeners.pointerdown({button: 0, pointerId: 1, preventDefault() {}});
assert.equal(drag.body.style.userSelect, "none");
drag.listeners.pointermove({pointerId: 1, clientX: 9999});
let share = parseFloat(drag.styles["--monitor-fr"]);
assert.ok(share / 100 * 982 >= 300 - 1e-8);
drag.listeners.pointermove({pointerId: 1, clientX: -9999});
share = parseFloat(drag.styles["--monitor-fr"]);
assert.ok((100 - share) / 100 * 982 >= 320 - 1e-8);
drag.listeners.pointerup();
assert.equal(drag.body.style.userSelect, "text");
assert.equal(drag.classes.has("is-resizing"), false);

// Narrow viewports and a temporarily narrow settings-drawer workspace stack;
// expanding again restores the preference rather than resetting the chat.
const responsive = environment({stored: 55});
responsive.mobile(true);
assert.equal(responsive.handle.tabIndex, -1);
assert.equal(responsive.classes.has("workspace-stacked"), true);
responsive.mobile(false);
assert.equal(responsive.attrs["aria-valuenow"], "55");
responsive.resize(600);
assert.equal(responsive.classes.has("workspace-stacked"), true);
responsive.resize(1400);
assert.equal(responsive.attrs["aria-valuenow"], "55");

// Restricted storage or invalid prior values cannot prevent the app mounting.
assert.equal(environment({blocked: true}).attrs["aria-valuenow"], "40");
assert.equal(environment({stored: "bad"}).attrs["aria-valuenow"], "40");
assert.equal(environment({shadow: true}).attrs["aria-valuenow"], "40");
console.log("Splitter interaction checks passed: keyboard, drag bounds, persistence, responsive restoration, restricted storage.");
