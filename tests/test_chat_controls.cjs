const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('agent_app/chat_controls.js', 'utf8');

class Element {
  constructor(tag = 'DIV') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
    this.attributes = {}; this.listeners = {}; this.className = ''; this.disabled = false;
    this.classList = {add: name => { this.className += ' ' + name; }};
  }
  append(...nodes) { for (const node of nodes) { node.parentElement = this; this.children.push(node); } }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  matches(selector) { return selector === ':disabled' ? this.disabled : selector === 'button' ? this.tagName === 'BUTTON' : selector.startsWith('.') && this.className.split(' ').includes(selector.slice(1)); }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
  querySelectorAll(selector) { return this.children.flatMap(node => [...(node.matches(selector) ? [node] : []), ...node.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this); }
  click() { this.listeners.click?.({preventDefault() {}, stopPropagation() {}}); }
}
class TextArea extends Element {
  constructor() { super('TEXTAREA'); this.events = []; }
  get value() { return this._value; }
  set value(value) { this._value = value; }
  dispatchEvent(event) { this.events.push(event); }
}
function fixture(shadow = false) {
  const ids = ['a'.repeat(32), 'b'.repeat(32), 'c'.repeat(32)];
  const conversation = new Element(), manifest = new Element(), request = new Element(), send = new Element('BUTTON');
  const marker = new Element(); marker.setAttribute('data-message-ids', JSON.stringify(ids));
  manifest.querySelector = () => marker;
  const field = new TextArea(); request.querySelector = () => field;
  for (let i = 0; i < 3; i++) { const row = new Element(); row.className = 'message-row'; const native = new Element(); native.className = 'message-buttons'; row.append(native); conversation.append(row); }
  const regions = {'#conversation': conversation, '#message-action-manifest': manifest, '#delete-message-request': request, '#send-message': send};
  const observers = [], frames = new Map(), windowEvents = {}; let nextFrame = 0;
  const root = {querySelector: selector => regions[selector] || null};
  const document = {
    body: new Element(),
    querySelector: selector => selector === 'gradio-app' ? (shadow ? {shadowRoot: root} : null) : shadow ? null : root.querySelector(selector),
    createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)
  };
  class Observer { constructor(callback) { this.callback = callback; observers.push(this); } observe() {} disconnect() { this.disconnected = true; } }
  const context = {document, window: {addEventListener: (type, callback) => windowEvents[type] = callback}, MutationObserver: Observer,
    HTMLTextAreaElement: TextArea, HTMLInputElement: Element, Event: class {constructor(type, options) {this.type = type; this.bubbles = options.bubbles;}},
    requestAnimationFrame: callback => { frames.set(++nextFrame, callback); return nextFrame; }, cancelAnimationFrame: id => frames.delete(id), Date};
  vm.runInNewContext('(' + source + ')()', context);
  const refresh = () => { observers[1].callback(); for (const [id, callback] of [...frames]) {frames.delete(id); callback();} };
  return {ids, conversation, marker, field, send, refresh, observers, windowEvents, context};
}
for (const shadow of [false, true]) {
  const f = fixture(shadow);
  let buttons = f.conversation.querySelectorAll('.brams-delete-message');
  assert.equal(buttons.length, 3);
  buttons[0].click();
  assert.equal(JSON.parse(f.field.value).message_id, f.ids[0]);
  assert.equal(f.field.events[0].type, 'input');
  assert.equal(f.field.events[0].bubbles, true);
  assert.equal(buttons[0].disabled, true);
  buttons[0].click(); assert.equal(f.field.events.length, 1);
  f.conversation.children[0].remove();
  f.marker.setAttribute('data-message-ids', JSON.stringify(f.ids.slice(1)));
  f.refresh(); buttons = f.conversation.querySelectorAll('.brams-delete-message');
  assert.equal(buttons.length, 2);
  buttons[1].click(); assert.equal(JSON.parse(f.field.value).message_id, f.ids[2]);
  f.send.disabled = true; f.refresh();
  const count = f.field.events.length; buttons[0].click(); assert.equal(f.field.events.length, count);
  f.send.disabled = false; f.marker.setAttribute('data-message-ids', JSON.stringify([f.ids[1]])); f.refresh();
  assert.equal(f.conversation.querySelectorAll('.brams-message-actions').length, 0);
  f.windowEvents.pagehide(); assert.equal(f.observers[1].disconnected, true);
}
console.log('Message controls: correct target after multiple messages/deletion, busy guards, stale-layout safety and shadow roots passed.');
