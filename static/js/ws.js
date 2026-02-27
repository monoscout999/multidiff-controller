/**
 * WsManager — WebSocket con auto-reconexión y sistema de eventos.
 */
export class WsManager {
  constructor(url) {
    this.url = url;
    this._handlers = {};
    this._pingTimer = null;
    this._connect();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Register an event handler: ws.on('step', handler) */
  on(type, handler) {
    if (!this._handlers[type]) this._handlers[type] = [];
    this._handlers[type].push(handler);
    return this;
  }

  /** Send a JSON message to the server. */
  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _connect() {
    this.ws = new WebSocket(this.url);

    this.ws.addEventListener('open', () => {
      console.debug('[WS] connected');
      this._startPing();
    });

    this.ws.addEventListener('message', ({ data }) => {
      try {
        const msg = JSON.parse(data);
        (this._handlers[msg.type] || []).forEach(h => h(msg));
      } catch (e) {
        console.warn('[WS] bad message', e);
      }
    });

    this.ws.addEventListener('close', () => {
      console.debug('[WS] disconnected — retry in 2s');
      clearInterval(this._pingTimer);
      setTimeout(() => this._connect(), 2000);
    });

    this.ws.addEventListener('error', () => {/* close will fire */});
  }

  _startPing() {
    clearInterval(this._pingTimer);
    this._pingTimer = setInterval(() => this.send({ type: 'ping' }), 25_000);
  }
}
