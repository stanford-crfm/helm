"use strict";
var v = Object.defineProperty;
var N = Object.getOwnPropertyDescriptor;
var j = Object.getOwnPropertyNames;
var z = Object.prototype.hasOwnProperty;
var P = (s, i, t) => i in s ? v(s, i, { enumerable: !0, configurable: !0, writable: !0, value: t }) : s[i] = t;
var q = (s, i) => {
  for (var t in i)
    v(s, t, { get: i[t], enumerable: !0 });
}, V = (s, i, t, e) => {
  if (i && typeof i == "object" || typeof i == "function")
    for (let n of j(i))
      !z.call(s, n) && n !== t && v(s, n, { get: () => i[n], enumerable: !(e = N(i, n)) || e.enumerable });
  return s;
};
var C = (s) => V(v({}, "__esModule", { value: !0 }), s);
var r = (s, i, t) => (P(s, typeof i != "symbol" ? i + "" : i, t), t);

// src/index.ts
var W = {};
q(W, {
  Bench: () => u,
  Task: () => l,
  default: () => U,
  hrtimeNow: () => R,
  now: () => w
});
module.exports = C(W);

// src/event.ts
function a(s, i = null) {
  let t = new Event(s);
  return Object.defineProperty(t, "task", {
    value: i,
    enumerable: !0,
    writable: !1,
    configurable: !1
  }), t;
}

// src/constants.ts
var $ = {
  1: 12.71,
  2: 4.303,
  3: 3.182,
  4: 2.776,
  5: 2.571,
  6: 2.447,
  7: 2.365,
  8: 2.306,
  9: 2.262,
  10: 2.228,
  11: 2.201,
  12: 2.179,
  13: 2.16,
  14: 2.145,
  15: 2.131,
  16: 2.12,
  17: 2.11,
  18: 2.101,
  19: 2.093,
  20: 2.086,
  21: 2.08,
  22: 2.074,
  23: 2.069,
  24: 2.064,
  25: 2.06,
  26: 2.056,
  27: 2.052,
  28: 2.048,
  29: 2.045,
  30: 2.042,
  31: 2.0399,
  32: 2.0378,
  33: 2.0357,
  34: 2.0336,
  35: 2.0315,
  36: 2.0294,
  37: 2.0273,
  38: 2.0252,
  39: 2.0231,
  40: 2.021,
  41: 2.0198,
  42: 2.0186,
  43: 2.0174,
  44: 2.0162,
  45: 2.015,
  46: 2.0138,
  47: 2.0126,
  48: 2.0114,
  49: 2.0102,
  50: 2.009,
  51: 2.0081,
  52: 2.0072,
  53: 2.0063,
  54: 2.0054,
  55: 2.0045,
  56: 2.0036,
  57: 2.0027,
  58: 2.0018,
  59: 2.0009,
  60: 2,
  61: 1.9995,
  62: 1.999,
  63: 1.9985,
  64: 1.998,
  65: 1.9975,
  66: 1.997,
  67: 1.9965,
  68: 1.996,
  69: 1.9955,
  70: 1.995,
  71: 1.9945,
  72: 1.994,
  73: 1.9935,
  74: 1.993,
  75: 1.9925,
  76: 1.992,
  77: 1.9915,
  78: 1.991,
  79: 1.9905,
  80: 1.99,
  81: 1.9897,
  82: 1.9894,
  83: 1.9891,
  84: 1.9888,
  85: 1.9885,
  86: 1.9882,
  87: 1.9879,
  88: 1.9876,
  89: 1.9873,
  90: 1.987,
  91: 1.9867,
  92: 1.9864,
  93: 1.9861,
  94: 1.9858,
  95: 1.9855,
  96: 1.9852,
  97: 1.9849,
  98: 1.9846,
  99: 1.9843,
  100: 1.984,
  101: 1.9838,
  102: 1.9836,
  103: 1.9834,
  104: 1.9832,
  105: 1.983,
  106: 1.9828,
  107: 1.9826,
  108: 1.9824,
  109: 1.9822,
  110: 1.982,
  111: 1.9818,
  112: 1.9816,
  113: 1.9814,
  114: 1.9812,
  115: 1.9819,
  116: 1.9808,
  117: 1.9806,
  118: 1.9804,
  119: 1.9802,
  120: 1.98,
  infinity: 1.96
}, b = $;

// src/utils.ts
var D = (s) => s / 1e6, R = () => D(Number(process.hrtime.bigint())), w = () => performance.now();
function G(s) {
  return s !== null && typeof s == "object" && typeof s.then == "function";
}
var B = (s) => s.reduce((i, t) => i + t, 0) / s.length || 0, L = (s, i) => s.reduce((e, n) => e + (n - i) ** 2, 0) / (s.length - 1) || 0, J = (async () => {
}).constructor, Q = (s) => s.constructor === J, T = async (s) => {
  if (Q(s.fn))
    return !0;
  try {
    if (s.opts.beforeEach != null)
      try {
        await s.opts.beforeEach.call(s);
      } catch (e) {
      }
    let i = s.fn(), t = G(i);
    if (t)
      try {
        await i;
      } catch (e) {
      }
    if (s.opts.afterEach != null)
      try {
        await s.opts.afterEach.call(s);
      } catch (e) {
      }
    return t;
  } catch (i) {
    return !1;
  }
};

// src/task.ts
var l = class extends EventTarget {
  constructor(t, e, n, h = {}) {
    super();
    r(this, "bench");
    r(this, "name");
    r(this, "fn");
    r(this, "runs", 0);
    r(this, "result");
    r(this, "opts");
    this.bench = t, this.name = e, this.fn = n, this.opts = h;
  }
  async run() {
    var h, c, m, f;
    this.dispatchEvent(a("start", this));
    let t = 0, e = [];
    if (await this.bench.setup(this, "run"), this.opts.beforeAll != null)
      try {
        await this.opts.beforeAll.call(this);
      } catch (o) {
        this.setResult({ error: o });
      }
    let n = await T(this);
    try {
      for (; (t < this.bench.time || this.runs < this.bench.iterations) && !((h = this.bench.signal) != null && h.aborted); ) {
        this.opts.beforeEach != null && await this.opts.beforeEach.call(this);
        let o = 0;
        if (n) {
          let p = this.bench.now();
          await this.fn(), o = this.bench.now() - p;
        } else {
          let p = this.bench.now();
          this.fn(), o = this.bench.now() - p;
        }
        e.push(o), this.runs += 1, t += o, this.opts.afterEach != null && await this.opts.afterEach.call(this);
      }
    } catch (o) {
      this.setResult({ error: o });
    }
    if (this.opts.afterAll != null)
      try {
        await this.opts.afterAll.call(this);
      } catch (o) {
        this.setResult({ error: o });
      }
    if (await this.bench.teardown(this, "run"), e.sort((o, p) => o - p), !((c = this.result) != null && c.error)) {
      let o = e[0], p = e[e.length - 1], g = t / this.runs, O = 1e3 / g, E = B(e), k = L(e, E), y = Math.sqrt(k), x = y / Math.sqrt(e.length), M = e.length - 1, A = b[String(Math.round(M) || 1)] || b.infinity, F = x * A, K = F / E * 100 || 0, _ = e[Math.ceil(e.length * (75 / 100)) - 1], S = e[Math.ceil(e.length * (99 / 100)) - 1], I = e[Math.ceil(e.length * (99.5 / 100)) - 1], H = e[Math.ceil(e.length * (99.9 / 100)) - 1];
      if ((m = this.bench.signal) != null && m.aborted)
        return this;
      this.setResult({
        totalTime: t,
        min: o,
        max: p,
        hz: O,
        period: g,
        samples: e,
        mean: E,
        variance: k,
        sd: y,
        sem: x,
        df: M,
        critical: A,
        moe: F,
        rme: K,
        p75: _,
        p99: S,
        p995: I,
        p999: H
      });
    }
    return (f = this.result) != null && f.error && (this.dispatchEvent(a("error", this)), this.bench.dispatchEvent(a("error", this))), this.dispatchEvent(a("cycle", this)), this.bench.dispatchEvent(a("cycle", this)), this.dispatchEvent(a("complete", this)), this;
  }
  async warmup() {
    var h;
    this.dispatchEvent(a("warmup", this));
    let t = this.bench.now(), e = 0;
    if (await this.bench.setup(this, "warmup"), this.opts.beforeAll != null)
      try {
        await this.opts.beforeAll.call(this);
      } catch (c) {
        this.setResult({ error: c });
      }
    let n = await T(this);
    for (; (e < this.bench.warmupTime || this.runs < this.bench.warmupIterations) && !((h = this.bench.signal) != null && h.aborted); ) {
      if (this.opts.beforeEach != null)
        try {
          await this.opts.beforeEach.call(this);
        } catch (c) {
          this.setResult({ error: c });
        }
      try {
        n ? await this.fn() : this.fn();
      } catch (c) {
      }
      if (this.runs += 1, e = this.bench.now() - t, this.opts.afterEach != null)
        try {
          await this.opts.afterEach.call(this);
        } catch (c) {
          this.setResult({ error: c });
        }
    }
    if (this.opts.afterAll != null)
      try {
        await this.opts.afterAll.call(this);
      } catch (c) {
        this.setResult({ error: c });
      }
    this.bench.teardown(this, "warmup"), this.runs = 0;
  }
  addEventListener(t, e, n) {
    super.addEventListener(t, e, n);
  }
  removeEventListener(t, e, n) {
    super.removeEventListener(t, e, n);
  }
  setResult(t) {
    this.result = { ...this.result, ...t }, Object.freeze(this.reset);
  }
  reset() {
    this.dispatchEvent(a("reset", this)), this.runs = 0, this.result = void 0;
  }
};

// src/bench.ts
var u = class extends EventTarget {
  constructor(t = {}) {
    var e, n, h, c, m, f, o;
    super();
    r(this, "_tasks", /* @__PURE__ */ new Map());
    r(this, "_todos", /* @__PURE__ */ new Map());
    r(this, "signal");
    r(this, "warmupTime", 100);
    r(this, "warmupIterations", 5);
    r(this, "time", 500);
    r(this, "iterations", 10);
    r(this, "now", w);
    r(this, "setup");
    r(this, "teardown");
    this.now = (e = t.now) != null ? e : this.now, this.warmupTime = (n = t.warmupTime) != null ? n : this.warmupTime, this.warmupIterations = (h = t.warmupIterations) != null ? h : this.warmupIterations, this.time = (c = t.time) != null ? c : this.time, this.iterations = (m = t.iterations) != null ? m : this.iterations, this.signal = t.signal, this.setup = (f = t.setup) != null ? f : () => {
    }, this.teardown = (o = t.teardown) != null ? o : () => {
    }, this.signal && this.signal.addEventListener(
      "abort",
      () => {
        this.dispatchEvent(a("abort"));
      },
      { once: !0 }
    );
  }
  async run() {
    var e;
    this.dispatchEvent(a("start"));
    let t = [];
    for (let n of [...this._tasks.values()])
      (e = this.signal) != null && e.aborted ? t.push(n) : t.push(await n.run());
    return this.dispatchEvent(a("complete")), t;
  }
  async warmup() {
    this.dispatchEvent(a("warmup"));
    for (let [, t] of this._tasks)
      await t.warmup();
  }
  reset() {
    this.dispatchEvent(a("reset")), this._tasks.forEach((t) => {
      t.reset();
    });
  }
  add(t, e, n = {}) {
    let h = new l(this, t, e, n);
    return this._tasks.set(t, h), this.dispatchEvent(a("add", h)), this;
  }
  todo(t, e = () => {
  }, n = {}) {
    let h = new l(this, t, e, n);
    return this._todos.set(t, h), this.dispatchEvent(a("todo", h)), this;
  }
  remove(t) {
    let e = this.getTask(t);
    return this.dispatchEvent(a("remove", e)), this._tasks.delete(t), this;
  }
  addEventListener(t, e, n) {
    super.addEventListener(t, e, n);
  }
  removeEventListener(t, e, n) {
    super.removeEventListener(t, e, n);
  }
  table() {
    return this.tasks.map(({ name: t, result: e }) => e ? {
      "Task Name": t,
      "ops/sec": parseInt(e.hz.toString(), 10).toLocaleString(),
      "Average Time (ns)": e.mean * 1e3 * 1e3,
      Margin: `\xB1${e.rme.toFixed(2)}%`,
      Samples: e.samples.length
    } : null);
  }
  get results() {
    return [...this._tasks.values()].map((t) => t.result);
  }
  get tasks() {
    return [...this._tasks.values()];
  }
  get todos() {
    return [...this._todos.values()];
  }
  getTask(t) {
    return this._tasks.get(t);
  }
};

// src/index.ts
var U = u;
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  Bench,
  Task,
  hrtimeNow,
  now
});
