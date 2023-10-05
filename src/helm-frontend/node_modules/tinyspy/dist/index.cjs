"use strict";
var k = Object.defineProperty;
var C = Object.getOwnPropertyDescriptor;
var E = Object.getOwnPropertyNames;
var G = Object.prototype.hasOwnProperty;
var j = (e, t) => {
  for (var n in t)
    k(e, n, { get: t[n], enumerable: !0 });
}, D = (e, t, n, a) => {
  if (t && typeof t == "object" || typeof t == "function")
    for (let r of E(t))
      !G.call(e, r) && r !== n && k(e, r, { get: () => t[r], enumerable: !(a = C(t, r)) || a.enumerable });
  return e;
};
var F = (e) => D(k({}, "__esModule", { value: !0 }), e);

// src/index.ts
var B = {};
j(B, {
  createInternalSpy: () => I,
  getInternalState: () => T,
  internalSpyOn: () => K,
  restoreAll: () => z,
  spies: () => x,
  spy: () => _,
  spyOn: () => $
});
module.exports = F(B);

// src/utils.ts
function m(e, t) {
  if (!e)
    throw new Error(t);
}
function y(e, t) {
  return typeof t === e;
}
function b(e) {
  return e instanceof Promise;
}
function d(e, t, n) {
  Object.defineProperty(e, t, n);
}
function p(e, t, n) {
  Object.defineProperty(e, t, { value: n });
}

// src/constants.ts
var u = Symbol.for("tinyspy:spy");

// src/internal.ts
var x = /* @__PURE__ */ new Set(), q = (e) => {
  e.called = !1, e.callCount = 0, e.calls = [], e.results = [], e.next = [];
}, V = (e) => (d(e, u, { value: { reset: () => q(e[u]) } }), e[u]), T = (e) => e[u] || V(e);
function I(e) {
  m(y("function", e) || y("undefined", e), "cannot spy on a non-function value");
  let t = function(...a) {
    let r = T(t);
    r.called = !0, r.callCount++, r.calls.push(a);
    let i = r.next.shift();
    if (i) {
      r.results.push(i);
      let [s, l] = i;
      if (s === "ok")
        return l;
      throw l;
    }
    let o, c = "ok";
    if (r.impl)
      try {
        new.target ? o = Reflect.construct(r.impl, a, new.target) : o = r.impl.apply(this, a), c = "ok";
      } catch (s) {
        throw o = s, c = "error", r.results.push([c, s]), s;
      }
    let S = [c, o];
    if (b(o)) {
      let s = o.then((l) => S[1] = l).catch((l) => {
        throw S[0] = "error", S[1] = l, l;
      });
      Object.assign(s, o), o = s;
    }
    return r.results.push(S), o;
  };
  p(t, "_isMockFunction", !0), p(t, "length", e ? e.length : 0), p(t, "name", e && e.name || "spy");
  let n = T(t);
  return n.reset(), n.impl = e, t;
}
function v(e) {
  let t = T(e);
  d(e, "returns", {
    get: () => t.results.map(([, n]) => n)
  }), ["called", "callCount", "results", "calls", "reset", "impl"].forEach((n) => d(e, n, { get: () => t[n], set: (a) => t[n] = a })), p(e, "nextError", (n) => (t.next.push(["error", n]), t)), p(e, "nextResult", (n) => (t.next.push(["ok", n]), t));
}

// src/spy.ts
function _(e) {
  let t = I(e);
  return v(t), t;
}

// src/spyOn.ts
var P = (e, t) => Object.getOwnPropertyDescriptor(e, t);
function K(e, t, n) {
  m(!y("undefined", e), "spyOn could not find an object to spy upon"), m(y("object", e) || y("function", e), "cannot spyOn on a primitive value");
  let a = () => {
    if (!y("object", t))
      return [t, "value"];
    if ("getter" in t && "setter" in t)
      throw new Error("cannot spy on both getter and setter");
    if ("getter" in t)
      return [t.getter, "get"];
    if ("setter" in t)
      return [t.setter, "set"];
    throw new Error("specify getter or setter to spy on");
  }, [r, i] = a(), o = P(e, r), c = Object.getPrototypeOf(e), S = c && P(c, r), s = o || S;
  m(s || r in e, `${String(r)} does not exist`);
  let l = !1;
  i === "value" && s && !s.value && s.get && (i = "get", l = !0, n = s.get());
  let f;
  s ? f = s[i] : i !== "value" ? f = () => e[r] : f = e[r], n || (n = f);
  let R = I(n), O = (h) => {
    let { value: H, ...w } = s || {
      configurable: !0,
      writable: !0
    };
    i !== "value" && delete w.writable, w[i] = h, d(e, r, w);
  }, M = () => s ? d(e, r, s) : O(f), A = R[u];
  return p(A, "restore", M), p(A, "getOriginal", () => l ? f() : f), p(A, "willCall", (h) => (A.impl = h, R)), O(l ? () => R : R), x.add(R), R;
}
function $(e, t, n) {
  let a = K(e, t, n);
  return v(a), ["restore", "getOriginal", "willCall"].forEach((r) => {
    p(a, r, a[u][r]);
  }), a;
}

// src/restoreAll.ts
function z() {
  for (let e of x)
    e.restore();
  x.clear();
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  createInternalSpy,
  getInternalState,
  internalSpyOn,
  restoreAll,
  spies,
  spy,
  spyOn
});
