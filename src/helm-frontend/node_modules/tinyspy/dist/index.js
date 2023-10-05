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
var I = /* @__PURE__ */ new Set(), M = (e) => {
  e.called = !1, e.callCount = 0, e.calls = [], e.results = [], e.next = [];
}, C = (e) => (d(e, u, { value: { reset: () => M(e[u]) } }), e[u]), v = (e) => e[u] || C(e);
function T(e) {
  m(y("function", e) || y("undefined", e), "cannot spy on a non-function value");
  let t = function(...a) {
    let r = v(t);
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
    let x = [c, o];
    if (b(o)) {
      let s = o.then((l) => x[1] = l).catch((l) => {
        throw x[0] = "error", x[1] = l, l;
      });
      Object.assign(s, o), o = s;
    }
    return r.results.push(x), o;
  };
  p(t, "_isMockFunction", !0), p(t, "length", e ? e.length : 0), p(t, "name", e && e.name || "spy");
  let n = v(t);
  return n.reset(), n.impl = e, t;
}
function h(e) {
  let t = v(e);
  d(e, "returns", {
    get: () => t.results.map(([, n]) => n)
  }), ["called", "callCount", "results", "calls", "reset", "impl"].forEach((n) => d(e, n, { get: () => t[n], set: (a) => t[n] = a })), p(e, "nextError", (n) => (t.next.push(["error", n]), t)), p(e, "nextResult", (n) => (t.next.push(["ok", n]), t));
}

// src/spy.ts
function z(e) {
  let t = T(e);
  return h(t), t;
}

// src/spyOn.ts
var P = (e, t) => Object.getOwnPropertyDescriptor(e, t);
function E(e, t, n) {
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
  }, [r, i] = a(), o = P(e, r), c = Object.getPrototypeOf(e), x = c && P(c, r), s = o || x;
  m(s || r in e, `${String(r)} does not exist`);
  let l = !1;
  i === "value" && s && !s.value && s.get && (i = "get", l = !0, n = s.get());
  let f;
  s ? f = s[i] : i !== "value" ? f = () => e[r] : f = e[r], n || (n = f);
  let S = T(n), O = (w) => {
    let { value: G, ...k } = s || {
      configurable: !0,
      writable: !0
    };
    i !== "value" && delete k.writable, k[i] = w, d(e, r, k);
  }, K = () => s ? d(e, r, s) : O(f), A = S[u];
  return p(A, "restore", K), p(A, "getOriginal", () => l ? f() : f), p(A, "willCall", (w) => (A.impl = w, S)), O(l ? () => S : S), I.add(S), S;
}
function W(e, t, n) {
  let a = E(e, t, n);
  return h(a), ["restore", "getOriginal", "willCall"].forEach((r) => {
    p(a, r, a[u][r]);
  }), a;
}

// src/restoreAll.ts
function Z() {
  for (let e of I)
    e.restore();
  I.clear();
}
export {
  T as createInternalSpy,
  v as getInternalState,
  E as internalSpyOn,
  Z as restoreAll,
  I as spies,
  z as spy,
  W as spyOn
};
