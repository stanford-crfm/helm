function assert(condition, message) {
  if (!condition) {
    throw message || "Assertion failed";
  }
}

function encodeUrlParams(params) {
  let s = "";
  for (let k in params) {
    if (params[k] != null) {
      s += (s === "" ? "?" : "&") + k + "=" + encodeURIComponent(params[k]);
    }
  }
  return s;
}

function decodeUrlParams(str) {
  const params = {};
  if (str === "") return params;
  const items = str.substring(1).split(/&/);
  for (let i = 0; i < items.length; i++) {
    const pair = items[i].split(/=/);
    params[pair[0]] = decodeURIComponent(pair[1]);
  }
  return params;
}

function updateBrowserLocation(params) {
  // Update the address bar
  window.history.pushState(
    {},
    "",
    window.location.pathname + encodeUrlParams(params),
  );
}

function createCookie(key, value, days) {
  let expires = "";
  if (days) {
    const date = new Date();
    date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
    expires = "; expires=" + date.toUTCString();
  }
  document.cookie = key + "=" + value + expires + "; path=/";
}

function readCookie(key) {
  let tokens = document.cookie.split(";");
  for (let i = 0; i < tokens.length; i++) {
    const [k, v] = tokens[i].trim().split("=", 2);
    if (key === k) return v;
  }
  return null;
}

function eraseCookie(key) {
  createCookie(key, "", -1);
}

function renderTimestamp(timestamp) {
  if (!timestamp) return null;
  const d = new Date(timestamp * 1000);
  return d.toLocaleString();
}

function renderDict(data) {
  return JSON.stringify(data).substring(0, 10000);
}

function loadScript(src, onload, onerror) {
  // Using jquery doesn't work, so do it in with our bare hands.
  const s = document.createElement("script");
  s.src = src;
  s.onload = onload;
  s.onerror = onerror;
  document.head.appendChild(s);
}

function getRandomString() {
  const vocab =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 6; i++)
    text += vocab.charAt(Math.floor(Math.random() * vocab.length));
  return text;
}

function round(x, n) {
  const base = Math.pow(10, n);
  return Math.round(x * base) / base;
}

function multilineHtml(s) {
  return s.replace(/\n/g, "<br>");
}

function renderError(e) {
  return $("<div>").addClass("alert alert-danger").append(multilineHtml(e));
}

function helpIcon(help, link) {
  // Show a ?
  return $("<a>", { href: link, target: "blank_", class: "help-icon" }).append(
    $("<img>", { src: "info-icon.png", width: 15, title: help }),
  );
}

const markdownConverter = new showdown.Converter({ optionKey: "value" });
function renderMarkdown(markdown) {
  return markdown && markdownConverter.makeHtml(markdown);
}

function refreshHashLocation() {
  // If we request a hash location (URL contains #foo), the problem is #foo
  // might not exist (since it's generated).  Call this function to jump to the
  // hash location once all the anchors are generated.
  if (location.hash) {
    const hash = location.hash;
    location.hash = "";
    location.hash = hash;
  }
}
