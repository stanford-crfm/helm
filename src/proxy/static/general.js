function encodeUrlParams(params) {
  let s = '';
  for (let k in params)
    s += (s === '' ? '?' : '&') + k + '=' + encodeURIComponent(params[k]);
  return s;
}

function decodeUrlParams(str) {
  const params = {};
  if (str === '')
    return params;
  const items = str.substring(1).split(/&/);
  for (let i = 0; i < items.length; i++) {
    const pair = items[i].split(/=/);
    params[pair[0]] = decodeURIComponent(pair[1]);
  }
  return params;
}

function updateBrowserLocation(params) {
  // Update the address bar
  window.history.pushState({}, '', window.location.pathname + encodeUrlParams(params));
}

function renderTableText(header, rows, renderers) {
  const lines = [];
  // Header
  lines.push(header.join('\t'));
  // Contents
  rows.forEach((row, r) => {
    lines.push(header.map((x) => {
      const obj = renderers && renderers[x] ? renderers[x](row[x], r) : row[x];
      const t = typeof(obj);
      if (obj !== null && typeof(obj) === 'object')
        return obj.text();
      return obj;
    }).join('\t'));
  });
  return lines.join('\n') + '\n';
}

function renderTable(header, rows, renderers) {
  const $table = $('<table>').addClass('table');
  // Header
  $table.append($('<thead>').addClass('thead-default').append($('<tr>').append(header.map((x) => $('<th>').append(x)))));
  // Contents
  $table.append($('<tbody>').append(rows.map((row, r) => {
    return $('<tr>').append(header.map((x) => {
      return $('<td>').append(renderers && renderers[x] ? renderers[x](row[x], r) : row[x]);
    }));
  })));
  // Enable sorting columns when click on them
  $table.tablesorter();
  return $table;
}

function createCookie(key, value, days) {
  let expires = '';
  if (days) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    expires = "; expires=" + date.toUTCString();
  }
  document.cookie = key + "=" + value + expires + "; path=/";
}

function readCookie(key) {
  let tokens = document.cookie.split(';');
  for (let i = 0; i < tokens.length; i++) {
    const [k, v] = tokens[i].trim().split('=', 2);
    if (key === k)
      return v;
  }
  return null;
}

function eraseCookie(key) {
	createCookie(key, '' , -1);
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
  const s = document.createElement('script');
  s.src = src;
  s.onload = onload;
  s.onerror = onerror;
  document.head.appendChild(s);
}

function getRandomString() {
  const vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 6; i++)
    text += vocab.charAt(Math.floor(Math.random() * vocab.length));
  return text;
}
