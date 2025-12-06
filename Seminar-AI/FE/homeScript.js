const API_BASE = 'http://127.0.0.1:8000';
const elText = document.getElementById('text');
const elModel = document.getElementById('model');
const btn = document.getElementById('btnAnalyze');
const btnH = document.getElementById('btnHistory');
const btnDownload = document.getElementById('btnDownload');
const srvStatus = document.getElementById('srvStatus');

const out = {
  root: document.getElementById('output'),
  text: document.getElementById('outText'),
  badge: document.getElementById('outBadge'),
  score: document.getElementById('outScore'),
  token: document.getElementById('outToken'),
  model: document.getElementById('outModel')
};
const historyTbody = document.querySelector('#historyTable tbody');
const charCount = document.getElementById('charCount');

let lastHistory = [];

// helper: map label to CSS class
function labelClass(label) {
  if (!label) return ['neutral', '--'];
  const l = label.toString().toLowerCase();
  if (l.includes('pos')) return ['positive', label];
  if (l.includes('neg')) return ['negative', label];
  return ['neutral', label];
}

function setStatus(text, ok = true) {
  srvStatus.textContent = text;
  srvStatus.style.color = ok ? '#e6fffa' : '#ffd7d7';
}

function updateCharCount() {
  const n = elText.value.length;
  charCount.textContent = `${n} ký tự`;
}
elText.addEventListener('input', updateCharCount);

async function analyze() {
  const text = elText.value.trim();
  if (!text) {
    alert('Nhập câu tiếng Việt để phân tích.');
    return;
  }
  if (text.length < 6) {
    alert('Câu quá ngắn.');
    return;
  }

  const model = elModel.value.trim();
  btn.disabled = true;
  btn.innerHTML = '<span class="loader"></span> Đang phân tích...';

  const fd = new FormData();
  fd.append('text', text);
  if (model) fd.append('model', model);

  try {
    const resp = await fetch(API_BASE + '/analyze', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(await resp.text());
    const j = await resp.json();
    renderResult(j);
    await loadHistory();
  } catch (err) {
    alert('Lỗi: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Phân tích';
  }
}

function renderResult(r) {
  out.root.style.display = 'flex';
  out.text.textContent = r.text;
  const [cls, labelText] = labelClass(r.label);
  out.badge.className = 'badge ' + cls;
  out.badge.textContent = labelText;
  out.score.textContent = Number(r.score).toFixed(3);
  out.token.textContent = 'Tokenized: ' + (r.tokenized_text || '');
  out.model.textContent = r.model || '';
}

async function loadHistory(limit = 30) {
  try {
    const resp = await fetch(API_BASE + `/history?limit=${limit}`);
    if (!resp.ok) throw new Error('failed');
    const arr = await resp.json();
    lastHistory = arr;
    historyTbody.innerHTML = '';

    arr.forEach(row => {
      const tr = document.createElement('tr');
      const [cls] = labelClass(row.label);
      tr.innerHTML = `
        <td>${row.id}</td>
        <td>${escapeHtml(row.text)}</td>
        <td>${escapeHtml(row.tokenized_text)}</td>
        <td><span class="badge ${cls}">${escapeHtml(row.label)}</span></td>
        <td>${Number(row.score).toFixed(3)}</td>
        <td>${escapeHtml(row.created_at)}</td>
      `;
      historyTbody.appendChild(tr);
    });
  } catch (e) {
    console.warn(e);
  }
}

function downloadCSV() {
  if (!lastHistory.length) return alert('Không có dữ liệu.');
  const header = ['id', 'text', 'label', 'score', 'model', 'created_at'];
  const rows = lastHistory.map(r =>
    header.map(h => `"${(r[h] + '').replace(/"/g, '""')}"`).join(',')
  );
  const csv = [header.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'vi_sentiment_history.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

elText.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
});

btn.addEventListener('click', analyze);
btnH.addEventListener('click', () => loadHistory(30));
btnDownload.addEventListener('click', downloadCSV);

loadHistory();
updateCharCount();
