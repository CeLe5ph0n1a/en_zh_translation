let translateHistory = [];

// ---------- 智能分句 ----------
const SENTENCE_RE = /([^.!?\n]+[.!?]+)(\s*)/g;

function splitSentences(paragraph) {
    const parts = [];
    let lastIdx = 0;
    let match;
    // 重置 lastIndex
    SENTENCE_RE.lastIndex = 0;
    while ((match = SENTENCE_RE.exec(paragraph)) !== null) {
        // 如果有前导空白，缓存下来
        if (match.index > lastIdx) {
            const prefix = paragraph.slice(lastIdx, match.index);
            if (parts.length > 0) {
                parts[parts.length - 1].suffix += prefix;
            }
        }
        parts.push({
            text: match[1],          // 句子正文（含标点）
            suffix: match[2],        // 句末空白
        });
        lastIdx = SENTENCE_RE.lastIndex;
    }
    // 尾部残留（无标点结尾的半截句子）
    if (lastIdx < paragraph.length) {
        const trail = paragraph.slice(lastIdx);
        if (trail.trim()) {
            parts.push({ text: trail, suffix: '' });
        } else if (parts.length > 0) {
            parts[parts.length - 1].suffix += trail;
        }
    }
    return parts;
}

function loadHistory() {
    try {
        const raw = localStorage.getItem('nmt_translate_history');
        if (raw) translateHistory = JSON.parse(raw);
    } catch (_) { translateHistory = []; }
}

function saveHistory() {
    localStorage.setItem('nmt_translate_history', JSON.stringify(translateHistory.slice(-50)));
}

function addHistory(src, tgt) {
    translateHistory.push({
        src, tgt,
        ts: new Date().toLocaleString('zh-CN'),
    });
    saveHistory();
    renderHistory();
}

function clearHistory() {
    translateHistory = [];
    saveHistory();
    renderHistory();
}

function renderHistory() {
    const el = document.getElementById('history-list');
    if (translateHistory.length === 0) {
        el.innerHTML = '<div class="empty-state"><div class="icon">&#128172;</div><div>暂无翻译记录</div></div>';
        return;
    }
    let html = '';
    // 倒序显示
    for (let i = translateHistory.length - 1; i >= 0; i--) {
        const h = translateHistory[i];
        html += `<div class="history-item">
            <span class="history-src">${esc(h.src)}</span>
            <span class="history-arrow">&rarr;</span>
            <span class="history-tgt">${esc(h.tgt)}</span>
            <span class="history-ts">${h.ts}</span>
        </div>`;
    }
    el.innerHTML = html;
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

async function doTranslate() {
    const input = document.getElementById('translate-input').value.trim();
    if (!input) return;
    const out = document.getElementById('translate-output');
    out.textContent = '翻译中...';
    out.className = 'translate-output';
    try {
        const r = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input }),
        });
        const data = await r.json();
        if (data.status === 'ok') {
            out.textContent = data.translation;
            addHistory(input, data.translation);
        } else {
            out.textContent = data.message;
            out.className = 'translate-output error';
        }
    } catch (e) {
        out.textContent = '翻译请求失败';
        out.className = 'translate-output error';
    }
}

async function doBatchTranslate() {
    const inputEl = document.getElementById('batch-input');
    const outputEl = document.getElementById('batch-output');
    const progressEl = document.getElementById('batch-progress');
    const rawText = inputEl.value;
    if (!rawText.trim()) return;

    // 按段落分行（空行作为段落分隔）
    const paragraphs = rawText.split('\n');
    // 收集所有待翻译的句子（扁平化）
    const allTasks = [];    // { paraIdx, sentIdx, text, suffix }
    const paraStruct = [];  // [{ sents: [{ text, suffix }], prefix }]  前缀空格等
    let currentParaSentences = [];
    let prefixBuffer = '';

    for (const line of paragraphs) {
        if (line.trim() === '') {
            // 空行 = 段落分隔
            if (currentParaSentences.length > 0 || prefixBuffer.trim()) {
                paraStruct.push({ sents: currentParaSentences, prefix: prefixBuffer });
                currentParaSentences = [];
                prefixBuffer = '';
            }
            // 空段落自身
            paraStruct.push({ sents: [], prefix: '' });
            continue;
        }
        const sents = splitSentences(line);
        if (sents.length === 0) {
            sents.push({ text: line, suffix: '' });
        }
        for (const s of sents) {
            allTasks.push({
                paraIdx: paraStruct.length,
                sentIdx: currentParaSentences.length,
                text: s.text.trimEnd(),
                suffix: s.suffix || '',
            });
            currentParaSentences.push({ text: '', suffix: s.suffix || '' });
        }
        prefixBuffer = '';
    }
    // 最后一个段落
    if (currentParaSentences.length > 0 || prefixBuffer.trim()) {
        paraStruct.push({ sents: currentParaSentences, prefix: prefixBuffer });
    }

    if (allTasks.length === 0) return;

    const totalSentences = allTasks.length;
    document.getElementById('btn-batch-translate').disabled = true;
    progressEl.textContent = `已拆分为 ${totalSentences} 个句子，翻译中... 0/${totalSentences}`;

    // 逐句翻译（批量但顺序调用）
    const results = new Array(totalSentences).fill('');
    for (let i = 0; i < totalSentences; i++) {
        const task = allTasks[i];
        try {
            const r = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: task.text }),
            });
            const data = await r.json();
            if (data.status === 'ok') {
                results[i] = data.translation;
                addHistory(task.text, data.translation);
            } else {
                results[i] = `【错误】${data.message}`;
            }
        } catch (e) {
            results[i] = '【错误】请求失败';
        }
        progressEl.textContent = `翻译中... ${i + 1}/${totalSentences} 句`;
    }

    // 重组：每个句子的翻译 + 后缀拼接回段落
    const outputLines = [];
    for (const para of paraStruct) {
        let line = '';
        for (let j = 0; j < para.sents.length; j++) {
            line += results.shift() + para.sents[j].suffix;
        }
        outputLines.push(line);
    }

    outputEl.value = outputLines.join('\n');
    progressEl.textContent = `完成 — 共 ${totalSentences} 句`;
    document.getElementById('btn-batch-translate').disabled = false;
    document.getElementById('btn-copy').disabled = false;
}

// 保留快速装载旧版逻辑作为备选（不按句子切分，仅按行翻译）
async function doBatchTranslateLineByLine() {
    const inputEl = document.getElementById('batch-input');
    const outputEl = document.getElementById('batch-output');
    const progressEl = document.getElementById('batch-progress');
    const lines = inputEl.value.split('\n').map(l => l.trim()).filter(l => l);
    if (lines.length === 0) return;

    document.getElementById('btn-batch-translate').disabled = true;
    outputEl.value = '';
    progressEl.textContent = `翻译中... 0/${lines.length}`;

    const results = [];
    for (let i = 0; i < lines.length; i++) {
        try {
            const r = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: lines[i] }),
            });
            const data = await r.json();
            if (data.status === 'ok') {
                results.push(data.translation);
                addHistory(lines[i], data.translation);
            } else {
                results.push(`[错误] ${data.message}`);
            }
        } catch (e) {
            results.push('[错误] 请求失败');
        }
        progressEl.textContent = `翻译中... ${i + 1}/${lines.length}`;
        outputEl.value = results.join('\n');
    }

    progressEl.textContent = `完成 — 共 ${lines.length} 行`;
    document.getElementById('btn-batch-translate').disabled = false;
    document.getElementById('btn-copy').disabled = false;
}

function clearBatch() {
    document.getElementById('batch-input').value = '';
    document.getElementById('batch-output').value = '';
    document.getElementById('batch-progress').textContent = '';
    document.getElementById('btn-copy').disabled = true;
}

async function copyBatchOutput() {
    const text = document.getElementById('batch-output').value;
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        const btn = document.getElementById('btn-copy');
        btn.textContent = '已复制!';
        setTimeout(() => { btn.textContent = '📋 复制结果'; }, 2000);
    } catch (e) {
        // fallback
        const ta = document.getElementById('batch-output');
        ta.select();
        document.execCommand('copy');
    }
}

function renderModelInfo(d) {
    if (!d || !d.model_loaded) {
        document.getElementById('model-info').innerHTML = '未加载模型 — 请先回到<a href="/train" style="color:#60a5fa;">训练页面</a>加载检查点';
        document.getElementById('model-card-panel').style.display = 'none';
        return;
    }
    document.getElementById('model-info').textContent = '模型已加载';
    document.getElementById('model-card-panel').style.display = '';

    const items = [];
    if (d.n_params !== undefined) items.push(`<b>参数量</b> ${d.n_params.toLocaleString()}`);
    if (d.d_model !== undefined) items.push(`<b>d_model</b> ${d.d_model}`);
    if (d.nhead !== undefined) items.push(`<b>nhead</b> ${d.nhead}`);
    if (d.num_encoder_layers !== undefined) items.push(`<b>编码器层数</b> ${d.num_encoder_layers}`);
    if (d.num_decoder_layers !== undefined) items.push(`<b>解码器层数</b> ${d.num_decoder_layers}`);
    if (d.dim_feedforward !== undefined) items.push(`<b>FFN 维度</b> ${d.dim_feedforward}`);
    if (d.dropout !== undefined) items.push(`<b>Dropout</b> ${d.dropout}`);
    if (d.src_vocab_size !== undefined) items.push(`<b>源词表</b> ${d.src_vocab_size.toLocaleString()}`);
    if (d.tgt_vocab_size !== undefined) items.push(`<b>目标词表</b> ${d.tgt_vocab_size.toLocaleString()}`);
    if (d.device !== undefined) items.push(`<b>设备</b> ${d.device}`);

    document.getElementById('model-card-grid').innerHTML =
        items.map(s => `<div class="info-item">${s}</div>`).join('');
}

async function checkModelStatus() {
    try {
        const r = await fetch('/api/status');
        return await r.json();
    } catch (_) { return { model_loaded: false }; }
}

async function refreshCheckpoints() {
    try {
        const r = await fetch('/api/checkpoints');
        const data = await r.json();
        const sel = document.getElementById('checkpoint-select');
        if (!sel) return;
        sel.innerHTML = '<option value="">-- 选择检查点 --</option>';
        data.forEach(cp => {
            const opt = document.createElement('option');
            opt.value = cp.path;
            opt.textContent = `${cp.name} (${cp.size_mb}MB)`;
            sel.appendChild(opt);
        });
    } catch (e) { /* ignore */ }
}

async function loadCheckpoint() {
    const sel = document.getElementById('checkpoint-select');
    const path = sel.value;
    if (!path) return;
    const statusEl = document.getElementById('model-load-status');
    if (statusEl) {
        statusEl.textContent = '加载中...';
        statusEl.className = 'info-text loading';
    }
    try {
        const r = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await r.json();
        if (data.status === 'ok') {
            if (statusEl) {
                statusEl.textContent = `已加载: epoch ${data.epoch}`;
                statusEl.className = 'info-text success';
            }
            // 刷新模型状态
            const s = await checkModelStatus();
            document.getElementById('device-badge').textContent = s.device || '--';
            renderModelInfo(s);
        } else {
            if (statusEl) {
                statusEl.textContent = `加载失败: ${data.message}`;
                statusEl.className = 'info-text error';
            }
        }
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = `请求失败: ${e}`;
            statusEl.className = 'info-text error';
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    loadHistory();
    renderHistory();
    refreshCheckpoints();

    const status = await checkModelStatus();
    document.getElementById('device-badge').textContent = status.device || '--';
    renderModelInfo(status);
});

document.getElementById('translate-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doTranslate();
});
