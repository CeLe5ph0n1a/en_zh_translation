let lossChart, bleuChart, batchLossChart;
let trainingActive = false;
let allEpochs = [];
let allTrainLosses = [];
let allValLosses = [];
let allBleuScores = [];
let allBatchLosses = [];

function initCharts() {
    const darkOpts = {
        color: '#a1a1aa',
        borderColor: '#374151',
        grid: { color: '#272a36' },
    };

    lossChart = new Chart(document.getElementById('loss-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96,165,250,0.1)',
                    fill: true, tension: 0.3, pointRadius: 3,
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#f472b6',
                    backgroundColor: 'rgba(244,114,182,0.1)',
                    fill: true, tension: 0.3, pointRadius: 3,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Epoch' }, ...darkOpts },
                y: { title: { display: true, text: 'Loss' }, ...darkOpts },
            },
            plugins: { legend: { labels: { color: '#a1a1aa' } } },
        },
    });

    bleuChart = new Chart(document.getElementById('bleu-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BLEU',
                data: [],
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34,197,94,0.1)',
                fill: true, tension: 0.3, pointRadius: 3,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Epoch' }, ...darkOpts },
                y: { title: { display: true, text: 'BLEU %' }, ...darkOpts },
            },
            plugins: { legend: { labels: { color: '#a1a1aa' } } },
        },
    });

    batchLossChart = new Chart(document.getElementById('batch-loss-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Batch Loss',
                data: [],
                borderColor: '#f59e0b',
                borderWidth: 1, pointRadius: 0, tension: 0.3,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                x: { title: { display: true, text: 'Step' }, ...darkOpts },
                y: { title: { display: true, text: 'Loss' }, ...darkOpts },
            },
            plugins: { legend: { labels: { color: '#a1a1aa' } } },
        },
    });
}

function log(msg, cls = 'info') {
    const c = document.getElementById('log-console');
    const d = new Date();
    const ts = d.toTimeString().slice(0, 8);
    c.innerHTML += `<div class="log-line ${cls}">[${ts}] ${msg}</div>`;
    c.scrollTop = c.scrollHeight;
}

async function startTraining() {
    if (trainingActive) return;
    trainingActive = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;

    const config = {
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        lr: parseFloat(document.getElementById('lr').value),
        max_len: parseInt(document.getElementById('max_len').value),
        d_model: parseInt(document.getElementById('d_model').value),
        nhead: parseInt(document.getElementById('nhead').value),
        num_encoder_layers: parseInt(document.getElementById('num_encoder_layers').value),
        num_decoder_layers: parseInt(document.getElementById('num_decoder_layers').value),
        dim_feedforward: parseInt(document.getElementById('dim_feedforward').value),
        dropout: parseFloat(document.getElementById('dropout').value),
        label_smoothing: parseFloat(document.getElementById('label_smoothing').value),
        grad_clip: parseFloat(document.getElementById('grad_clip').value),
    };
    const ms = document.getElementById('max_samples').value;
    if (ms) config.max_samples = parseInt(ms);

    log(`Starting: ${config.epochs} epochs, batch=${config.batch_size}, d_model=${config.d_model}`, 'highlight');

    allEpochs = []; allTrainLosses = []; allValLosses = []; allBleuScores = []; allBatchLosses = [];
    updateCharts();

    try {
        const r = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await r.json();
        if (data.status !== 'ok') {
            log(`Error: ${data.message}`, 'error');
            trainingActive = false;
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
        }
    } catch (e) {
        log(`Request failed: ${e}`, 'error');
        trainingActive = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
    }
}

async function stopTraining() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        log('Stopping...', 'warn');
    } catch (e) {
        log(`Stop failed: ${e}`, 'error');
    }
}

async function loadCheckpoint() {
    const sel = document.getElementById('checkpoint-select');
    const path = sel.value;
    if (!path) return;
    log(`Loading checkpoint: ${path}`, 'info');
    try {
        const r = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await r.json();
        log(data.message, data.status === 'ok' ? 'success' : 'error');
    } catch (e) {
        log(`Load failed: ${e}`, 'error');
    }
}

async function refreshCheckpoints() {
    try {
        const r = await fetch('/api/checkpoints');
        const data = await r.json();
        const sel = document.getElementById('checkpoint-select');
        sel.innerHTML = '<option value="">-- 选择检查点 --</option>';
        data.forEach(cp => {
            const opt = document.createElement('option');
            opt.value = cp.path;
            opt.textContent = `${cp.name} (${cp.size_mb}MB)`;
            sel.appendChild(opt);
        });
    } catch (e) { /* ignore */ }
}

function updateCharts() {
    if (lossChart) {
        lossChart.data.labels = allEpochs;
        lossChart.data.datasets[0].data = allTrainLosses;
        lossChart.data.datasets[1].data = allValLosses;
        lossChart.update('none');
    }
    if (bleuChart) {
        bleuChart.data.labels = allEpochs;
        bleuChart.data.datasets[0].data = allBleuScores;
        bleuChart.update('none');
    }
    if (batchLossChart) {
        const recent = allBatchLosses.slice(-200);
        batchLossChart.data.labels = recent.map((_, i) => i + Math.max(0, allBatchLosses.length - 200));
        batchLossChart.data.datasets[0].data = recent;
        batchLossChart.update('none');
    }
}

function connectSSE() {
    const es = new EventSource('/api/events');

    es.addEventListener('status', e => {
        const d = JSON.parse(e.data);
        log(d.message, 'info');
        if (d.stage === 'init' || d.stage === 'loaded') {
            document.getElementById('device-badge').textContent = d.device || '--';
            renderModelInfo(d);
        }
    });

    es.addEventListener('training_start', e => {
        const d = JSON.parse(e.data);
        log(`Training started: ${d.epochs} epochs, ${d.train_batches} batches`, 'success');
    });

    es.addEventListener('epoch_start', e => {
        const d = JSON.parse(e.data);
        document.getElementById('epoch-info').textContent =
            `Epoch ${d.epoch} / Batch 0/${d.total_batches}`;
    });

    es.addEventListener('batch_progress', e => {
        const d = JSON.parse(e.data);
        document.getElementById('epoch-info').textContent =
            `Epoch ${d.epoch} / Batch ${d.batch}/${d.total_batches}`;
        document.getElementById('batch-bar').style.width = `${d.progress * 100}%`;
        document.getElementById('batch-pct').textContent = `${Math.round(d.progress * 100)}%`;
        document.getElementById('metric-train-loss').textContent = d.avg_loss.toFixed(4);
        allBatchLosses.push(d.loss);
        if (allBatchLosses.length % 5 === 0) updateCharts();
    });

    es.addEventListener('epoch_end', e => {
        const d = JSON.parse(e.data);
        log(`Epoch ${d.epoch}/${d.total_epochs}: train_loss=${d.train_loss.toFixed(4)}, val_loss=${d.val_loss.toFixed(4)}, BLEU=${d.bleu.toFixed(2)}${d.is_best ? ' [BEST]' : ''}`, d.is_best ? 'success' : 'info');

        allEpochs.push(d.epoch); allTrainLosses.push(d.train_loss);
        allValLosses.push(d.val_loss); allBleuScores.push(d.bleu);

        document.getElementById('epoch-bar').style.width = `${(d.epoch / d.total_epochs * 100)}%`;
        document.getElementById('epoch-pct').textContent = `${Math.round(d.epoch / d.total_epochs * 100)}%`;
        document.getElementById('metric-train-loss').textContent = d.train_loss.toFixed(4);
        document.getElementById('metric-val-loss').textContent = d.val_loss.toFixed(4);
        document.getElementById('metric-bleu').textContent = d.bleu.toFixed(2);
        document.getElementById('metric-best-bleu').textContent = d.best_bleu.toFixed(2);
        document.getElementById('time-elapsed').textContent = formatTime(d.elapsed);

        updateCharts(); refreshCheckpoints();
    });

    es.addEventListener('training_complete', e => {
        const d = JSON.parse(e.data);
        log(`Training complete! ${d.total_epochs_completed} epochs. Best BLEU: ${d.best_bleu.toFixed(2)}`, 'success');
        log(`Final: train_loss=${d.final_train_loss.toFixed(4)}, val_loss=${d.final_val_loss.toFixed(4)}, time=${formatTime(d.elapsed)}`, 'success');
        trainingActive = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        refreshCheckpoints();
        renderModelInfo(d);
    });

    es.addEventListener('error', e => {
        try {
            const d = JSON.parse(e.data);
            log(`ERROR: ${d.message}`, 'error');
        } catch (_) { log(`SSE error`, 'error'); }
        trainingActive = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
    });

    es.addEventListener('state_sync', e => {
        const d = JSON.parse(e.data);
        restoreFromState(d);
        log('已恢复训练状态', 'highlight');
    });

    es.addEventListener('checkpoint', e => {
        const d = JSON.parse(e.data);
        if (d.is_best) log(`Best checkpoint saved: epoch ${d.epoch}`, 'success');
    });

    es.onerror = () => { log('SSE connection lost, reconnecting...', 'warn'); };
}

function formatTime(s) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
}

function restoreFromState(d) {
    trainingActive = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('device-badge').textContent = d.device || '--';
    renderModelInfo(d);

    allEpochs = []; allTrainLosses = []; allValLosses = []; allBleuScores = [];
    if (d.train_losses && d.train_losses.length > 0) {
        const n = d.train_losses.length;
        for (let i = 0; i < n; i++) allEpochs.push(i + 1);
        allTrainLosses = [...d.train_losses];
        allValLosses = d.val_losses ? [...d.val_losses] : [];
        allBleuScores = d.bleu_scores ? [...d.bleu_scores] : [];
    }
    allBatchLosses = d.batch_losses ? [...d.batch_losses] : [];
    updateCharts();

    const totalEpochs = d.total_epochs || 1;
    const currentEpoch = d.current_epoch || 0;
    document.getElementById('epoch-bar').style.width = `${Math.round(currentEpoch / totalEpochs * 100)}%`;
    document.getElementById('epoch-pct').textContent = `${Math.round(currentEpoch / totalEpochs * 100)}%`;

    const currentBatch = d.current_batch || 0;
    const totalBatches = d.total_batches_current || 1;
    document.getElementById('batch-bar').style.width = `${Math.round(currentBatch / totalBatches * 100)}%`;
    document.getElementById('batch-pct').textContent = `${Math.round(currentBatch / totalBatches * 100)}%`;
    document.getElementById('epoch-info').textContent = `Epoch ${currentEpoch} / Batch ${currentBatch}/${totalBatches}`;

    if (allTrainLosses.length > 0) document.getElementById('metric-train-loss').textContent = allTrainLosses[allTrainLosses.length - 1].toFixed(4);
    if (allValLosses.length > 0) document.getElementById('metric-val-loss').textContent = allValLosses[allValLosses.length - 1].toFixed(4);
    if (allBleuScores.length > 0) document.getElementById('metric-bleu').textContent = allBleuScores[allBleuScores.length - 1].toFixed(2);
    if (d.best_bleu !== undefined) document.getElementById('metric-best-bleu').textContent = d.best_bleu.toFixed(2);
    if (d.elapsed !== undefined) document.getElementById('time-elapsed').textContent = formatTime(d.elapsed);
    updateCharts();
}

function renderModelInfo(d) {
    let html = '';
    if (d.n_params !== undefined) html += `<b>Params:</b> ${d.n_params?.toLocaleString()}<br>`;
    if (d.d_model !== undefined) html += `<b>d_model:</b> ${d.d_model}<br>`;
    if (d.nhead !== undefined) html += `<b>nhead:</b> ${d.nhead}<br>`;
    if (d.num_encoder_layers !== undefined) html += `<b>Enc Layers:</b> ${d.num_encoder_layers}<br>`;
    if (d.num_decoder_layers !== undefined) html += `<b>Dec Layers:</b> ${d.num_decoder_layers}<br>`;
    if (d.dim_feedforward !== undefined) html += `<b>FFN Dim:</b> ${d.dim_feedforward}<br>`;
    if (d.dropout !== undefined) html += `<b>Dropout:</b> ${d.dropout}<br>`;
    if (d.src_vocab_size !== undefined) html += `<b>Src Vocab:</b> ${d.src_vocab_size?.toLocaleString()}<br>`;
    if (d.tgt_vocab_size !== undefined) html += `<b>Tgt Vocab:</b> ${d.tgt_vocab_size?.toLocaleString()}<br>`;
    if (d.device !== undefined) html += `<b>Device:</b> ${d.device}`;
    if (!html) html = '未加载模型';
    document.getElementById('model-info').innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectSSE();
    refreshCheckpoints();
    fetch('/api/status').then(r => r.json()).then(d => {
        document.getElementById('device-badge').textContent = d.device || '--';
        if (d.model_loaded) renderModelInfo(d);
        if (d.running && !trainingActive) {
            restoreFromState(d);
            log('已恢复训练状态（页面加载）', 'highlight');
        }
    }).catch(() => {});
});
