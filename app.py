import json
import os
import queue
import threading
import time

import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from dataset import create_dataloaders
from model import TransformerNMT
from trainer import Trainer

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

training_events: queue.Queue = queue.Queue()
trainer_instance: Trainer | None = None
train_thread: threading.Thread | None = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sse_callback(data: dict):
    training_events.put(data)


@app.route('/')
def index():
    return render_template('train.html')


@app.route('/train')
def train_page():
    return render_template('train.html')


@app.route('/translate')
def translate_page():
    return render_template('translate.html')


@app.route('/api/events')
def events():
    def generate():
        # 新连接时注入全量状态快照，解决刷新页面后进度丢失问题
        if trainer_instance is not None and trainer_instance.state.running:
            # 先清空积压在队列中的旧事件，避免 state_sync 后面跟一堆过期事件
            while not training_events.empty():
                try:
                    training_events.get_nowait()
                except queue.Empty:
                    break

            s = trainer_instance.state
            m = trainer_instance.model
            elapsed = time.time() - s.start_time if s.start_time else 0
            sync_data = {
                'type': 'state_sync',
                'running': True,
                'current_epoch': s.current_epoch,
                'total_epochs': s.total_epochs,
                'current_batch': s.current_batch,
                'total_batches_current': s.total_batches_current,
                'train_losses': s.train_losses,
                'val_losses': s.val_losses,
                'bleu_scores': s.bleu_scores,
                'batch_losses': s.batch_losses[-100:] if s.batch_losses else [],
                'best_val_loss': s.best_val_loss,
                'best_bleu': s.best_bleu,
                'elapsed': elapsed,
                'device': str(device),
                'n_params': m.count_parameters(),
                'd_model': m.d_model,
                'nhead': m.nhead,
                'num_encoder_layers': m.num_encoder_layers,
                'num_decoder_layers': m.num_decoder_layers,
                'dim_feedforward': m.dim_feedforward,
                'dropout': m.dropout,
                'src_vocab_size': len(trainer_instance.src_vocab),
                'tgt_vocab_size': len(trainer_instance.tgt_vocab),
            }
            yield f"event: state_sync\ndata: {json.dumps(sync_data, ensure_ascii=False)}\n\n"

        while True:
            try:
                data = training_events.get(timeout=15)
                yield f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield f"event: heartbeat\ndata: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/start', methods=['POST'])
def start_training():
    global trainer_instance, train_thread

    if trainer_instance is not None and trainer_instance.state.running:
        return jsonify({'status': 'error', 'message': 'Training is already running.'}), 409

    config = request.get_json() or {}
    train_path = config.get('train_path', os.path.join(DATA_DIR, 'train.json'))
    val_path = config.get('val_path', os.path.join(DATA_DIR, 'validation.json'))
    batch_size = int(config.get('batch_size', 32))
    epochs = int(config.get('epochs', 10))
    lr = float(config.get('lr', 0.0005))
    max_samples = config.get('max_samples', None)
    max_len = int(config.get('max_len', 64))
    d_model = int(config.get('d_model', 256))
    nhead = int(config.get('nhead', 8))
    num_encoder_layers = int(config.get('num_encoder_layers', 3))
    num_decoder_layers = int(config.get('num_decoder_layers', 3))
    dim_feedforward = int(config.get('dim_feedforward', 512))
    dropout = float(config.get('dropout', 0.1))
    label_smoothing = float(config.get('label_smoothing', 0.1))
    grad_clip = float(config.get('grad_clip', 1.0))
    src_max_vocab = int(config.get('src_max_vocab', 30000))
    tgt_max_vocab = int(config.get('tgt_max_vocab', 12000))

    if max_samples is not None:
        max_samples = int(max_samples)

    try:
        training_events.put({
            'type': 'status',
            'message': 'Building vocabulary and loading data...',
            'stage': 'prepare',
        })

        train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
            train_path=train_path,
            val_path=val_path,
            batch_size=batch_size,
            max_len=max_len,
            max_samples=max_samples,
            src_max_vocab=src_max_vocab,
            tgt_max_vocab=tgt_max_vocab,
        )

        model = TransformerNMT(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        n_params = model.count_parameters()
        training_events.put({
            'type': 'status',
            'message': f'Model created: {n_params:,} parameters. Starting training on {device}...',
            'stage': 'init',
            'n_params': n_params,
            'device': str(device),
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
        })

        trainer_instance = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            lr=lr,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        trainer_instance.on_event(sse_callback)

        def run():
            trainer_instance.train(epochs=epochs)

        train_thread = threading.Thread(target=run, daemon=True)
        train_thread.start()

        return jsonify({'status': 'ok', 'message': 'Training started.'})

    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'message': f'Data file not found: {e}'}), 404
    except Exception as e:
        training_events.put({'type': 'error', 'message': str(e)})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_training():
    global trainer_instance
    if trainer_instance is None or not trainer_instance.state.running:
        return jsonify({'status': 'ok', 'message': 'No training running.'})
    trainer_instance.stop()
    training_events.put({'type': 'status', 'message': 'Stopping training...', 'stage': 'stopping'})
    return jsonify({'status': 'ok', 'message': 'Stopping...'})


@app.route('/api/translate', methods=['POST'])
def translate():
    global trainer_instance
    if trainer_instance is None:
        return jsonify({'status': 'error', 'message': 'No model loaded.'}), 400

    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided.'}), 400

    try:
        result = trainer_instance.translate_single(text)
        return jsonify({'status': 'ok', 'translation': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    global trainer_instance
    if trainer_instance is None:
        return jsonify({
            'running': False,
            'model_loaded': False,
            'device': str(device),
        })

    s = trainer_instance.state
    m = trainer_instance.model
    status_data = {
        'running': s.running,
        'model_loaded': True,
        'current_epoch': s.current_epoch,
        'total_epochs': s.total_epochs,
        'current_batch': s.current_batch,
        'total_batches_current': s.total_batches_current,
        'train_losses': s.train_losses,
        'val_losses': s.val_losses,
        'bleu_scores': s.bleu_scores,
        'batch_losses': s.batch_losses[-100:] if s.batch_losses else [],
        'learning_rates': s.learning_rates,
        'best_val_loss': s.best_val_loss,
        'best_bleu': s.best_bleu,
        'elapsed': time.time() - s.start_time if s.start_time else 0,
        'device': str(device),
        'n_params': m.count_parameters(),
        'd_model': m.d_model,
        'nhead': m.nhead,
        'num_encoder_layers': m.num_encoder_layers,
        'num_decoder_layers': m.num_decoder_layers,
        'dim_feedforward': m.dim_feedforward,
        'dropout': m.dropout,
        'src_vocab_size': len(trainer_instance.src_vocab),
        'tgt_vocab_size': len(trainer_instance.tgt_vocab),
    }
    return jsonify(status_data)


@app.route('/api/load', methods=['POST'])
def load_model():
    global trainer_instance, train_thread
    data = request.get_json() or {}
    path = data.get('path', '')

    if not path or not os.path.exists(path):
        return jsonify({'status': 'error', 'message': 'Checkpoint not found.'}), 404

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        state_dict = checkpoint['model_state_dict']

        # 从检查点读取模型配置，自动匹配架构参数
        cfg = checkpoint.get('model_config', {})

        # d_model, max_len 优先从 cfg 取，缺失时从 state_dict 的 tensor 形状推断
        d_model = cfg.get('d_model')
        max_len = cfg.get('max_len')
        if d_model is None or max_len is None:
            pe = state_dict.get('pos_encoder.pe')
            src_emb = state_dict.get('src_embedding.weight')
            if d_model is None:
                d_model = src_emb.size(1) if src_emb is not None else (pe.size(2) if pe is not None else 256)
            if max_len is None:
                max_len = pe.size(1) if pe is not None else 512

        nhead = cfg.get('nhead', 8)
        num_encoder_layers = cfg.get('num_encoder_layers', 3)
        num_decoder_layers = cfg.get('num_decoder_layers', 3)
        dim_feedforward = cfg.get('dim_feedforward', 512)
        dropout = cfg.get('dropout', 0.1)

        model = TransformerNMT(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        model.load_state_dict(state_dict, strict=False)

        train_loader, val_loader, _, _ = create_dataloaders(
            train_path=os.path.join(DATA_DIR, 'train.json'),
            val_path=os.path.join(DATA_DIR, 'validation.json'),
            batch_size=1, max_samples=10,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        )

        trainer_instance = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab, device=device,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        if 'optimizer_state_dict' in checkpoint:
            trainer_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer_instance.on_event(sse_callback)
        trainer_instance.state.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epoch = checkpoint.get('epoch', 0)
        trainer_instance.state.train_losses = checkpoint.get('train_losses', [])
        trainer_instance.state.val_losses = checkpoint.get('val_losses', [])
        trainer_instance.state.bleu_scores = checkpoint.get('bleu_scores', [])
        if checkpoint.get('best_bleu', 0) is not None:
            trainer_instance.state.best_bleu = checkpoint.get('best_bleu', 0.0)

        training_events.put({
            'type': 'status',
            'message': f'Model loaded from epoch {epoch}. Ready for inference.',
            'stage': 'loaded',
            'n_params': model.count_parameters(),
            'd_model': model.d_model,
            'nhead': model.nhead,
            'num_encoder_layers': model.num_encoder_layers,
            'num_decoder_layers': model.num_decoder_layers,
            'dim_feedforward': model.dim_feedforward,
            'dropout': model.dropout,
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'device': str(device),
        })

        return jsonify({'status': 'ok', 'message': f'Loaded checkpoint from epoch {epoch}', 'epoch': epoch})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return jsonify([])
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')],
        key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)),
        reverse=True,
    )
    result = []
    for f in files:
        full = os.path.join(CHECKPOINT_DIR, f)
        result.append({
            'name': f,
            'path': full,
            'size_mb': round(os.path.getsize(full) / (1024 * 1024), 2),
        })
    return jsonify(result)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f'Device: {device}')
    print(f'Data dir: {DATA_DIR}')
    print(f'Checkpoint dir: {CHECKPOINT_DIR}')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
