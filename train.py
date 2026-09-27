"""Single-device LinCIR training: frozen CLIP + official Phi + optional InfoNCE."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
import transformers

from lincir_core import Phi, encode_pseudo, smp_loss

MODEL = 'openai/clip-vit-large-patch14'


def load_tokens(path, tokenizer, max_length):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Caption file is empty.')
    if any(not isinstance(r.get('caption'), str) or not r['caption'].strip()
           or not isinstance(r.get('masked'), str) or '$' not in r['masked'] for r in records):
        raise ValueError('Each JSONL row needs a nonempty caption and masked text containing $.')
    captions = [r['caption'] for r in records]
    if len(set(captions)) != len(captions):
        raise ValueError('Duplicate captions: run prepare_captions.py before training.')
    if any('$' in caption for caption in captions):
        raise ValueError('Original captions must not contain the pseudo-token $.')
    options = dict(padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    original = tokenizer(captions, **options).input_ids
    masked = tokenizer([r['masked'] for r in records], **options).input_ids
    return original, masked


def run(args):
    if min(args.steps, args.batch_size, args.accumulation, args.save_every) < 1:
        raise ValueError('Steps, batch size, accumulation and save frequency must be positive.')
    if args.contrastive_weight and args.batch_size < 2:
        raise ValueError('InfoNCE requires batch size >= 2.')
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError('Learning rate must be finite and positive.')
    # Validate loss options before any model download.
    smp_loss(torch.eye(2), torch.eye(2), args.contrastive_weight, args.temperature)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a new run directory.')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.precision == 'fp16' and device.type != 'cuda':
        raise ValueError('fp16 requires CUDA; use --precision fp32 on CPU.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    placeholder = tokenizer.encode('$', add_special_tokens=False)
    if len(placeholder) != 1:
        raise ValueError('Tokenizer must represent $ as one token.')
    encoder = CLIPTextModelWithProjection.from_pretrained(args.model).to(device).eval().requires_grad_(False)
    if args.gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # HF checkpoint layers activate only in training mode. CLIP's configured dropout is zero.
        if encoder.config.attention_dropout != 0:
            raise ValueError('Checkpointing recipe requires CLIP attention_dropout=0.')
        encoder.train()
    original, masked = load_tokens(args.captions, tokenizer, encoder.config.max_position_embeddings)
    if len(original) < args.batch_size or not masked.eq(placeholder[0]).any(dim=1).all():
        raise ValueError('Too few captions for a batch, or a masked caption lost its pseudo-token.')
    loader = DataLoader(TensorDataset(original, masked), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    phi = Phi(encoder.config.projection_dim, encoder.config.projection_dim * 4,
              encoder.config.hidden_size, 0.5).to(device).train()
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if state.get('model', args.model) != args.model:
            raise ValueError('Initial checkpoint uses a different backbone.')
        phi.load_state_dict(state['Phi'])
    optimizer = torch.optim.AdamW(phi.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.captions).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    config = {**vars(args), 'captions_sha256': digest, 'captions_count': len(original),
              'effective_batch_size': args.batch_size * args.accumulation,
              'contrastive_batch_size': args.batch_size, 'torch': str(torch.__version__),
              'transformers': transformers.__version__, 'device_used': str(device),
              'upstream_commit': '1dec42d118da816be2a43fd43cc0746e05f63881'}
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device)
        for _ in range(args.accumulation):
            try:
                ids, masked_ids = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                ids, masked_ids = next(iterator)
            ids, masked_ids = ids.to(device), masked_ids.to(device)
            context = torch.autocast('cuda', dtype=torch.float16) if args.precision == 'fp16' else nullcontext()
            with context:
                with torch.no_grad():
                    target = encoder(input_ids=ids).text_embeds
                    noisy = target.float() + torch.rand(len(ids), 1, device=device) * torch.randn(target.shape, device=device)
                predicted = encode_pseudo(encoder, masked_ids, phi(noisy), placeholder[0])
                loss, mse, nce = smp_loss(predicted, target, args.contrastive_weight, args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}.')
            scaler.scale(loss / args.accumulation).backward()
            totals += torch.stack([loss.detach(), mse.detach(), nce.detach()]) / args.accumulation
        scaler.unscale_(optimizer)
        # Abort instead of silently counting a skipped fp16 update as a training step.
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in phi.parameters()):
            raise FloatingPointError('Non-finite Phi gradient; retry with --precision fp32.')
        scaler.step(optimizer)
        scaler.update()
        values = totals.tolist()
        record = dict(step=step, loss=values[0], mse=values[1], contrastive=values[2],
                      elapsed_seconds=time.perf_counter() - start)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            state = {'Phi': {k: v.detach().cpu() for k, v in phi.state_dict().items()},
                     'step': step, 'model': args.model, 'config': config}
            checkpoint = output / f'phi_{step:06d}.pt'
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save(state, temporary)
            temporary.replace(checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulation', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--contrastive-weight', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--device')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--init-checkpoint', help='Weights-only warm start; optimizer and data order restart.')
    args = parser.parse_args()
    print(f'Saved: {run(args)}')


if __name__ == '__main__':
    main()
