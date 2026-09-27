"""Evaluate a LinCIR Phi checkpoint on the complete CIRR validation gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from lincir_core import Phi, encode_pseudo, cirr_metrics
from train import MODEL


@torch.no_grad()
def run(args):
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose another output file.')
    root = Path(args.dataset)
    records = json.loads((root / 'cirr/captions/cap.rc2.val.json').read_text(encoding='utf-8'))
    paths = json.loads((root / 'cirr/image_splits/split.rc2.val.json').read_text(encoding='utf-8'))
    names = list(paths)
    if not records or not names:
        raise ValueError('CIRR annotations/gallery are empty.')
    for name, path in paths.items():
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f'Image path escapes dataset directory: {path}')
        if not resolved.is_file():
            raise FileNotFoundError(f'Missing gallery image {name}: {resolved}')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model_name = state.get('model', MODEL)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device).eval()
    start = time.perf_counter()
    chunks = []
    for offset in range(0, len(names), args.batch_size):
        images = []
        for name in names[offset:offset + args.batch_size]:
            with Image.open(root / paths[name]) as im:
                images.append(im.convert('RGB'))
        pixels = processor(images=images, return_tensors='pt').pixel_values.to(device)
        chunks.append(image_encoder(pixel_values=pixels).image_embeds.float().cpu())
        if offset % (10 * args.batch_size) == 0:
            print(f'Gallery {min(offset + args.batch_size, len(names))}/{len(names)}', flush=True)
    gallery = torch.cat(chunks)
    del image_encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    text_encoder = CLIPTextModelWithProjection.from_pretrained(model_name).to(device).eval()
    phi = Phi(text_encoder.config.projection_dim, text_encoder.config.projection_dim * 4,
              text_encoder.config.hidden_size, 0.5).to(device).eval()
    phi.load_state_dict(state['Phi'])
    lookup = {name: i for i, name in enumerate(names)}
    placeholder = tokenizer.encode('$', add_special_tokens=False)
    if len(placeholder) != 1:
        raise ValueError('Tokenizer must represent $ as one token.')
    queries = []
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset:offset + args.batch_size]
        refs = gallery[[lookup[r['reference']] for r in batch]].to(device)
        prompts = ['a photo of $ that ' + r['caption'].replace('$', '') for r in batch]
        ids = tokenizer(prompts, padding='max_length', truncation=True,
                        max_length=text_encoder.config.max_position_embeddings, return_tensors='pt').input_ids.to(device)
        queries.append(encode_pseudo(text_encoder, ids, phi(refs), placeholder[0]).float().cpu())
    metrics = cirr_metrics(torch.cat(queries), gallery, names, records, args.batch_size)
    with Path(args.checkpoint).open('rb') as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {**vars(args), 'model': model_name, 'split': 'CIRR-val',
              'queries': len(records), 'gallery_size': len(names),
              'checkpoint_sha256': checkpoint_sha256, 'metrics': metrics,
              'elapsed_seconds': time.perf_counter() - start,
              'note': 'Validation/development metrics; not paper test-set results.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
