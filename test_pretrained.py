"""Run the published LinCIR checkpoint on a real photo; this is NOT a benchmark.

Downloads pinned public model files (~1.8 GB) into output/baseline on first run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
import transformers

from lincir_core import Phi, encode_pseudo

PHI_REVISION = '5118df4683de6efa09f8e5336d86ca626ed21c44'
PHI_SHA256 = '5ba98d52db7a5e9a78f9ba7436991235477b89544caf246e4659642bf9b409bd'
CLIP_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/baseline/pretrained_smoke.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new report path.')
    photo = Path('third_party/lincir/example1.jpg')
    if not photo.is_file():
        raise FileNotFoundError('Run git submodule update --init --recursive first.')
    checkpoint = Path(hf_hub_download(
        'navervision/zeroshot-cir-models', 'lincir_large.pt', revision=PHI_REVISION,
        local_dir='output/baseline/checkpoint'))
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == PHI_SHA256, 'Published checkpoint checksum mismatch'
    model = snapshot_download(
        'openai/clip-vit-large-patch14', revision=CLIP_REVISION,
        allow_patterns=['*.json', 'merges.txt', 'vocab.json', 'model.safetensors'],
        local_dir='output/baseline/clip-vit-large-patch14')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    start = time.perf_counter()
    processor = CLIPImageProcessor.from_pretrained(model)
    vision = CLIPVisionModelWithProjection.from_pretrained(model).to(device).eval()
    with Image.open(photo) as image:
        pixels = processor(images=image.convert('RGB'), return_tensors='pt').pixel_values.to(device)
    image_features = vision(pixel_values=pixels).image_embeds
    del vision, pixels
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    text = CLIPTextModelWithProjection.from_pretrained(model).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model)
    phi = Phi(text.config.projection_dim, 4 * text.config.projection_dim,
              text.config.hidden_size, 0.5).to(device).eval()
    phi.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['Phi'])
    prompts = ['a photo of $ that is a pencil sketch', 'a photo of $ that is covered in snow']
    ids = tokenizer(prompts, padding='max_length', max_length=77, return_tensors='pt').input_ids.to(device)
    assert tokenizer.encode('$', add_special_tokens=False) == [259]
    pseudo = phi(image_features).expand(len(prompts), -1)
    composed = encode_pseudo(text, ids, pseudo)
    assert image_features.shape == (1, 768) and pseudo.shape == (2, 768) and composed.shape == (2, 768)
    assert all(torch.isfinite(t).all() for t in (image_features, pseudo, composed))
    difference = (composed[0] - composed[1]).norm().item()
    assert difference > 1e-4, 'Changing the modification text had no effect'
    result = {
        'status': 'PASS', 'kind': 'published_checkpoint_real_photo_smoke',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256': digest, 'checkpoint_revision': PHI_REVISION,
        'backbone': 'openai/clip-vit-large-patch14', 'backbone_revision': CLIP_REVISION,
        'photo': photo.as_posix(), 'photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
        'prompts': prompts, 'image_shape': list(image_features.shape),
        'pseudo_shape': list(pseudo.shape), 'composed_shape': list(composed.shape),
        'prompt_output_l2_difference': difference,
        'device': str(device), 'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'elapsed_seconds': time.perf_counter() - start,
        'note': 'Official Phi weights with project encode_pseudo; inference only. No CIRR metrics or training reproduction.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
