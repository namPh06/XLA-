"""Offline integration with a randomly initialized TINY CLIP. Not a quality benchmark."""
import argparse
import json
from pathlib import Path
import string
import tempfile

from PIL import Image
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer, CLIPImageProcessor

import evaluate
import train
from lincir_core import placeholder_token_id


def main():
    torch.set_num_threads(2)
    torch.manual_seed(12345)
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lincir-check-', dir='tmp') as temporary:
        root = Path(temporary).resolve()
        model_dir = root / 'tiny-clip'
        model_dir.mkdir()
        chars = list(string.ascii_lowercase + '$.,')
        tokens = chars + [c + '</w>' for c in chars] + ['$.</w>', '<|startoftext|>', '<|endoftext|>']
        vocab = {token: index for index, token in enumerate(tokens)}
        (model_dir / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
        (model_dir / 'merges.txt').write_text('#version: 0.2\n$ .</w>\n', encoding='utf-8')
        # Positional paths work with both Transformers 4 (vocab_file/merges_file)
        # and Transformers 5 (vocab/merges).
        tokenizer = CLIPTokenizer(str(model_dir / 'vocab.json'),
                                  str(model_dir / 'merges.txt'), model_max_length=77)
        tokenizer.save_pretrained(model_dir)
        placeholder = placeholder_token_id(tokenizer)
        assert placeholder not in tokenizer.encode('$.', add_special_tokens=False)
        punctuation = root / 'punctuation.jsonl'
        punctuation.write_text(json.dumps({'caption': 'a cat and a dog.', 'masked': '$ and $.'}), encoding='utf-8')
        _, masked = train.load_tokens(punctuation, tokenizer, 77)
        assert masked.eq(placeholder).sum().item() == 2
        for invalid in [[], {'caption': 'cat', 'masked': 'x ' * 100 + '$'}]:
            punctuation.write_text(json.dumps(invalid), encoding='utf-8')
            try:
                train.load_tokens(punctuation, tokenizer, 77)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid/truncated masked row accepted')
        config = CLIPConfig(projection_dim=8, text_config=dict(
            vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, projection_dim=8,
            max_position_embeddings=77, bos_token_id=vocab['<|startoftext|>'],
            eos_token_id=vocab['<|endoftext|>'], pad_token_id=vocab['<|endoftext|>']),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, projection_dim=8, image_size=32, patch_size=8))
        CLIPModel(config).save_pretrained(model_dir)
        CLIPImageProcessor(size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32}).save_pretrained(model_dir)
        rows = [{'caption': f'a {color} cat sleeps', 'masked': '$ sleeps'}
                for color in ['red', 'green', 'blue', 'white']]
        captions = root / 'captions.jsonl'
        captions.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        checkpoints = []
        for name, weight, checkpointing in [('baseline', 0.0, False), ('contrastive', 0.1, True)]:
            args = argparse.Namespace(captions=str(captions), output=str(root / name),
                model=str(model_dir), steps=2, batch_size=2, accumulation=2,
                learning_rate=1e-4, contrastive_weight=weight, temperature=.07,
                seed=12345, save_every=1, precision='fp32', device='cpu',
                gradient_checkpointing=checkpointing, init_checkpoint=None)
            checkpoint = train.run(args)
            state = torch.load(checkpoint, weights_only=True)
            assert state['step'] == 2 and state['config']['effective_batch_size'] == 4
            logs = [json.loads(line) for line in (root / name / 'train.jsonl').read_text().splitlines()]
            assert len(logs) == 2 and all(record['loss'] >= 0 for record in logs)
            checkpoints.append(checkpoint)
        assert any(not torch.equal(torch.load(checkpoints[0], weights_only=True)['Phi'][key],
                                   torch.load(checkpoints[1], weights_only=True)['Phi'][key])
                   for key in state['Phi'])
        dataset = root / 'CIRR'
        (dataset / 'cirr/captions').mkdir(parents=True)
        (dataset / 'cirr/image_splits').mkdir(parents=True)
        (dataset / 'dev').mkdir()
        paths = {}
        for name, color in [('ref', 'red'), ('a', 'green'), ('b', 'blue')]:
            paths[name] = f'dev/{name}.png'
            Image.new('RGB', (32, 32), color).save(dataset / paths[name])
        records = [dict(reference='ref', target_hard='a', caption='is green',
                        img_set={'members': list(paths)})]
        (dataset / 'cirr/captions/cap.rc2.val.json').write_text(json.dumps(records), encoding='utf-8')
        (dataset / 'cirr/image_splits/split.rc2.val.json').write_text(json.dumps(paths), encoding='utf-8')
        for index, checkpoint in enumerate(checkpoints):
            result = evaluate.run(argparse.Namespace(dataset=str(dataset), checkpoint=str(checkpoint),
                                  output=str(root / f'eval-{index}.json'), batch_size=2, device='cpu'))
            assert result['gallery_size'] == 3 and result['queries'] == 1
            assert result['metrics']['R@5'] == 100
        print('PASS: offline tiny CLIP train/save/load/evaluate, baseline and contrastive, accumulation and checkpointing')
        print('These synthetic checks DO NOT measure LinCIR retrieval quality.')


if __name__ == '__main__':
    main()
