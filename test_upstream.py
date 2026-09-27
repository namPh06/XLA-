"""Offline execution of the notebook's patched upstream functions on real tiny CLIP.

Run with Transformers 4.57.6 and Accelerate 1.12.0. Dataset/validation I/O is
synthetic; this verifies compatibility and optimizer steps, not retrieval quality.
"""
import ast
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.nn.functional as F
import transformers
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from tqdm import tqdm

from lincir_core import Phi, encode_pseudo


def functions_only(path, namespace):
    # Ignore unused upstream imports (OpenAI CLIP/torchvision); execute original
    # function bodies unchanged, including the real trainer/Accelerator/optimizer.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(path), 'exec'), namespace)


def main():
    assert transformers.__version__ == '4.57.6', 'Use the pinned Colab test environment.'
    torch.set_num_threads(2)
    notebook = json.loads(Path('notebooks/LinCIR_Colab.ipynb').read_text(encoding='utf-8'))
    patch = next(''.join(cell['source']) for cell in notebook['cells']
                 if 'def patch_upstream(' in ''.join(cell.get('source', [])))
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for filename in ['train_phi.py', 'loader.py', 'data_utils.py',
                         'encode_with_pseudo_tokens.py', 'generate_test_submission.py']:
            shutil.copyfile(Path('third_party/lincir') / filename, work / filename)
        exec(patch, {'WORK': work})
        first = {p.name: p.read_bytes() for p in work.iterdir()}
        exec(patch, {'WORK': work})
        assert first == {p.name: p.read_bytes() for p in work.iterdir()}, 'Patch is not idempotent'
        namespace = {'torch': torch, 'CLIPTextModelWithProjection': CLIPTextModelWithProjection}
        functions_only(work / 'encode_with_pseudo_tokens.py', namespace)
        official_encode = namespace['encode_with_pseudo_tokens_HF']
        encoder = CLIPTextModelWithProjection(CLIPTextConfig(
            vocab_size=300, hidden_size=16, intermediate_size=32, projection_dim=8,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
            bos_token_id=298, eos_token_id=299, pad_token_id=299,
        )).eval().requires_grad_(False)
        ids = torch.full((2, 77), 299)
        ids[:, :5] = torch.tensor([298, 259, 12, 259, 299])
        pseudo = torch.randn(2, 16, requires_grad=True)
        actual = official_encode(encoder, ids, pseudo)
        expected = encode_pseudo(encoder, ids, pseudo)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        assert pseudo.grad.abs().sum() > 0

        # Three micro-batches per epoch, accumulation=2: the last micro-batch
        # must carry across the epoch boundary, not cause an early update.
        original = ids.repeat(3, 1)
        original[original == 259] = 7
        masked = ids.repeat(3, 1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            original, masked, torch.ones(6)), batch_size=2, drop_last=True)
        forward_calls = []
        handle = encoder.register_forward_hook(lambda *unused: forward_calls.append(1))
        recalls = iter([10., 20., 15.])
        def metrics(*unused):
            recall = next(recalls)
            return {f'cirr_recall_at{k}': recall for k in [1, 5, 10, 50]}
        namespace = dict(globals(), logger=get_logger(__name__),
            build_text_encoder=lambda args: (torch.nn.Identity(), None, encoder, None),
            build_loader=lambda *unused: loader, CIRRDataset=lambda *unused: None,
            extract_image_features=lambda *unused: (torch.zeros(1, 8), ['ref']),
            extract_pseudo_tokens_with_phi=lambda *unused: (torch.zeros(1, 16), ['ref']),
            cirr_compute_val_metrics=metrics, encode_with_pseudo_tokens_HF=official_encode)
        functions_only(work / 'train_phi.py', namespace)
        from types import SimpleNamespace
        args = SimpleNamespace(output_dir=str(work / 'run'), logging_dir='logs',
            gradient_accumulation_steps=2, mixed_precision='no', report_to=None,
            seed=12345, phi_dropout=0., resume=None, use_ema=False, use_8bit_adam=False,
            cirr_dataset_path='', learning_rate=1e-4, weight_decay=.01,
            lr_scheduler='constant', lr_warmup_steps=0, max_train_steps=3,
            batch_size=2, l2_normalize=False, max_grad_norm=1.,
            checkpointing_steps=1, validation_steps=1)
        namespace['train_phi'](args)
        handle.remove()
        assert len(forward_calls) == 6, f'Expected 6 teacher micro-batches, got {len(forward_calls)}'
        latest = torch.load(work / 'run/checkpoints/phi_latest.pt', weights_only=True)
        best = torch.load(work / 'run/checkpoints/phi_best.pt', weights_only=True)
        assert latest['epoch'] == 3 and best['epoch'] == 2
        one = torch.load(work / 'run/checkpoints/phi_000000001.pt', weights_only=True)
        assert any(not torch.equal(one['Phi'][k], latest['Phi'][k]) for k in latest['Phi'])
        assert all(p.grad is None for p in encoder.parameters())
        # No silent dropping of images during validation or submission export.
        tree = ast.parse((work / 'data_utils.py').read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collate_fn']
        exec(compile(tree, 'data_utils.py', 'exec'), namespace)
        try:
            namespace['collate_fn']([torch.zeros(1), None])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid image was silently skipped')
    print('PASS: upstream encoding/gradients, accumulation across epochs, optimizer/checkpoints, best selection, image guard')


if __name__ == '__main__':
    main()
