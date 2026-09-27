"""Offline checks: real small CLIP, gradients, loss and hand-calculated CIRR ranks."""
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

import lincir_core as core


def main():
    torch.manual_seed(12345)
    torch.set_num_threads(2)
    from types import SimpleNamespace
    broken_tokenizer = SimpleNamespace(encode=lambda *a, **kw: [2], all_special_ids=[1, 2])
    try:
        core.placeholder_token_id(broken_tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError('Broken tokenizer mapped the placeholder to EOS')
    encoder = CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, projection_dim=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        bos_token_id=30, eos_token_id=31, pad_token_id=0,
    )).eval().requires_grad_(False)
    # Independent oracle: putting each pseudo-vector in the embedding table
    # must match substitution at every occurrence, including repeated keywords.
    repeated_ids = torch.tensor([[30, 7, 3, 7, 31, 0], [30, 4, 7, 31, 0, 0]])
    replacements = torch.randn(2, 16)
    actual = core.encode_pseudo(encoder, repeated_ids, replacements, placeholder_id=7)
    embedding = encoder.text_model.embeddings.token_embedding.weight
    original_embedding = embedding[7].clone()
    expected = []
    try:
        with torch.no_grad():
            for row, replacement in zip(repeated_ids, replacements):
                embedding[7].copy_(replacement)
                expected.append(encoder(row.unsqueeze(0)).text_embeds)
    finally:
        with torch.no_grad():
            embedding[7].copy_(original_embedding)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)
    ids = torch.tensor([[30, 7, 3, 31, 0], [30, 4, 7, 31, 0]])
    phi = core.Phi(8, 32, 16, 0.0)
    with torch.no_grad():
        teacher = encoder(ids).text_embeds
    pseudo = phi(teacher)
    predicted = core.encode_pseudo(encoder, ids, pseudo, placeholder_id=7)
    loss, mse, contrastive = core.smp_loss(predicted, teacher, 0.1, 0.07)
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in phi.parameters()) > 0
    assert all(p.grad is None for p in encoder.parameters())
    assert torch.equal(encoder(ids).text_embeds, teacher), 'pseudo hook leaked'
    optimizer = torch.optim.AdamW(phi.parameters(), lr=1e-3)
    before = next(phi.parameters()).detach().clone()
    optimizer.step()
    assert not torch.equal(before, next(phi.parameters()))
    baseline, _, nce = core.smp_loss(predicted, teacher, 0.0, 0.07)
    assert torch.equal(baseline, torch.nn.functional.mse_loss(predicted, teacher))
    assert nce.item() == 0
    identity = torch.eye(3)
    _, _, good = core.smp_loss(identity, identity, 0.1, 0.07)
    _, _, bad = core.smp_loss(identity.roll(1, 0), identity, 0.1, 0.07)
    assert good < bad
    for weight, temperature in [(-1, .07), (.1, 0), (float('nan'), .07)]:
        try:
            core.smp_loss(identity, identity, weight, temperature)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid loss configuration accepted')
    # MSE must not silently broadcast a different batch or accept an empty batch.
    for predicted_shape, target_shape in [((2, 8), (1, 8)), ((0, 8), (0, 8)), ((2, 0), (2, 0)), ((8,), (8,))]:
        try:
            core.smp_loss(torch.zeros(predicted_shape), torch.zeros(target_shape))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid SMP feature shapes silently accepted')
    # Reference has highest similarity. Targets: a ranks 1st, b ranks 2nd.
    gallery = torch.tensor([[1., .15], [1., 0.], [0., 1.], [-1., 0.]])
    queries = torch.tensor([[1., .2], [1., .1]])
    records = [dict(reference='ref', target_hard=t, img_set={'members':['ref','a','b']})
               for t in ['a', 'b']]
    metrics = core.cirr_metrics(queries, gallery, ['ref','a','b','c'], records)
    assert metrics['R@1'] == 50.0 and metrics['R@5'] == 100.0
    assert metrics['subset_R@1'] == 50.0 and metrics['subset_R@2'] == 100.0
    try:
        core.cirr_metrics(queries, gallery, ['ref','a','x','c'], records)
    except ValueError:
        pass
    else:
        raise AssertionError('missing ground truth silently accepted')
    # Global target ranks are 1, 6, 11, 51; subset ranks are 1, 2, 3, 4.
    # Reference is the closest image and must be removed from both rankings.
    names = ['ref'] + [f'g{i}' for i in range(1, 53)]
    gallery = torch.tensor([[53. - i, float(i)] for i in range(53)])
    records = [dict(reference='ref', target_hard=f'g{i}',
                    img_set={'members': ['ref', 'g1', 'g6', 'g11', 'g51', 'g52']})
               for i in [1, 6, 11, 51]]
    queries = torch.tensor([[1., 0.]]).repeat(4, 1)
    assert core.cirr_metrics(queries, gallery, names, records, batch_size=3) == {
        'R@1': 25., 'R@5': 25., 'R@10': 50., 'R@50': 75.,
        'subset_R@1': 25., 'subset_R@2': 50., 'subset_R@3': 75.,
    }
    print('PASS: pseudo substitution parity, frozen CLIP gradients, optimizer, hook cleanup, losses, CIRR rank boundaries/validation')


if __name__ == '__main__':
    main()
