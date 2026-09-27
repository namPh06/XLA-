"""LinCIR building blocks; official Phi is reused without changing its weights/layout."""
import math

import torch
import torch.nn.functional as F

from third_party.lincir.models import Phi


def encode_pseudo(encoder, input_ids, pseudo, placeholder_id=259):
    mask = input_ids.eq(placeholder_id)
    if not mask.any(dim=1).all():
        raise ValueError('Every prompt must contain a pseudo-token before truncation.')
    if pseudo.shape != (input_ids.shape[0], encoder.config.hidden_size):
        raise ValueError('Pseudo-token shape does not match the text encoder.')
    # ponytail: one forward at a time; use an explicit embedding-input model for concurrent serving.
    def replace(_module, _inputs, output):
        return torch.where(mask.unsqueeze(-1), pseudo.to(output.dtype).unsqueeze(1), output)
    handle = encoder.text_model.embeddings.token_embedding.register_forward_hook(replace)
    try:
        return encoder(input_ids=input_ids).text_embeds
    finally:
        handle.remove()


def smp_loss(predicted, target, weight=0.0, temperature=0.07):
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Contrastive weight must be finite/nonnegative; temperature finite/positive.')
    predicted, target = predicted.float(), target.detach().float()
    mse = F.mse_loss(predicted, target)
    contrastive = mse.new_zeros(())
    if weight:
        if len(predicted) < 2:
            raise ValueError('Contrastive loss needs at least two captions per micro-batch.')
        logits = F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).T / temperature
        labels = torch.arange(len(predicted), device=predicted.device)
        contrastive = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return mse + weight * contrastive, mse, contrastive


def cirr_metrics(queries, gallery, names, records, batch_size=64):
    if not records or queries.ndim != 2 or gallery.ndim != 2 or batch_size < 1:
        raise ValueError('Nonempty queries, 2D feature matrices and positive batch size are required.')
    if len(queries) != len(records) or len(gallery) != len(names) or len(set(names)) != len(names):
        raise ValueError('Feature counts/IDs do not match annotations, or gallery IDs are duplicated.')
    if queries.shape[1] != gallery.shape[1] or not torch.isfinite(queries).all() or not torch.isfinite(gallery).all():
        raise ValueError('Feature dimensions must match and all features must be finite.')
    lookup = {name: i for i, name in enumerate(names)}
    for record in records:
        ref, target = record['reference'], record['target_hard']
        members = record['img_set']['members']
        if ref == target or ref not in members or target not in members or len(set(members)) != len(members):
            raise ValueError('Invalid CIRR group or reference/target pair.')
        if any(name not in lookup for name in [ref, target, *members]):
            raise ValueError('Reference, target or group member is missing from the gallery.')
    gallery = F.normalize(gallery.float(), dim=-1)
    queries = F.normalize(queries.float(), dim=-1).to(gallery.device)
    totals = {**{f'R@{k}': 0 for k in [1, 5, 10, 50]},
              **{f'subset_R@{k}': 0 for k in [1, 2, 3]}}
    for start in range(0, len(records), batch_size):
        scores = queries[start:start + batch_size] @ gallery.T
        for row, record in enumerate(records[start:start + batch_size]):
            reference, target = lookup[record['reference']], lookup[record['target_hard']]
            scores[row, reference] = -torch.inf
            order = scores[row].argsort(descending=True, stable=True).tolist()
            order.remove(reference)
            rank = order.index(target) + 1
            group = {lookup[name] for name in record['img_set']['members']}
            group_order = [i for i in order if i in group]
            group_rank = group_order.index(target) + 1
            for k in [1, 5, 10, 50]:
                totals[f'R@{k}'] += rank <= k
            for k in [1, 2, 3]:
                totals[f'subset_R@{k}'] += group_rank <= k
    return {key: 100.0 * value / len(records) for key, value in totals.items()}
