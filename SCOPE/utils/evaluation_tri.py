"""The archived Tri evaluation, including source-RGB exclusion."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SCOPE.datasets.tri import TriGalleryDataset, TriTextSketchDataset, build_transforms

NAMES = ("R1", "R5", "R10", "mAP", "mINP")

def tri_eval_loaders(data, args):
    transform = build_transforms(args.img_size, training=False)
    gallery = TriGalleryDataset(data['image_pids'], data['image_paths'], data['image_ids'], transform)
    sketch = TriGalleryDataset(data['sketch_pids'], data['sketch_paths'], data['sketch_ids'], transform)
    text = TriTextSketchDataset(data, transform, args.text_length)
    opts = dict(batch_size=args.test_batch_size, num_workers=args.num_workers,
                shuffle=False, pin_memory=True, generator=torch.Generator().manual_seed(2))
    return DataLoader(gallery, **opts), DataLoader(sketch, **opts), DataLoader(text, **opts)

@torch.no_grad()
def encode_tri(model, loaders):
    device = next(model.parameters()).device
    model.eval()
    gallery, sketch, text = loaders
    cache = {}
    for key, loader, method in [('gallery', gallery, model.encode_rgb_cls),
                                ('sketch', sketch, model.encode_sk_cls)]:
        values, pids, ids = [], [], []
        for pid, image, iid in loader:
            values.append(method(image.to(device)).float())
            pids.append(pid)
            ids.append(iid)
        cache[key] = torch.cat(values)
        cache[key + '_pids'] = torch.cat(pids).numpy()
        cache[key + '_ids'] = torch.cat(ids).numpy()
    values, pids, ids = [], [], []
    for _, iid, pid, caption in text:
        values.append(model.encode_text_cls(caption.to(device)).float())
        pids.append(pid)
        ids.append(iid)
    cache.update(text=torch.cat(values), text_pids=torch.cat(pids).numpy(),
                 text_ids=torch.cat(ids).numpy())
    assert np.array_equal(cache['sketch_ids'], np.arange(len(cache['sketch_ids'])))
    cache['paired_sketch'] = cache['sketch'][torch.as_tensor(cache['text_ids'], device=device)]
    return cache

def rank_arrays(similarity, qids, gids, qimages, gimages, exclude):
    """Same numpy ordering, identity positives and source-image exclusion as UNIReID."""
    orders = np.argsort(-similarity, axis=1)
    totals = np.zeros(5, np.float64)
    valid = 0
    for i, order in enumerate(orders):
        if exclude:
            order = order[~((gids[order] == qids[i]) & (gimages[order] == qimages[i]))]
        matched = gids[order] == qids[i]
        positive_ranks = np.flatnonzero(matched)
        if not len(positive_ranks):
            continue
        valid += 1
        cumulative = matched.cumsum()
        totals[:3] += [(cumulative[min(k - 1, len(cumulative) - 1)] > 0) for k in (1, 5, 10)]
        totals[3] += np.mean(cumulative[positive_ranks] / (positive_ranks + 1))
        totals[4] += cumulative[positive_ranks[-1]] / (positive_ranks[-1] + 1)
    return totals, valid

@torch.no_grad()
def evaluate_tri(model, cache, fused):
    device = next(model.parameters()).device
    model.eval()
    if fused:
        outputs = []
        for start in range(0, len(cache['text']), 512):
            sk, text = cache['paired_sketch'][start:start + 512], cache['text'][start:start + 512]
            outputs.append(model.msf_fuse_by_indices([None, None, sk, text], (2, 3))[0])
        combined = torch.cat(outputs)
    else:
        combined = (cache['paired_sketch'] + cache['text']) / 2
    gallery = F.normalize(cache['gallery'], dim=1)
    cases = [('TEXT_original', cache['text'], 'text', False),
             ('TEXT', cache['text'], 'text', True),
             ('SK', cache['sketch'], 'sketch', True),
             ('TEXT+SK', combined, 'text', True)]
    result = {}
    for name, features, key, exclude in cases:
        total, valid = np.zeros(5), 0
        for start in range(0, len(features), 256):
            end = start + 256
            similarity = (F.normalize(features[start:end], dim=1) @ gallery.T).cpu().numpy()
            a, n = rank_arrays(similarity, cache[key + '_pids'][start:end], cache['gallery_pids'],
                               cache[key + '_ids'][start:end], cache['gallery_ids'], exclude)
            total += a
            valid += n
        assert valid > 0
        result[name] = dict(zip(NAMES, (100 * total / valid).tolist()))
        result[name].update(valid_queries=valid, total_queries=len(features), exclude_source_rgb=exclude)
    primary = ('TEXT', 'SK', 'TEXT+SK')
    single = {metric: (result['TEXT'][metric] + result['SK'][metric]) / 2 for metric in NAMES}
    return dict(per_query=result, per_group={'ONE_AVER': single, 'TWO_AVER': result['TEXT+SK']},
                mean_mAP=float(np.mean([result[k]['mAP'] for k in primary])),
                mean_R1=float(np.mean([result[k]['R1'] for k in primary])),
                group_mean_mAP=(single['mAP'] + result['TEXT+SK']['mAP']) / 2,
                selection='mean mAP of TEXT/SK/TEXT+SK, all excluding paired source RGB',
                official_protocol='UNIReID utils/metrics.py eval_func(set=2); TEXT_original separately set=0')
