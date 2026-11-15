"""Replay a recorded cross-modal candidate order across worker filesystems."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@lru_cache(maxsize=4)
def _validated_index(index_path, dataset_root):
    root = Path(dataset_root)
    index = json.loads(Path(index_path).read_text())
    assert index['version'] == 1
    assert set(index['modalities']) == {'nir', 'cp', 'sk'}
    for name, digest in index['annotation_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    for modality, identities in index['modalities'].items():
        assert set(os.listdir(root / modality)) == set(identities), modality
        for identity, filenames in identities.items():
            assert len(filenames) == len(set(filenames)), (modality, identity)
            assert set(os.listdir(root / modality / identity)) == set(filenames), (modality, identity)
    return index


def recorded_candidate_paths(dataset_root, absolute):
    """Return three dictionaries in the original order, or retain legacy behavior."""
    index_path = os.environ.get('SCOPE_ORBENCH_CANDIDATE_INDEX')
    if not index_path:
        return None
    root = Path(dataset_root)
    index = _validated_index(str(Path(index_path).resolve()), str(root.resolve()))
    result = []
    for modality in ('nir', 'cp', 'sk'):
        paths = {}
        for identity, filenames in index['modalities'][modality].items():
            relatives = [os.path.join(modality, identity, name) for name in filenames]
            paths[identity] = [str(root / name) for name in relatives] if absolute else relatives
        result.append(paths)
    return tuple(result)
