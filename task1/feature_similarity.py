import torch


def cosine_stability(feat_clean: torch.Tensor, feat_transformed: torch.Tensor) -> float:
    assert feat_clean.shape[0] == feat_transformed.shape[0]
    fc = torch.nn.functional.normalize(feat_clean, dim=1)
    ft = torch.nn.functional.normalize(feat_transformed, dim=1)
    return (fc * ft).sum(dim=1).mean().item()


def compute_all_stabilities(backbones, clean_images: torch.Tensor,
                             variant_images: dict, device: str = "cuda") -> dict:
    """
    Convenience wrapper for interventions that are paired 1:1 against the
    SAME clean image set (grayscale, translation, patch shuffle). Cue
    conflicts are paired against their own content images instead, and are
    computed separately in scripts/run_task1.py.

    variant_images: {"grayscale": tensor, "translation_16px": tensor, ...},
                     each aligned index-for-index with `clean_images`.
    Returns {variant_name: {backbone_name: I_T}}.
    """
    from analysis.evaluate_bias import extract_features  # local import avoids cycles

    results = {}
    clean_feats_cache = {
        name: extract_features(bb, clean_images, device=device)
        for name, bb in backbones.items()
    }
    for variant_name, imgs in variant_images.items():
        results[variant_name] = {}
        for name, bb in backbones.items():
            trans_feats = extract_features(bb, imgs, device=device)
            results[variant_name][name] = cosine_stability(
                clean_feats_cache[name], trans_feats)
    return results