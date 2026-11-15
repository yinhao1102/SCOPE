from SCOPE.utils.reid5o_metrics import Evaluator1


def supports_msf(model):
    """Whether this model was constructed with the learned fusion heads."""
    base_model = getattr(model, "module", model)
    return all(hasattr(base_model, f"mm_{group}_adapter")
               for group in ("two", "three", "four"))


ORBENCH_QUERY_LOADERS = [
    "nir_query_loader", "cp_query_loader", "sk_query_loader", "text_query_loader",
    "nir_cp_query_loader", "cp_nir_query_loader", "nir_sk_query_loader", "sk_nir_query_loader",
    "nir_text_query_loader", "text_nir_query_loader", "cp_sk_query_loader", "sk_cp_query_loader",
    "cp_text_query_loader", "text_cp_query_loader", "sk_text_query_loader", "text_sk_query_loader",
    "nir_cp_sk_query_loader", "cp_nir_sk_query_loader", "sk_nir_cp_query_loader",
    "nir_cp_text_query_loader", "cp_nir_text_query_loader", "text_nir_cp_query_loader",
    "nir_sk_text_query_loader", "sk_nir_text_query_loader", "text_nir_sk_query_loader",
    "cp_sk_text_query_loader", "sk_cp_text_query_loader", "text_cp_sk_query_loader",
    "nir_cp_sk_text_query_loader", "cp_nir_sk_text_query_loader",
    "sk_nir_cp_text_query_loader", "text_nir_cp_sk_query_loader",
]


def build_orbench_evaluator(dataloaders, training):
    eval_loaders = dataloaders[1:-1] if training else dataloaders[:-1]
    gallery_loader, query_loaders = eval_loaders[0], eval_loaders[1:]
    if len(query_loaders) != len(ORBENCH_QUERY_LOADERS):
        raise RuntimeError(
            f"Expected {len(ORBENCH_QUERY_LOADERS)} ORBench query loaders, "
            f"received {len(query_loaders)}"
        )
    return Evaluator1(
        gallery_loader=gallery_loader,
        get_mAP=True,
        **dict(zip(ORBENCH_QUERY_LOADERS, query_loaders)),
    )


