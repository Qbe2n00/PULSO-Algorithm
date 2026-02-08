from typing import Dict, Iterable, Literal, Optional, Tuple, Union
from typing import Any, Mapping, Union
from scipy.spatial.distance import cdist

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
import seaborn as sns

from utils import isiterable

def get_per_sample_nz_counts_of_genes(
    adata, sample_key: str = "sample"
) -> pd.DataFrame:
    bexp_df = sc.get.obs_df(adata, adata.var_names.to_list()) > 0.0
    bexp_df[sample_key] = adata.obs[sample_key]
    return bexp_df.groupby(sample_key, observed=True).sum()

def get_sample_reproducible_genes(
    adata, sample_key: str = "sample", n_cells_per_sample: int = 10, n_samples: int = 3
) -> pd.Series:
    return (
        get_per_sample_nz_counts_of_genes(adata, sample_key=sample_key)
        >= n_cells_per_sample
    ).sum(axis=0) >= n_samples

def cell_cell_emb(
        _adata: sc.AnnData,
        dict_adata: Dict,
        out_file: str,
):
    print(f"Creating {out_file}...")
    adata_c = sc.concat(([dict_adata[c] for c in dict_adata.keys() if c[0] == "C"]))
    adata_c.obs = _adata.obs.loc[adata_c.obs.index].copy()
    sc.pp.neighbors(adata_c, n_neighbors=15, use_rep="X", transformer="sklearn")
    sc.tl.umap(adata_c)
    print(f"Writing {out_file}...")
    adata_c.write(out_file)
    del dict_adata
    return adata_c

def cell_gene_emb(
        _adata: sc.AnnData,
        dict_adata: Dict,
        adata_c: sc.AnnData,
        out_file: str,
        cat_order: list
):
    print(f"Creating {out_file}...")
    adata_c.obs = pd.DataFrame(index=adata_c.obs_names)
    import simba as si
    adata_cg = si.tl.embed(
        adata_ref=adata_c,
        list_adata_query=[dict_adata["G"]],
    )
    adata_cg.obs["entity_anno"] = "cell"
    adata_cg.obs.loc[dict_adata["G"].obs_names, "entity_anno"] = "gene"
    adata_cg.obs["entity_anno"] = adata_cg.obs["entity_anno"].astype("category")
    adata_cg.obs["condition"] = "gene"
    adata_cg.obs.loc[_adata.obs_names, "condition"] = _adata.obs["condition"].astype(str)
    adata_cg.obs["condition"] = adata_cg.obs["condition"].astype("category")
    adata_cg.obs["condition"] = adata_cg.obs["condition"].cat.reorder_categories(cat_order)
    sc.pp.neighbors(adata_cg, n_neighbors=50, use_rep="X", transformer="sklearn")
    sc.tl.umap(adata_cg)
    print(f"Writing {out_file}...")
    adata_cg.write(out_file)
    del dict_adata
    return adata_cg  

def cg_dist(
        adata_cg: sc.AnnData,
        out_file: str
):
    cg_dist = pd.DataFrame(
        cdist(
            adata_cg[adata_cg.obs["entity_anno"] == "cell", :].copy().X,
            adata_cg[adata_cg.obs["entity_anno"] == "gene", :].copy().X,
        ),
        index=adata_cg.obs_names[adata_cg.obs["entity_anno"] == "cell"],
        columns=adata_cg.obs_names[adata_cg.obs["entity_anno"] == "gene"],
    )
    cg_dist.to_parquet(out_file, compression=None, index=True)

    return cg_dist

def dist_prep(
        prev_file: str,
        out_file: str,
):
    from sklearn.preprocessing import PowerTransformer
    cg_dist_df = pd.read_parquet(prev_file)
    cg_zdist_df = pd.DataFrame(
        PowerTransformer(method="box-cox", standardize=True).fit_transform(cg_dist_df),
        index=cg_dist_df.index,
        columns=cg_dist_df.columns,
    )
    cg_zdist_df.to_parquet(out_file, compression=None, index=True)
    return cg_zdist_df

def get_lig_actv(
    adata: sc.AnnData,
    cell_id: str,
    gene_dist: pd.Series,
    abs_lower_bound: int,
    lig_target_rel_df: pd.DataFrame,
    lig_rec_dict: Dict[str, set[str]],
    dist_hard_thres: float = -2.0,
    frac_soft_thres: float = 0.25,
) -> pd.Series:
    exp_genes = sc.get.obs_df(
        adata[(adata.obs_names == cell_id)], adata.var_names.to_list()
    ).squeeze()
    exp_genes = exp_genes[exp_genes > 0.0].copy()

    val_gene_dist = gene_dist[gene_dist.index.isin(exp_genes.index)].copy()
    if (val_gene_dist < 0.0).sum() < abs_lower_bound:
        val_gene_dist -= val_gene_dist.sort_values().iloc[abs_lower_bound]
    val_gene_dist = val_gene_dist.clip(
        lower=(-1 * np.abs(dist_hard_thres)), upper=0.0
    ) / (-1 * np.abs(dist_hard_thres))

    all_genes = lig_target_rel_df.columns.intersection(val_gene_dist.index)
    _exp_genes = set(exp_genes.index)

    lig_response_x = val_gene_dist[all_genes]

    val_lig = set(
        [
            l
            for l in lig_target_rel_df.index
            if len(_exp_genes & set(lig_rec_dict[l])) > 0
        ]
    )
    lig_response_y = lig_target_rel_df.loc[
        lig_target_rel_df.index.isin(val_lig), all_genes
    ].copy(deep=True)

    lig_actv = lig_response_y.corrwith(
        lig_response_x[all_genes], axis=1, method="pearson", numeric_only=True
    )
    lig_actv.name = cell_id
    return lig_actv

def ligand_receptor(
        resource_path: str
):
    lig_rec_df = pd.read_csv(resource_path, index_col=0, float_precision="round_trip")
    lig_rec_df = lig_rec_df.astype("category")
    lig_rec_dict = {}
    for _, row in lig_rec_df.iterrows():
        l = row["from"]
        r = row["to"]
        if l not in lig_rec_dict.keys():
            lig_rec_dict[l] = []
        lig_rec_dict[l].append(r)
    for l in lig_rec_dict.keys():
        lig_rec_dict[l] = set(lig_rec_dict[l])
    return lig_rec_dict

def ligand_target(
        resource_path: str,
        all_genes: pd.DataFrame,
        chk_genes: list
):
    lig_target_rel_df = pd.read_csv(
        resource_path, index_col=0, float_precision="round_trip"
    ).transpose()

    og_shape = lig_target_rel_df.shape
    lig_target_rel_df = lig_target_rel_df.loc[
        lig_target_rel_df.index.isin(all_genes.index),
        lig_target_rel_df.columns.isin(chk_genes),
    ].copy()
    lig_target_rel_df = lig_target_rel_df.loc[
        (lig_target_rel_df.sum(axis=1) > 0.0), (lig_target_rel_df.sum(axis=0) > 0.0)
    ].copy()
    print(f"{og_shape} > {lig_target_rel_df.shape}")
    return lig_target_rel_df

def val_lig(
        scores: pd.DataFrame
):
    scores[scores < 0.0] = 0.0
    val_ligs = (scores > 0.0).sum(axis=0) >= 10
    ldata = sc.AnnData(scores.loc[:, val_ligs].copy().astype(np.float32))
    ldata.X = csr_matrix(ldata.X)
    return ldata

def lig_pred_pp(
        adata: sc.AnnData,
        adata_all: sc.AnnData,
        prev_file: str,
        out_file: str
):
    print(f"Creating {out_file}...")
    from scipy.sparse import csr_matrix
    ldata = sc.read(prev_file)
    X = ldata.X.toarray()
    X[X < 1e-4] = 0.0
    ldata.X = csr_matrix(X)
    ldata = ldata[:, ldata.var_names.isin(adata_all.var_names)].copy()
    ldata.obs = adata.obs.copy()
    ldf = ldata.to_df() > 0.0
    ldf["sample"] = ldata.obs["sample"]
    val_ligs = (ldf.groupby("sample", observed=True).sum() > 3).sum(axis=0) > 3
    ldata = ldata[:, ldata.var_names.isin(val_ligs.index[val_ligs])].copy()
    ldata.write(out_file)

    return ldata

def run_ligand_differential_test(
    ldata: sc.AnnData,
    celltypes_oi: Union[str, Iterable[str]],
    ref_celltypes: Iterable[str],
    celltype_key: str = "celltype_1",
    condition_oi: str = "PV",
    ref_conditions: Iterable[str] = ["PV", "HC"],
    condition_key: str = "condition",
    sep: str = ":",
    lower_clip: float = 1e-8,
    group_thres: int = 50,
    lfc_thres: float = np.log2(1.5),
    pval_thres: float = 1e-2,
    pct_thres: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    from itertools import product

    from scipy.stats import ranksums
    from statsmodels.stats import multitest

    group_key = "condition_celltype"
    cell_df = ldata.obs.copy()
    cell_df[group_key] = (
        cell_df[condition_key].astype(str) + sep + cell_df[celltype_key].astype(str)
    ).astype("category")
    groups_oi = list(
        product(
            [condition_oi],
            celltypes_oi if isiterable(celltypes_oi) else [celltypes_oi],
        )
    )
    groups_oi = [(x[0] + sep + x[1]) for x in groups_oi]
    _ref_celltypes = (
        ref_celltypes if isiterable(ref_celltypes) else [ref_celltypes]
    )
    _ref_conditions = (
        ref_conditions
        if isiterable(ref_conditions)
        else [ref_conditions]
    )
    ref_groups = [sep.join(x) for x in product(_ref_conditions, _ref_celltypes)]
    for group_oi in groups_oi:
        if group_oi in ref_groups:
            ref_groups.remove(group_oi)

    group_counts = cell_df[group_key].value_counts()
    val_groups = group_counts[group_counts >= group_thres].index
    ref_groups = [g for g in ref_groups if g in val_groups]

    lig_df = ldata.to_df()

    res = {}
    for group_oi in groups_oi:
        idx_oi = cell_df.index[cell_df[group_key] == group_oi]
        ldf_oi = lig_df.loc[idx_oi].copy()
        mean_oi = ldf_oi.mean(axis=0).clip(lower=lower_clip)
        pct_oi = (ldf_oi > 0.0).mean(axis=0)

        for ref_group in ref_groups:
            ref_idx = cell_df.index[cell_df[group_key] == ref_group]
            ref_ldf = lig_df.loc[ref_idx].copy()
            ref_mean = ref_ldf.mean(axis=0).clip(lower=lower_clip)
            stat, p = ranksums(ldf_oi, ref_ldf)
            pa = multitest.multipletests(
                p, alpha=5e-2, method="fdr_bh", is_sorted=False
            )[1]
            cur_res = pd.DataFrame(
                dict(
                    scores=stat,
                    logfoldchanges=np.log2(mean_oi / ref_mean),
                    pvals=p,
                    pvals_adj=pa,
                    pct_nz_group=pct_oi,
                    pct_nz_reference=(ref_ldf > 0.0).mean(axis=0),
                ),
                index=ldf_oi.columns,
            )
            cur_res["sig"] = (
                (np.abs(cur_res["logfoldchanges"]) >= lfc_thres)
                & (cur_res["pvals_adj"] < pval_thres)
                & (cur_res["pct_nz_group"] >= pct_thres)
                & (cur_res["pct_nz_group"] >= cur_res["pct_nz_reference"])
            )
            cur_res["class"] = cur_res.apply(
                lambda row: "ns"
                if not row["sig"]
                else "up"
                if row["logfoldchanges"] > 0.0
                else "down",
                axis=1,
            )

            cur_key = f"{group_oi} vs {ref_group}"
            res[cur_key] = cur_res
            _sum = cur_res["class"].value_counts()
            _sum.name = cur_key
            print(_sum)

    return res

def get_differentially_acting_ligands(
    lig_dt_df: Dict[str, pd.DataFrame],
    ldata: sc.AnnData,
    celltypes_oi: Union[str, Iterable[str]],
    celltype_key: str = "celltype_1",
    condition_oi: str = "PV",
    condition_key: str = "condition",
    n_up_offset: int = 1,
    lfc_thres: float = 0.0,
    pval_thres: float = 1e-2,
    pct_thres: float = 0.25,
) -> Iterable[str]:
    _celltypes_oi = (
        celltypes_oi if isiterable(celltypes_oi) else [celltypes_oi]
    )

    res_sum = pd.DataFrame(
        {k: (lig_dt_df[k]["class"] == "up") for k in lig_dt_df.keys()}
    )
    lfc_sum = pd.DataFrame(
        {k: (lig_dt_df[k]["logfoldchanges"] > lfc_thres) for k in lig_dt_df.keys()}
    )
    pval_sum = pd.DataFrame(
        {k: (lig_dt_df[k]["pvals_adj"] < pval_thres) for k in lig_dt_df.keys()}
    )
    pct_sum = pd.DataFrame(
        {k: (lig_dt_df[k]["pct_nz_group"] >= pct_thres) for k in lig_dt_df.keys()}
    )
    pct_sum2 = pd.DataFrame(
        {
            k: (lig_dt_df[k]["pct_nz_group"] >= lig_dt_df[k]["pct_nz_reference"])
            for k in lig_dt_df.keys()
        }
    )

    sig_ligs = (
        (res_sum.sum(axis=1) >= (res_sum.shape[1] - n_up_offset))
        & (lfc_sum.sum(axis=1) == (lfc_sum.shape[1]))
        & (pval_sum.sum(axis=1) == (pval_sum.shape[1]))
        & (pct_sum.sum(axis=1) == (pct_sum.shape[1]))
        & (pct_sum2.sum(axis=1) == (pct_sum.shape[1]))
    )
    print(sig_ligs.value_counts())

    sorted_ligs = (
        ldata[
            (ldata.obs[condition_key] == condition_oi)
            & (ldata.obs[celltype_key].isin(_celltypes_oi))
        ]
        .to_df()
        .mean(axis=0)
        .sort_values(ascending=False)
    )
    return sorted_ligs.index[sorted_ligs.index.isin(sig_ligs[sig_ligs].index)].to_list()


def run_ligrec_for_sender_discovery(
    adata: sc.AnnData,
    adata_all: sc.AnnData,
    ligands_oi: Iterable[str],
    celltypes_oi: Iterable[str],
    celltype_coarse_oi: str,
    celltype_key: str = "celltype_1",
    celltype_coarse_key: str = "celltype_0",
    condition_key: str = "condition",
    key_added: str = "celltype_01c",
    sep: str = ":",
    group_thres: int = 50,
    exp_frac_thres: float = 1e-4,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    import warnings
    from itertools import product

    import omnipath
    import squidpy as sq

    lr_df = omnipath.interactions.import_intercell_network(
        transmitter_params={"categories": "ligand"},
        receiver_params={"categories": "receptor"},
    )
    _lr_df = lr_df[["genesymbol_intercell_source", "genesymbol_intercell_target"]]
    _lr_df.columns = ["source", "target"]
    _lr_df = _lr_df.loc[_lr_df["source"].isin(ligands_oi)]
    _lr_df = _lr_df.loc[_lr_df["source"].isin(adata_all.var_names)]
    _lr_df = _lr_df.loc[_lr_df["target"].isin(adata.var_names)]
    _lr_df = _lr_df.drop_duplicates().reset_index(drop=True)

    annot = adata.obs[celltype_key]
    adata_all.obs[key_added] = (
        adata_all.obs[condition_key].astype(str)
        + sep
        + adata_all.obs.apply(
            lambda row: (annot.loc[row.name])
            if (row.name in annot.index)
            else (row[celltype_coarse_key]),
            axis=1,
        ).astype("str")
    ).astype("category")

    val_cts = adata_all.obs[key_added].value_counts() >= group_thres
    adata_all_ = adata_all[adata_all.obs[key_added].isin(val_cts[val_cts].index)].copy()

    sr_pairs = []
    for g in adata_all.obs[condition_key].cat.categories:
        senders = (
            adata_all_.obs.loc[
                (
                    (adata_all_.obs[celltype_coarse_key] != celltype_coarse_oi)
                    & (adata_all_.obs[condition_key] == g)
                ),
                key_added,
            ]
            .unique()
            .tolist()
        )
        rcvs = (
            adata_all_.obs.loc[
                (
                    (adata_all_.obs[key_added].isin(celltypes_oi))
                    & (adata_all_.obs[condition_key] == g)
                ),
                key_added,
            ]
            .unique()
            .tolist()
        )
        sr_pairs += list(product(senders, rcvs))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return sq.gr.ligrec(
            adata_all_,
            cluster_key=key_added,
            use_raw=False,
            clusters=sr_pairs,
            interactions=_lr_df,
            threshold=exp_frac_thres,
            corr_method="fdr_bh",
            alpha=5e-2,
            n_perms=int(1e4),
            n_jobs=(sc.settings.n_jobs * 2),
            copy=True,
            numba_parallel=False,
            **kwargs,
        )
