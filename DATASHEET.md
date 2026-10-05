# Datasheet

This describes the raw LINCS L1000 Phase II files in `data/`.

## Source

Publicly funded LINCS Connectivity Map L1000 Phase II, GEO GSE70138:

- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138
- L1000 GEO guide: https://clue.io/GEO-guide

This is an assessment dataset, not a clinical or production benchmark.

## Files

| File | What it is |
|---|---|
| `GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx.gz` | Level 3 inferred expression, 345,976 profiles × 12,328 genes |
| `GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz` | Level 5 consensus signatures, 118,050 × 12,328 |
| `GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz` | cell-line metadata |
| `GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz` | gene metadata (landmark + inferred) |
| `GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz` | per-well / instance metadata |
| `GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz` | perturbation metadata |
| `GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz` | signature metadata |
| `SHA512SUMS.txt` | GEO SHA-512 checksums |

There is no prepared train/validation split and no pre-quantized target file. How you subset, split, and represent the data is part of the exercise.

## Leakage boundary

The underlying targets are public and may be recoverable. Do not look up public target values for evaluation rows, reconstruct held-out identities from public sources, or use external data known to contain LINCS expression labels.

## Known limitations

- Immortalized cell lines do not represent patients or primary tissues.
- Level 3 and Level 5 values inherit assay, normalization, replicate, and consensus artifacts.
- MOA labels simplify polypharmacology and may be incomplete or wrong.
- Dose and time are observed values, not a balanced factorial design.
- Chemical and biological groups are dependent; row-level uncertainty can overstate effective sample size.
