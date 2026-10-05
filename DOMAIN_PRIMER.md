# Domain primer

Background for people who have not worked with L1000 data. It describes the measurements, not how to model them.

## What one example means

A perturbation experiment asks: after a cell line is exposed to a compound at a stated dose and duration, how did measured gene expression change.

LINCS L1000 reports that in two main layers in this workspace:

- **Level 3** — inferred expression profiles for individual wells / instances.
- **Level 5** — consensus signatures. These are z-scores. Positive values indicate relative up-regulation and negative values indicate relative down-regulation in the source processing system. They are not probabilities, fold changes, clinical effects, or direct measurements of efficacy.

Typical metadata includes cell line, perturbation identity, dose, time, and sometimes a mechanism-of-action (MOA) label. Chemical structure (SMILES) is not always in the GEO tables; you may have to join or derive it if you need it.

## Landmark vs inferred genes

L1000 directly measures 978 landmark genes. The Phase II matrices here are 12,328 genes: the 978 landmarks plus inferred genes. Gene identity is in `GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz`. Which gene space you use is your decision.

## What signal is present

Cell line, dose, duration, and MOA carry strong group-level information. Structure can distinguish compounds inside those groups. L1000 is noisy: baseline and treated profiles are highly correlated, and the residual signal can be weak.

A positive result here supports only improved prediction within this dataset's supported cell, MOA, dose, and time distribution. It does not establish mechanism, efficacy, safety, patient response, or generalization to arbitrary chemistry.

## Things that make this data awkward

- Level 5 values inherit assay, normalization, replicate, and consensus artifacts.
- MOA labels simplify polypharmacology and are sometimes wrong.
- Dose and time are observed, not a balanced design.
- Distinct scaffolds can still be close analogues.
- Chemical and biological groups are dependent, so treating rows as independent overstates how much evidence you have.
- Aggregate performance can hide complete failure inside one cell line or MOA.
- The underlying data is public, which is why the no-lookup rule is a declaration rather than a technical barrier.
