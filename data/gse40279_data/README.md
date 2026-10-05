# GSE40279 data

The large raw GEO archives are not tracked in this repository. They are
downloaded from NCBI GEO and verified against the SHA-256 values in
`manifest.json`.

From the repository root, run:

```bash
python scripts/prepare_gse40279.py --config config/gse40279_paper.json
```

This creates the local, untracked `prepared/runner_ready/` representation used by the
GSE40279 experiment. The source dataset is GEO accession GSE40279; the URLs
and checksums are recorded in `manifest.json`.
