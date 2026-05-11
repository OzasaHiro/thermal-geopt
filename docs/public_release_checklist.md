# Public Release Checklist

This checklist is for publishing the preliminary Thermal GeoPT results on
GitHub in a citable form.

## Before Release

- Confirm that the public report is up to date:
  - `docs/original_geopt_heat_transfer_preliminary_report.md`
  - `docs/m4_heatsink_transfer_results.md`
  - `docs/m5_complex_heatsink_transfer_results.md`
  - `docs/results/m4_heatsink_transfer_summary.json`
  - `docs/results/m5_complex_heatsink_transfer_summary.json`
  - `docs/assets/figures/m5_weak_cooling_surface_temperature.png`
- Confirm the repository does not include heavy local artifacts:
  - OpenFOAM raw cases
  - NPZ datasets
  - checkpoints
  - full logs
  - large generated figures
- Add only selected lightweight figures if needed.
- Check `CITATION.cff`.
- Decide the repository license before broad public use.

## Suggested Tag

```bash
git tag -a v0.1.0-preliminary -m "Preliminary original GeoPT heat-transfer transfer report"
git push origin v0.1.0-preliminary
```

## Suggested GitHub Release Title

```text
v0.1.0-preliminary: Original GeoPT transfers to solver-backed heat-sink conduction
```

## Suggested Release Notes

```text
This preliminary release documents an early cross-physics transfer finding:
the original GeoPT pretrained checkpoint improves label efficiency on
OpenFOAM-backed heat-sink solid-conduction surrogate tasks.

Highlights:
- M4 heat-sink D1: original GeoPT improves scratch by about 13% at 25 labels
  and 17% at 50 labels.
- M5 complex heat-sink D1: original GeoPT improves scratch by about 11% at
  25 labels and 16% at 50 labels.
- The current Thermal GeoPT R4 transport-lifted pretraining causes negative
  transfer on these downstream tasks.

Main report:
docs/original_geopt_heat_transfer_preliminary_report.md
```

## Zenodo

After creating the GitHub release, archive the release with Zenodo to obtain a
DOI.  Add the DOI to:

- `README.md`
- `CITATION.cff`
- `docs/original_geopt_heat_transfer_preliminary_report.md`

## LinkedIn or Blog Post

Use LinkedIn only as an announcement pointing to the GitHub release or Zenodo
DOI.  Keep the technical claims and reproducibility details in the repository.
