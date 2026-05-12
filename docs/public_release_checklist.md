# Public Release Checklist

This checklist is for publishing the preliminary Thermal GeoPT results on
GitHub in a citable form.

## Before Release

- Confirm that the public report is up to date:
  - `docs/original_geopt_heat_transfer_preliminary_report.md`
  - `docs/original_geopt_heat_transfer_preliminary_report_ja.md`
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
  - local PDF copies
  - private discussion notes
  - large generated figures
- Confirm the public wording is scoped as preliminary checkpoint-transfer
  evidence, not a paper-level claim.
- Confirm internal project labels are described as internal IDs:
  - `D1`: first OpenFOAM-solved solid-conduction dataset family
  - `M4`: Benchmark A
  - `M5`: Benchmark B
  - `R4`: project-local thermal-specific pretraining prototype
- Check `CITATION.cff`.
- Confirm license terms:
  - code: MIT
  - reports and figures: CC-BY-4.0

## Suggested Tag

```bash
git tag -a v0.1.0-preliminary -m "Preliminary GeoPT checkpoint transfer in OpenFOAM-solved heat-sink conduction"
git push origin v0.1.0-preliminary
```

## Suggested GitHub Release Title

```text
v0.1.0-preliminary: Original GeoPT checkpoint transfer in OpenFOAM-solved heat-sink conduction
```

## Suggested Release Notes

```text
This preliminary release reports checkpoint-transfer evidence for applying the original GeoPT pretrained checkpoint to OpenFOAM-solved heat-sink solid-conduction surrogate modeling.

A Transolver initialized from shape-compatible tensors of the original GeoPT checkpoint improves mean Relative L2 over scratch training by about 10-17% at 25- and 50-training-case settings across two generated heat-sink conduction benchmarks.

A small project-local thermal-specific pretraining prototype, internal ID R4, shows negative transfer under the same evaluation protocol.

These results are preliminary and should not be read as a controlled ablation of the full Thermal GeoPT concept.

Main reports:
- docs/original_geopt_heat_transfer_preliminary_report.md
- docs/original_geopt_heat_transfer_preliminary_report_ja.md
```

## Zenodo

After creating the GitHub release, archive the release with Zenodo to obtain a
DOI.  Add the DOI to:

- `README.md`
- `CITATION.cff`
- `docs/original_geopt_heat_transfer_preliminary_report.md`
- `docs/original_geopt_heat_transfer_preliminary_report_ja.md`

Notes:

- GitHub-Zenodo integration does not support pre-reserving a DOI before the
  GitHub release.
- Zenodo issues a version DOI and a concept DOI after the release is archived.
- The original GeoPT checkpoint is external and is not redistributed here.

## LinkedIn or Blog Post

Use LinkedIn only as an announcement pointing to the GitHub release or Zenodo
DOI.  Keep the technical claims and reproducibility details in the repository.

Suggested short wording:

```text
I released a preliminary technical report on applying the original GeoPT checkpoint to OpenFOAM-solved heat-sink solid-conduction surrogate modeling.

The result is preliminary checkpoint-transfer evidence: a Transolver initialized from shape-compatible original GeoPT tensors improved mean Relative L2 over scratch by about 10-17% in two low-data heat-sink conduction benchmarks.

Repository/DOI: ...
```
