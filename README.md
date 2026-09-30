# CHORD Analysis Pipeline

This is the repository for the CHORD analysis pipeline: the
[`caput.pipeline`](https://caput.readthedocs.io/) tasks, containers and
telescope models used to process CHORD data (loading X-engine data, flagging,
calibration, ...) and to simulate it. It builds on the shared radiocosmology
packages ([caput](https://github.com/radiocosmology/caput),
[draco](https://github.com/radiocosmology/draco),
[driftscan](https://github.com/radiocosmology/driftscan),
[cora](https://github.com/radiocosmology/cora)) and on
[chord_util](https://github.com/chord-observatory/chord_util), much as CHIME's
[ch_pipeline](https://github.com/chime-experiment/ch_pipeline) builds on them
and on ch_util.

Development follows the
[CHORD pipeline guidelines](https://github.com/chord-observatory/Pipeline),
which describe the workflow, reviews, coding rules, testing and releases for
all CHORD analysis software. Please read them before contributing.

## Installation

1. Prerequisites: `python3`, `pip`, `virtualenv`, and SSH access to GitHub for
   the repositories listed in `requirements.txt` (chord_util is private).
2. From the repository root, create and populate the environment (the
   dependencies are cloned into `venv/src`):

    ```sh
    bash mkvenv.sh
    ```

   - For a clean environment that ignores system packages, add `-i`.
   - On the site computers, use `-s` to install from
     `requirements_site_computers.txt`.
   - The default install is `pip install -e .`; for the old
     `setup.py develop` flow add `-l`.
   - `bash mkvenv.sh -h` lists the other options (`-v` environment path, `-n`
     prompt name, `-e` source path).
3. Activate the environment:

    ```sh
    source venv/bin/activate
    ```

4. Check that it works:

    ```sh
    python -c "import chord_pipeline; print(chord_pipeline.__version__)"
    ```

## Structure

- `chord_pipeline/core/`: the basic building blocks
  - `container.py`: containers for CHORD data, e.g. `CHORDTimeStream`, a draco
    `TimeStream` holding X-engine data with its data quality, timing and
    per-input information
  - `io.py`: tasks to find and load data, e.g. `QueryAcquisitionFiles` and
    `LoadCorrDataFiles` (X-engine files, via `chord_util.andata`)
  - `telescope.py`: driftscan telescope models of the CHORD array
- `chord_pipeline/analysis/`: tasks operating on data, e.g. `SelectInputs` and
  `RemoveRFIMonitors`
- `chord_pipeline/processing/`: tools to run the pipeline in production
- `chord_pipeline/synthesis/`: tasks for simulating data

New tasks go into the most closely matching module; ask if it isn't clear.
Code that is not specific to the pipeline belongs in chord_util, and code that
is not specific to CHORD belongs upstream in caput, draco or driftscan.

## Example: loading X-engine data

This pipeline configuration concatenates all the files of an acquisition into
a single timestream, distributed over frequency, and saves it:

```yaml
pipeline:
  tasks:
    - type: chord_pipeline.core.io.QueryAcquisitionFiles
      out: filelist
      params:
        acquisition: acq_20260911_232919_986789057

    - type: chord_pipeline.core.io.LoadCorrDataFiles
      requires: filelist
      out: tstream
      params:
        exclude_datasets: [eval, evec, erms]
        save: true
        output_name: "tstream_{tag}.h5"

    # Optionally, keep only the dish inputs
    - type: chord_pipeline.analysis.inputs.RemoveRFIMonitors
      in: tstream
      params:
        save: true
        output_name: "tstream_dish_{tag}.h5"
```

Run it with `mpirun -np <N> caput-pipeline run config.yaml`. A full-band,
48-input timestream needs ~32 GB of memory per hour of data; use
`channel_range`, `input_type: dish` or `files_per_container` to reduce it.

## Tests

The tests use small synthetic X-engine files from `tests/xengine_testdata.py`,
so no real data is needed. With the environment activated:

```sh
python -m pytest                # serial
mpirun -np 3 python -m pytest   # the same tests under MPI
```

`tests/test_pathfinder_data.py` also checks the loader against real pathfinder
X-engine data. It runs only where the data is readable (on the cluster, with
rrg-kmsmith access; a few GB of memory, a few minutes) and is skipped
elsewhere. Point it at other data with `CHORD_XENGINE_DATA` (directory of
acquisitions) and `CHORD_XENGINE_ACQ` (acquisition name).

## Development

The [CHORD pipeline guidelines](https://github.com/chord-observatory/Pipeline)
have the details; in short:

- **Branches.** Work in a short-lived feature branch named with your initials
  (e.g. `ab/short-description`) off `main`, and merge it back through a pull
  request once it is finished and tested. Don't commit to `main` directly, and
  don't keep long-lived personal branches.
- **Pull requests** need passing checks and an approval from a code owner, and
  are squash-merged. Their titles follow the conventional commit format (e.g.
  `feat(io): add LoadCorrDataFiles`), as they become the release notes.
- **Code** is formatted with `ruff format`, kept clean with `ruff check`, and
  documented with NumPy style docstrings; every change comes with tests.
- **Dependencies** are listed in `requirements.txt` (and
  `requirements_site_computers.txt`). Use `https` for public repositories and
  `ssh` only for private ones. Processing runs used for science should use
  released versions or exact commits, recorded with `save_versions` in the
  pipeline configuration.
- **Reproducibility.** The aim is that the input data together with this
  repository (and the recorded versions of its dependencies) always gives the
  same output.
