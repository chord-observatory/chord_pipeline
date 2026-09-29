# CHORD Pipeline

## Quickstart

To get set up:

1. Prereqs: `python3`, `pip`, `virtualenv`, and SSH access to the GitHub orgs listed in `requirements.txt`.
2. From the repo root, create and populate the env (clones deps to `venv/src`):  
   `bash mkvenv.sh`
   - For a clean env that ignores system packages, add `-i`: `bash mkvenv.sh -i`.
   - Default install is modern `pip install -e .`; if you need the old `setup.py develop` flow, add `-l`.
3. Activate the env: `source venv/bin/activate`.
4. Test the import works: `python -c "import chord_pipeline; print(chord_pipeline.__version__)"`.

If you prefer a different location/name, use `mkvenv.sh -h` for options (`-v` for env path, `-n` for prompt name, `-e` for source path).

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
