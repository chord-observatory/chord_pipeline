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
