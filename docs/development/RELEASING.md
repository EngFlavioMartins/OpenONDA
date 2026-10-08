# Publishing a release

`main` is the public, installable branch; `development` carries day-to-day
work. Publish from a clean `development` whose tests pass.

## 1. Check the release candidate

```bash
source install.sh --dev
python tests/run_isolated.py -m "not gpu"
CUDA_PATH=/usr python tests/run_isolated.py -m gpu      # on a CUDA machine
python -m build && twine check dist/*
```

Install the wheel in a fresh virtual environment and run
`openonda-verify-install` outside the checkout. Re-run any tutorial whose
solver changed and commit its refreshed figures.

## 2. One-time history clean-up (before the first public release from `development`)

The history of `development`, `openfoam-journal` and `openfoam-wrapper`
contains two copyrighted journal articles (`docs/LBM.pdf`, Cheng et al. 2015;
`tutorials/vpm/01_lamb_oseen_vortex/assets/references/the-physical-mechanism-for-vortex-merging.pdf`,
Cerretelli & Williamson 2003) and about 5.9 GB of retired tutorial outputs,
renamed copies and studies. `main` contains none of them. Publish the current
tree to `main` as one commit, then replace the remote branches that carry the
old history:

```bash
git fetch origin
git switch main && git reset --hard origin/main
git merge --squash development
git commit -m "OpenONDA <version>: native FVM, VPM, VLM and coupled solvers"
git push origin main

# Continue development from the published commit.
git branch -f development main
git push --force-with-lease origin development
git push origin --delete openfoam-journal openfoam-wrapper
git branch -D codex/cfd-nomenclature
```

Keep a local mirror of the old history first if it is still needed
(`git clone --mirror`), outside the published repository. Ask GitHub support
to purge cached views of the removed objects; unreferenced commits remain
reachable by hash until GitHub garbage-collects them. Other clones of the
repository must re-clone or reset to the new `development`.

## 3. Tag and archive

1. Set the version in `openonda/core/version.py` and `CITATION.cff`, and move
   the `Unreleased` entries of `CHANGELOG.md` under that version.
2. Tag `main` (`git tag v<version> && git push origin v<version>`) and publish
   a GitHub release from the tag. The release workflow builds, checks and
   uploads the package to PyPI.
3. Zenodo archives the GitHub release and mints a version DOI; add it to
   `CITATION.cff` and the changelog link.
4. Attach large media (tutorial GIFs and videos) to the release if they are
   moved out of the repository.
