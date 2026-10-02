"""Install the tutorial TeX tools inside the active Conda environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile

PACKAGES = (
    # TeX Live records dependencies at collection level; installing newpx
    # alone does not install the LaTeX packages used by its style files.
    "collection-latexrecommended",
    "collection-latexextra",
    "collection-fontsrecommended",
    "collection-mathscience",
    "collection-plaingeneric",
    # NewPX shares font encodings with NewTX even when newtxtext is not loaded.
    "newpx",
    "newtx",
    "type1cm",
    "cm-super",
    "dvipng",
)


def download(url: str, destination: Path) -> None:
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "3", "--output", str(destination), url],
        check=True,
    )


def install_tex(prefix: Path) -> None:
    if not (prefix / "conda-meta").is_dir():
        raise RuntimeError("Run source install.sh to install the OpenONDA environment.")
    tex_root = prefix / "share" / "openonda-tex"
    managers = list(tex_root.glob("bin/*/tlmgr"))
    if not managers:
        if tex_root.exists():
            raise RuntimeError(f"Incomplete TeX installation at {tex_root}")
        system = "darwin" if platform.system() == "Darwin" else "linux-x86_64"
        with tempfile.TemporaryDirectory(prefix="openonda-tex-", dir=prefix) as directory:
            staging = Path(directory)
            metadata = staging / "release.json"
            download(
                "https://api.github.com/repos/rstudio/tinytex-releases/releases/latest", metadata
            )
            release = json.loads(metadata.read_text())
            assets = [
                asset
                for asset in release["assets"]
                if asset["name"].startswith(f"TinyTeX-1-{system}-v")
                and asset["name"].endswith(".tar.xz")
            ]
            if len(assets) != 1:
                raise RuntimeError(f"No unique TinyTeX release for {system}")
            asset = assets[0]
            archive = staging / "tex.tar.xz"
            download(asset["browser_download_url"], archive)
            with archive.open("rb") as stream:
                digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != asset.get("digest"):
                raise RuntimeError("TinyTeX download failed its SHA-256 check")
            unpacked = staging / "unpacked"
            with tarfile.open(archive) as bundle:
                bundle.extractall(unpacked, filter="data")
            roots = [item for item in unpacked.iterdir() if (item / "bin").is_dir()]
            if len(roots) != 1:
                raise RuntimeError("Unexpected TinyTeX archive layout")
            tex_root.parent.mkdir(parents=True, exist_ok=True)
            roots[0].rename(tex_root)
        managers = list(tex_root.glob("bin/*/tlmgr"))
    if len(managers) != 1:
        raise RuntimeError(f"Cannot locate TinyTeX's package manager in {tex_root}")
    manager = managers[0]
    manifest = tex_root / ".openonda-packages.json"
    # A working TeX tree remains usable after the yearly upstream repository
    # rollover, and rerunning the installer need not download it again.
    if not manifest.is_file() or json.loads(manifest.read_text()) != list(PACKAGES):
        subprocess.run([str(manager), "update", "--self"], check=True)
        subprocess.run([str(manager), "install", *PACKAGES], check=True)
        manifest.write_text(json.dumps(PACKAGES) + "\n")
    # TeX tools become available through ordinary Conda activation. Keep their
    # actual tree inside this environment, with no system-wide TeX changes.
    for executable in manager.parent.iterdir():
        link = prefix / "bin" / executable.name
        relative_target = executable.relative_to(prefix)
        target = Path("..") / relative_target
        if link.is_symlink() and link.readlink() == target:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"Refusing to replace an existing TeX command: {link}")
        link.symlink_to(target)
    print(f"Thesis plotting tools installed in {tex_root}.")


if __name__ == "__main__":
    install_tex(Path(sys.prefix))
