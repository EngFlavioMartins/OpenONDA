# Clean installation VM

The local VirtualBox machine `OpenONDA-Ubuntu-24.04` provides a repeatable fresh
Ubuntu installation test. It uses 1536 MiB RAM, one virtual CPU capped at 50%,
4 GiB of guest swap, and a dynamically allocated 64 GiB disk. It runs headless
with NAT networking and SSH forwarded only on `127.0.0.1:22524`.

The `clean-before-openonda` snapshot contains Ubuntu 24.04, Git, curl, CA
certificates, and SSH access. Neither Conda nor OpenONDA is installed. Restore
this snapshot only when the VM is powered off; restoring discards subsequent
changes in the guest.

The `openonda-verified` snapshot preserves the successful installation and its
tutorial results. The VM is left powered off in this installed state.

## Repeat the test

On the host:

```bash
VBoxManage snapshot OpenONDA-Ubuntu-24.04 restore clean-before-openonda
VBoxManage startvm OpenONDA-Ubuntu-24.04 --type headless
ssh -i "$HOME/VirtualBox VMs/OpenONDA-Ubuntu-24.04/access/id_ed25519" \
    -o "UserKnownHostsFile=\"$HOME/VirtualBox VMs/OpenONDA-Ubuntu-24.04/access/known_hosts\"" \
    -p 22524 tester@127.0.0.1
```

Wait for Ubuntu to boot before connecting. In the guest, run the same commands
as in the installation documentation:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

The tutorial advances a 24-by-24 Taylor–Green vortex for ten steps. Its fresh
history is `solution/history.csv`, and its figures are
`figures/taylor_green_decay.png` and `.pdf`. The installer also verifies thesis PNG/PDF
export, headless ParaView rendering, two-process MPI/PETSc, native meshing,
FVM/VPM stepping, output, restart, and coupled continuation.

Finish with `sudo poweroff` inside the guest so the VM uses no host RAM or CPU
between tests. For an already installed snapshot, start a new terminal and run
`conda activate OpenONDA` before using the solver.

## Verified result — 2026-10-02

The final test started from `clean-before-openonda`, confirmed that neither
Miniforge nor the checkout existed, and cloned commit
`429aa824a17068d1b26334e7863fc914eedb12ae` from the public `development` branch.
The installation and the two tutorial scripts above completed in sequence,
without guest-side fixes, extra packages, or manual configuration.

- Ubuntu 24.04.5, Python 3.11.16, editable OpenONDA installation.
- All built-in installation checks passed: 19 tutorial resources, 10 direct
  tutorial entry points, native Cartesian meshing, FVM, VPM, restart, coupled
  continuation, thesis PNG/PDF, headless ParaView, and two-process MPI/PETSc.
- Taylor–Green completed ten steps to `t = 0.05 s`. All history values were
  finite; final kinetic-energy relative error was `7.0371e-5`, and maximum
  continuity error was `1.1601e-15 s^-1`.
- A new interactive shell activated `OpenONDA` normally and imported it from
  outside the checkout. The documented `openonda tutorial run` and `plot`
  commands also passed in `~/first-flow`, reproducing the same final values.
- Deactivation restored the original graphics-driver environment.
- Installer regression tests: 82 passed; 12 Zsh variants skipped because Zsh
  was absent on the test host. Changed Python files passed Ruff checks and
  formatting checks; the shell installers passed Bash syntax checks.

The fresh-machine tests led to fixes for the Miniforge checksum URL, stalled
downloads, Mesa vendor discovery, the shared figure-export API, and standard
input handling when installation commands are pasted together. The concurrent
tutorial cleanup also removed a retired helper with machine-specific paths.
These fixes are part of the installer and shared verification code.

Local evidence is retained under `build/vm-validation/`: `installation.log`,
`verified-install.json`, `new-terminal.log`, `openonda-install-validation.json`,
the tutorial history, and the generated PNG/PDF.

## Independent backup

A powered-off copy of the clean baseline was exported to:

```text
~/VirtualBox VMs/Backups/OpenONDA-Ubuntu-24.04-clean/baseline.ova
```

The approximately 755 MiB appliance contains the guest disk and VM configuration.
Both entries passed their exported SHA-256 checksum checks. `SHA256SUMS` beside
the appliance records the checksum of the complete OVA:

```text
e69ac538c0508f0db3c9a110da04be340f68d51cf83c22de8e7ba8380e8840c6
```

Use VirtualBox's **Import Appliance** to recover it independently of the original
VM's snapshots. Its adjacent `access/` directory contains the matching SSH key
and host record; retain these private files with the backup. The backup and
keys are local files, not repository contents.

## Baseline environment

The machine was created on 2026-10-02 with VirtualBox 7.2.6 from Ubuntu's
[official Noble cloud OVA](https://cloud-images.ubuntu.com/noble/20260926/).
The downloaded `noble-server-cloudimg-amd64.ova` matched the published SHA-256:

```text
513a22ebe3982b9387b038f8a2a6dad1af980d4623dca03511312e55ba620b1a
```

Cloud-init prepared the OS and a dedicated SSH key before the snapshot. The VM,
snapshots, and private SSH key live under
`~/VirtualBox VMs/OpenONDA-Ubuntu-24.04/`, outside the repository. Keep that
directory to retain the machine and its access key.

## macOS

No macOS VM was created on this Dell Linux host. Oracle documents hardware and
licensing restrictions for macOS guests on non-Apple computers; its macOS guest
feature is also experimental. See section 16.1.2 of the
[VirtualBox 7.2 guide](https://download.virtualbox.org/virtualbox/7.2.6/UserManual.pdf).
Actual macOS installation testing needs an Apple host or a macOS CI runner.
The nightly workflow exercises the installer on an Apple Silicon macOS runner;
dependency resolution alone is not an executed macOS installation test.
