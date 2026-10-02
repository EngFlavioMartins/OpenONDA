# Clean installation VM

The local VirtualBox machine `OpenONDA-Ubuntu-24.04` provides a repeatable fresh
Ubuntu installation test. It uses 1536 MiB RAM, one virtual CPU capped at 50%,
4 GiB of guest swap, and a dynamically allocated 64 GiB disk. It runs headless
with NAT networking and SSH forwarded only on `127.0.0.1:22524`.

The `clean-before-openonda` snapshot contains Ubuntu 24.04, Git, curl, CA
certificates, and SSH access. Neither Conda nor OpenONDA is installed. Restore
this snapshot only when the VM is powered off; restoring discards subsequent
changes in the guest.

## Repeat the test

On the host:

```bash
VBoxManage snapshot OpenONDA-Ubuntu-24.04 restore clean-before-openonda
VBoxManage startvm OpenONDA-Ubuntu-24.04 --type headless
ssh -i "$HOME/VirtualBox VMs/OpenONDA-Ubuntu-24.04/access/id_ed25519" \
    -o UserKnownHostsFile="$HOME/VirtualBox VMs/OpenONDA-Ubuntu-24.04/access/known_hosts" \
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
history is `solution/history.csv`, and the figure is
`figures/taylor_green_decay.png`. The installer also verifies thesis PNG/PDF
export, headless ParaView rendering, two-process MPI/PETSc, native meshing,
FVM/VPM stepping, output, restart, and coupled continuation.

Finish with `sudo poweroff` inside the guest so the VM uses no host RAM or CPU
between tests. For an already installed snapshot, start a new terminal and run
`conda activate OpenONDA` before using the solver.

## Baseline provenance

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
