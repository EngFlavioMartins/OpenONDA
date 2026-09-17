# FVM–VPM coupler: current algorithm

Subject: How the updated FVM–VPM coupling works

The coupler combines a resolved finite-volume (FVM) region around the body with a
vortex-particle method (VPM) wake outside it. This is a synchronized state exchange:
the FVM owns the renewal region and the VPM owns the outer wake; their states are not
added together.

Before a run, it checks common fluid properties and an integer, fixed FVM-to-VPM time
step ratio. It then builds a fixed body-aware lattice and a buffer that covers particle
advection during one coupling interval plus the complete M4-prime kernel support.

## Key simplified equations

The FVM exports the cell-centred velocity gradient with the convention
$G_{ij}=\partial u_j/\partial x_i$. With the Gauss gradient scheme, its cell value
is the face-flux approximation

$
G_{c,ij} \approx \frac{1}{V_c}\sum_{f\in\partial c} S_{f,i}\,u_{f,j},
$

where $V_c$ is cell volume, $\mathbf{S}_f$ is the outward face-area vector, and
$\mathbf{u}_f$ is linearly interpolated to the face (with the boundary normal
component corrected from the boundary condition). The optional least-squares scheme
instead minimizes $\sum_n w_{cn}^2\|\mathbf u_n-\mathbf u_c-G_c^T(\mathbf x_n-\mathbf
x_c)\|^2$ over the cell stencil.

The transfer then forms

$
\boldsymbol\omega_c=\nabla\times\mathbf u_c
=\left(G_{yz}-G_{zy},\;G_{zx}-G_{xz},\;G_{xy}-G_{yx}\right),
\qquad
\boldsymbol\Gamma_c=V_c\boldsymbol\omega_c.
$

Here $\boldsymbol\Gamma$ is particle vortex strength, in m$^3$/s. On the renewal
lattice, the conceptual authority blend is
$\boldsymbol\Gamma=(1-\eta)\boldsymbol\Gamma_{\mathrm{VPM}}+
\eta\boldsymbol\Gamma_{\mathrm{FVM}}$, where $\eta$ smoothly changes from one in
the FVM-owned interior to zero at release. The buffer length is
$L_{\mathrm{buffer}}=s\|\mathbf U_\infty\|\Delta t_{\mathrm{couple}}+2h$.

For the cylinder's mixed-vorticity boundary, the VPM supplies
$\mathbf g_t=(I-\mathbf n\mathbf n^T)J_{\mathrm{VPM}}\mathbf n$, the tangential
part of $d\mathbf u/dn$. The FVM reconstructs the face velocity as
$\mathbf U_b=(I-\mathbf n\mathbf n^T)\mathbf U_{\mathrm{owner}}+
\mathbf n U_n+d\mathbf g_t$.

For each coupling interval, it:

1. Advance the VPM cloud by one macro-step, with output deferred.
2. Sample its trace on the outer FVM patch. The cylinder uses the mixed-vorticity
   trace: normal velocity plus tangential normal-gradient information.
3. Advance the FVM through its integer number of substeps to the same time. With
   interface iteration enabled, repeat the FVM/renewal sweep from the same accepted
   start state and fixed VPM predictor until the interface residuals meet tolerance or
   the sweep limit is reached; only the final sweep publishes output.
4. Gather accepted FVM velocity and gradient, derive vorticity, and renew the overlap
   on the buffered M4-prime lattice. Existing inner particles are scattered to that
   lattice, blended with the FVM target according to FVM authority, and particles
   beyond the release buffer remain as the outer wake.
5. Apply solid masking, conservative local redistribution after pruning, and
   circulation/impulse recovery. Commit the new cloud atomically; a failed quality or
   capacity check restores the prior VPM state.
6. Refresh boundary history, then write due samples, diagnostics, and backups.

The main recent fix concerns weak wake vorticity at the FVM-to-VPM release surface.
Previously, the transfer could use a stronger pruning threshold near that surface,
which could remove weak but physical wake structures as FVM authority decayed. The
new rule uses the configured pruning threshold only in the FVM-owned interior and
smoothly reduces it through the overlap to the VPM GBD solver's own absolute
vorticity floor at release. This removes the competing cutoff and preserves weak
wake content after it becomes VPM-owned.

Equivalently, the local particle-strength pruning threshold is
$\tau=\tau_{\mathrm{GBD}}+(\tau_{\mathrm{interior}}-\tau_{\mathrm{GBD}})\eta$,
instead of increasing the threshold near release.

Each transfer records particle counts, circulation, moments, pruning, amplification,
closure, and interface residuals. This makes the exchange auditable, but a completed
run is not by itself a convergence claim; resolution and validation still matter.
