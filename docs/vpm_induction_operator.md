# Canonical VPM induction operator

This document freezes the discrete equations used by the VPM particle
evolution.  It is a numerical contract for direct summation, treecode, and
FMM implementations; it is not a statement that those implementations use
the same computational algorithm.

## State and kernel notation

For the active particles, let

\[
  \mathbf x_i \in \mathbb R^3,\qquad
  \boldsymbol\Gamma_i \in \mathbb R^3,\qquad
  \sigma_i > 0,
\]

denote position, vortex strength, and core radius.  The inviscid particle
state is the pair \((\mathbf x,\boldsymbol\Gamma)\).  Core-radius changes due
to diffusion or another explicitly selected physical operation are outside
this inviscid right-hand side.

Each supported isotropic radial vortex-blob kernel supplies two dimensionless
functions:

* \(q(\rho)\), the enclosed-circulation factor including the
  \(1/(4\pi)\) Biot--Savart constant;
* \(\zeta(\rho)\), the normalized radial vorticity kernel.

Their common far-field limits are

\[
  q(\rho) \longrightarrow q_\infty = \frac{1}{4\pi},
  \qquad
  \zeta(\rho) \longrightarrow 0.
\]

The kernel-specific functions, constants, and tail behaviour belong to the
kernel implementation.  The induction method must not select a kernel by
duplicating its formula.

## Particle-particle radius pairing

For a particle target \(i\) and particle source \(j\), define

\[
  \mathbf r_{ij}=\mathbf x_i-\mathbf x_j,\qquad
  r_{ij}=\|\mathbf r_{ij}\|,\qquad
  a_{ij}=\frac{\sigma_i+\sigma_j}{2},\qquad
  \rho_{ij}=\frac{r_{ij}}{a_{ij}}.
\]

The production particle-particle operator uses the symmetric arithmetic mean
\(a_{ij}\) for both directions of a pair.  Consequently,

\[
  a_{ij}=a_{ji},
\]

even when the particle core radii are unequal.  This is a discrete choice,
not an assertion that two unequal blobs have a physical convolution radius.
The convolution radius used by energy or other diagnostic kernels is a
separate diagnostic definition and must not be substituted into induction.

## Induced velocity

The self-induced particle velocity is

\[
  \mathbf u_i^{\mathrm{self}}
  = \sum_{\substack{j=0\\j\ne i}}^{N-1}
    \frac{q(\rho_{ij})}{r_{ij}^{3}}
    \left(\boldsymbol\Gamma_j\times\mathbf r_{ij}\right).
\]

This is equivalent to the implementation form
\(-q(\rho_{ij})\,\mathbf r_{ij}\times\boldsymbol\Gamma_j/r_{ij}^3\).
The self term is excluded whenever \(r_{ij}\leq\varepsilon\), with the
same declared coincidence tolerance used by the backend.  A centered radial
blob has zero self-induced velocity.

The direct reference operator evaluates the regularized expression for every
non-self pair.  There is no physical distance cutoff in the direct equation.

## Canonical vortex-strength rate

The retained production equation is the conservative pairwise transposed
equation.  Define

\[
  A_{ij}=\frac{q(\rho_{ij})}{r_{ij}^{3}},
  \qquad
  B_{ij}=\frac{3q(\rho_{ij})-\zeta(\rho_{ij})\rho_{ij}^{3}}
                 {a_{ij}^{5}\rho_{ij}^{5}}.
\]

For every non-self pair, the contribution to the target strength rate is

\[
  \mathbf s_{ij}^{\mathrm T}
  = A_{ij}\left(\boldsymbol\Gamma_i\times\boldsymbol\Gamma_j\right)
    +B_{ij}
      \left[\boldsymbol\Gamma_i\cdot
        \left(\mathbf r_{ij}\times\boldsymbol\Gamma_j\right)\right]
      \mathbf r_{ij},
\]

and

\[
  \dot{\boldsymbol\Gamma}_i^{\mathrm{self}}
  = \sum_{\substack{j=0\\j\ne i}}^{N-1}
    \mathbf s_{ij}^{\mathrm T}.
\]

The superscript \(\mathrm T\) identifies the transposed strength equation;
it does not mean that every gradient-based evaluator is interchangeable with
this pairwise sum.  The classical \(J\boldsymbol\Gamma\) and symmetric
\(\tfrac12(J+J^\mathsf T)\boldsymbol\Gamma\) formulations are study-only
comparators unless separately justified for production.

## Conservation and pair symmetry

The coefficients satisfy \(A_{ij}=A_{ji}\) and \(B_{ij}=B_{ji}\).  The two
directions of each pair cancel algebraically:

\[
  \mathbf s_{ij}^{\mathrm T}+\mathbf s_{ji}^{\mathrm T}=\mathbf 0.
\]

Therefore the unforced self-induced operator obeys

\[
  \sum_i\dot{\boldsymbol\Gamma}_i^{\mathrm{self}}=\mathbf 0.
\]

This cancellation is part of the declared discrete equation.  It must not be
replaced by silently projecting a non-pairwise approximation onto zero total
rate.  If a future approximate method needs a conservation correction, it must
report the uncorrected defect and the correction norm against an explicit
tolerance.

## Gradient relation and unequal radii

For a particle-particle pair using the same symmetric radius, the velocity
gradient contribution \(J_{ij}=\nabla\mathbf u_{ij}\) obeys

\[
  J_{ij}^{\mathsf T}\boldsymbol\Gamma_i
  = \mathbf s_{ij}^{\mathrm T}.
\]

Thus, for equal radii—and also for unequal radii when the gradient uses the
same pair radius—the pairwise transposed rate equals the transpose-gradient
contraction in exact arithmetic.

An arbitrary target point is different: it has no target blob radius.  A
source-only field evaluator therefore uses \(\sigma_j\), not
\((\sigma_i+\sigma_j)/2\), in the source contribution.  At a particle
location with unequal radii, contracting that source-radius gradient with
\(\boldsymbol\Gamma_i\) is a different discrete operator and is not the
canonical strength rate.  This distinction is intentional and must remain
explicit in APIs and tests:

| Operation | Radius used for source \(j\) |
| --- | --- |
| Particle self-induced velocity | \(a_{ij}=(\sigma_i+\sigma_j)/2\) |
| Canonical particle strength rate | \(a_{ij}=(\sigma_i+\sigma_j)/2\) |
| Source-only arbitrary-target velocity/gradient | \(\sigma_j\) |

An induction method may return a velocity gradient as an auxiliary field, but
it must return the canonical \(\dot{\boldsymbol\Gamma}\) directly when the
selected strength equation requires it.  It must not label a source-radius
gradient contraction as the transposed pairwise rate.

## Near field, far field, and cutoff

The exact direct reference includes all non-self pairs.  A hierarchical method
may partition an interaction into near and far regions, but the partition is
a numerical approximation mechanism, not a change to the equation.

For a requested error tolerance, a near-field threshold
\(\rho_{\mathrm{near}}\) is valid only when the kernel's regularization
correction and any expansion remainder outside that threshold are bounded by
that tolerance for every represented core radius.  The threshold must
therefore be kernel- and tolerance-aware, and for variable radii it must use a
safe cell bound (at least the largest represented radius).

In the far limit, the regularized pair coefficients become

\[
  A_{ij}^{\infty}=\frac{q_\infty}{r_{ij}^{3}},
  \qquad
  B_{ij}^{\infty}=\frac{3q_\infty}{r_{ij}^{5}},
\]

which is the singular Biot--Savart/Laplace far field.  A method may use this
limit only after its declared regularization and multipole errors are within
the requested tolerance.  The historical direct velocity evaluation has no
cutoff, while historical gradient paths use different hard-coded cutoffs;
those implementation details are not separate physical equations and must
not be propagated into the new induction contract.

## External contributions

The stage right-hand side combines the self-induced result with explicitly
declared external providers at the actual RK stage state and time:

\[
  \dot{\mathbf x}_i
    =\mathbf u_i^{\mathrm{self}}+\mathbf u_i^{\mathrm{ext}},
  \qquad
  \dot{\boldsymbol\Gamma}_i
    =\dot{\boldsymbol\Gamma}_i^{\mathrm{self}}
     +\mathbf s_i^{\mathrm{ext}}.
\]

Providers must declare whether they supply:

* velocity only: contributes to \(\dot{\mathbf x}\) and contributes no
  strength rate by implication;
* velocity and gradient: contributes velocity and, when the model declares
  external stretching active, contributes
  \((J_i^{\mathrm{ext}})^\mathsf T\boldsymbol\Gamma_i\) to the retained
  transposed rate;
* a direct strength-rate contribution: contributes the supplied
  \(\mathbf s_i^{\mathrm{ext}}\) without reconstructing it from a gradient.

Freestream, body, VLM, and FVM/VPM-overlap inputs must pass through this
explicit boundary.  A provider must not read the accepted particle state or
silently omit an external gradient required by the configured physical model.

The self-induced conservation statement applies only to the self-induced
pairwise term.  External forcing may change total vector strength when its
declared rate does so.
