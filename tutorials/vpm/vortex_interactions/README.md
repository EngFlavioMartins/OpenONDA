# Vortex-ring interactions

This tutorial compares DNS, LES and stabilized LES for two three-dimensional
vortex-ring interactions:

| Interaction | DNS | LES | LES + stabilization |
|---|---|---|---|
| Leapfrogging | `leapfrog_dns` | `leapfrog_les` | `leapfrog_les_stabilized` |
| Collision | `collide_dns` | `collide_les` | `collide_les_stabilized` |

The leapfrogging rings have equal circulation and travel in the same direction.
The collision rings have opposite circulation and travel towards each other.

## Physical and numerical setup

Each ring uses the calibrated single-ring discretization from `vortex_ring/`:

| Quantity | Value |
|---|---:|
| Ring radius, `R` | `1.0` m |
| Core radius, `a` | `0.1` m |
| Circulation, `Gamma` | `pi` m2/s |
| Reynolds number, `Gamma/nu` | `3000` |
| Particle spacing, `h` | `0.035` m |
| Blob radius, `sigma` | `0.07` m |
| Particles per ring | `8772` |

The interaction cases therefore start with `17544` particles. They use the
same Gaussian core, broadband Widnall perturbation and `C_s = 0.20` LES
coefficient as the calibrated single-ring case.

All cases use transposed stretching, coupled RK2 integration and the same
treecode settings. The nominal timestep is

```text
Delta t = 20 h^2 / Gamma
```

and the common upper limit is 6000 steps. A case stops earlier if its particle
strength diverges; the plots retain all samples written before it stops.

## Stabilization

Stabilized LES uses the same LES model and adds filament splitting. Every five
steps, particles whose strength exceeds twice their reference strength are
split into two weaker particles. No remeshing or relaxation is used.

The comparison is the resulting stability ladder:

```text
DNS  ->  LES  ->  LES + stabilization
```

## Running

Run the six cases:

```sh
./allrun.sh
```

Run selected cases:

```sh
./allrun.sh leapfrog_les_stabilized collide_les_stabilized
```

Existing case directories are skipped. Remove one explicitly before rerunning
it:

```sh
./allclean.sh collide_les
```

## Figures

Generate PNG or PDF figures from whichever cases are available:

```sh
./allplot.sh
./allplot.sh pdf
```

The figures compare the methods separately for leapfrogging and collision:

- `rings_trajectory` shows the two ring paths;
- `rings_energy` shows kinetic-energy evolution;
- `rings_circulation` shows total particle-strength growth;
- `rings_stability` shows peak particle-strength growth on a logarithmic scale.

Each curve ends at the final available sample, making the relative survival of
DNS, LES and stabilized LES visible directly.
