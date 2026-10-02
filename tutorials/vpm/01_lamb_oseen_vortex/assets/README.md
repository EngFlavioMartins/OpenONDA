# Lamb–Oseen analysis

Run the simulations and comparisons through the [case README](../README.md). It gives the physical inputs, method choices and analytical profile.

RWM Brownian scatter is numerical sampling noise. Plotters average signed velocity and vorticity across independent seeds at each physical time, then extract centres, radii and separation. Shading shows 95% ensemble uncertainty; it does not measure particle spread or physical turbulence. Definitions and files are in the [RWM methodology](references/rwm_statistical_methodology.md).

Diffusion broadens the finite columns axially and radially. `setup.py` reserves a physical spread margin of $3.6\sqrt{4\nu t_{\rm end}}$ beyond the initial support. DVH/GBD grid padding is additional numerical support. Recheck the domain when changing viscosity, spacing or duration.
