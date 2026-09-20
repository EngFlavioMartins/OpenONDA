# Cube reference evidence

The saved historical wall-spacing requests 0.12, 0.10, 0.08 and 0.06 m produced
nominal wall spacings 0.09, 0.075, 0.06 and 0.045 m. Recorded cell counts are
98,424, 163,844, 294,872 and 692,604. The historical dense mesh has 2,233,688
cells. New geometric-campaign results must remain separate from these records.

Over the historical 15–30 s window, mean Cd is 1.00498, 0.999316, 0.971000 and
0.965156. Medium/fine mean Cd differs by 0.606%, but fine-grid half-window drag
drift is 11.9%, drag fluctuation RMS changes by 33.4%, and wake-profile changes
are approximately 6% and 10%. Force-derived shedding frequencies remain
unresolved. These records do not establish grid independence.

A read-only memory audit of saved four-rank `performance.jsonl` records found
summed rank peak RSS of 5.98 GiB for 692,604 cells and 10.66 GiB for 2,233,688
cells; the latter's root-rank peak was 4.28 GiB. Rank peaks need not occur
simultaneously, and two-rank allocation differs. The proposed roughly
2.34-million-cell grid needs approximately 11–12 GiB allowance, with uncertainty.
The resource queue releases only this assessment's verified coupled jobs; it
does not inspect or stop unrelated applications.

At t≈44 s, the 68,271-cell geometric coarse run held about 187 MiB of outputs,
including 12.3 MiB for two rolling checkpoints. Extrapolating native field sizes
and current diagnostic/profile cadence suggests approximately 4 GiB for the
whole cube spatial/temporal campaign, before OS swap and other studies.

Configuration and current commands are in the reference tutorial's
[grid assessment](../../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/GRID_CAMPAIGN.md).
