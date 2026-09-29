# Renewal cadence and the current exchange algorithm

The cylinder sensitivity study varies `exchange_dt`. In the current algorithm this changes the actual injection/renewal rate: the transfer runs at every accepted exchange time. It also changes integration and boundary exchange errors. Its results therefore measure sensitivity to the **combined exchange clock**, rather than an independently held renewal cloud.

There is no independent production renewal-rate setting. The transfer diagnostic interval changes reporting only. Keeping the integration clock while skipping transfer calls would change the numerical algorithm, not provide an equivalent performance setting.

`source/coupler/interface_iteration.py` assumes a renewed FVM-to-VPM field for each interface sweep over the fixed VPM predictor. Skipping that update removes the feedback being converged and can produce a falsely small interface residual. `source/coupler/backup.py` reconstructs the transfer count from the coupling step, which also assumes one accepted renewal per exchange. Transfer support currently covers one `coupling_time_step`; a longer-held cloud would require support for elapsed advection and diffusion between actual renewals.

An independent held-cloud comparison would require a defined ownership rule for the held overlap field, conservation of transferred circulation and impulse, accurate residuals while the cloud is held, and distinct provisional/accepted emission counts. Tests would need to cover one-renewal-per-step equivalence, initial synchronization, rollback, native restart in both renewal and held phases, MPI agreement, and kernel support across the entire held interval. The changed algorithm would need a restart identity so older checkpoints cannot silently resume with changed physics.

No independent renewal algorithm or additional public control was introduced during this cleanup. The existing exchange-clock study is a valid sensitivity test for the implemented algorithm, with the confounding stated above. Independent renewal sensitivity remains a separate algorithm qualification requirement; it has not been implemented or qualified.
