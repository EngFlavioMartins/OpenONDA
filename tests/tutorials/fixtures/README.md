# Native rotor loading fixture

`rotor_chordwise_first_step.csv` contains the header and first 132 rows of the
native blade-0 chordwise loading history. The one accepted step contains
22 span stations with six chord panels each. No values, names or ordering
were changed. Tests check complete station/panel keys and reject missing or
duplicate panel rows; this is schema validation, not rotor-flow qualification.

Source: `tutorials/vpm/06_rotor_flow_PENDING/samples/rotor/vlm_chordwise_blade_0.csv`.
The source is stored through Git LFS; the local object was checked against its
SHA256 before extracting these rows.

- Source object SHA256: `4b045eaa852bbd4051437d29963d7f35905f3f38c00123e970ac02c43d9e861c`.
- Extract SHA256: `02992743658a1e586b0982a07b3e58458052f5276468a978b0a9515014fd35bc`.
- Recorded step: 1; time: 0.006 s.
