# AutoAnalog-RL research notes

This folder contains the supplied CTLE paper and implementation-oriented references.
The papers inform design choices; they are not substitutes for corner simulation.

## Supplied reference

- `ctle_researchpaper.pdf`: Challayya Naidu and Lu, "Receiver Analog Front-End Cascading Transimpedance Amplifier and Continuous-Time Linear Equalizer for Signals of 5 to 30 Gb/s," *Electronics*, 2022, 11, 1546.
- DOI: https://doi.org/10.3390/electronics11101546
- Open-access article: https://www.mdpi.com/2079-9292/11/10/1546

Useful takeaways applied to this project:

- Source degeneration with tunable `R_s` and `C_s` controls CTLE peaking and bandwidth.
- Tail-current allocation is a key speed/power tradeoff and should remain an optimization action.
- PRBS eye validation is necessary; a single sine-wave AC result is insufficient.
- Equalization must be evaluated with channel/receiver interaction, not only isolated gain.

## Standards and measurement references

- PCI-SIG, PCI Express Base Specification Revision 2.0. Use the licensed specification for the authoritative Gen 2 electrical limits and compliance masks.
- IEEE 802.3, Clause 52 and related receiver-equalization material. Use the licensed standard for formal receiver stress and eye measurements.

## Reproducibility policy

Only references that are legally redistributable should be copied into this folder.
For standards, keep links and citations rather than redistributing copyrighted PDFs.
