# AutoAnalog-RL research notes

This folder contains the supplied CTLE paper and implementation-oriented references.

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

## Additional supplied references

- `1-s2.0-S0026269221000689-main.pdf`: charge-steering receiver/equalizer work. It reinforces that CTLE and DFE have complementary roles: CTLE addresses pre-cursor loss, while DFE removes post-cursor ISI without directly amplifying it. It also motivates power-efficiency and BER/eye validation instead of gain-only optimization.
- `A_56_Gb_s_receiver_front_end_with_a_CTLE.pdf`: high-speed CTLE plus speculative 1-tap DFE architecture. The useful system lesson is that the DFE needs an explicit slicer, one-UI decision history, tap coefficient, and timing/decision validation; the 20 nm power and clocking numbers are not targets for this 130 nm PCIe Gen 2 design.
- `electronics-08-01233.pdf`: additional receiver/equalizer reference supplied with the project. Its relevant guidance is to keep CTLE tuning and decision feedback as separate measurable stages and to evaluate equalization with waveform/eye behavior, not only small-signal peaking.

The circuit diagrams in `circuits/` are treated as topology references. The active CTLE netlist implements the source-degenerated differential pair; the active DFE measurement implements the reference equation $y[n] = x[n] - b_1\hat{d}[n-1]$. The analog DFE SPICE template remains a separate topology artifact until a clocked slicer and feedback delay are modeled in ngspice.
