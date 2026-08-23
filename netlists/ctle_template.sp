* Parameterized differential gm-C CTLE.
* Analysis directives are intentionally injected by SpiceEvaluator.

.param VDD=1.2

VDD vdd 0 {VDD}
Vinp inP 0 {VINP}
Vinn inN 0 {VINN}

* Differential NMOS input pair with source degeneration.
M1 outP inP sourceP 0 ctle_nmos W={W_in} L=0.13u
M2 outN inN sourceN 0 ctle_nmos W={W_in} L=0.13u
RdegP sourceP tail {R_s}
RdegN sourceN tail {R_s}
Cs sourceP sourceN {C_s}

* Passive loads preserve bandwidth in the initial architecture.
RloadP vdd outP {R_load}
RloadN vdd outN {R_load}

* A common tail current establishes the differential-pair operating point.
Ibias tail 0 DC {I_bias}

* Generic ngspice fallback model. The IHP PSP deck is selected by a separate backend.
.model ctle_nmos nmos level=1 vto=0.45 kp=200u lambda=0.04 gamma=0.4 phi=0.7

.end
