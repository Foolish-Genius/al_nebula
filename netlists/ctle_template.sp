* Parameterized differential gm-C CTLE for the IHP sg13g2 process.
* Analysis directives are intentionally injected by SpiceEvaluator.

.param VDD=1.2

VDD vdd 0 {VDD}
Vinp inP 0 DC 0.6 AC 1
Vinn inN 0 DC 0.6 AC -1

* Differential NMOS input pair with source degeneration.
M1 outP inP sourceP 0 sg13_lv_nmos W={W_in} L=0.13u
M2 outN inN sourceN 0 sg13_lv_nmos W={W_in} L=0.13u
Rs sourceP sourceN {R_s}
Cs sourceP sourceN {C_s}

* Passive loads preserve bandwidth in the initial architecture.
RloadP vdd outP {R_load}
RloadN vdd outN {R_load}

* Ideal tail current source keeps the bias parameter explicit.
Ibias tail 0 DC {I_bias}
Mtail tail tail 0 0 sg13_lv_nmos W={W_in} L=0.13u

* Equal tail-current split is represented by ideal current steering.
BsteerP outP sourceP I={I_bias/2}
BsteerN outN sourceN I={I_bias/2}

.end
