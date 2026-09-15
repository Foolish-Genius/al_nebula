* AutoAnalog-RL sized CTLE
* model source: ihp_ngspice
* W_in = 1.15787e-05
* R_load = 997.529
* I_bias = 0.000741519
* R_s = 336.599
* C_s = 2.71938e-13
* Parameterized differential gm-C CTLE.
* Analysis directives are intentionally injected by SpiceEvaluator.

.param VDD=1.2

VDD vdd 0 1.2
Vinp txP 0 DC 0.6 AC 1
Vinn txN 0 DC 0.6 AC -1

* Channel between the transmitter and the CTLE input, injected by SpiceEvaluator:
* a direct connection for the DC/AC gates, a lossy RC ladder for the transient gate.
RchP txP inP 1m
RchN txN inN 1m

* Differential NMOS input pair with source degeneration.
X1 outP inP sourceP 0 sg13_lv_nmos W=1.15786840347e-05 L=0.13u
X2 outN inN sourceN 0 sg13_lv_nmos W=1.15786840347e-05 L=0.13u
RdegP sourceP tail 336.599451231
RdegN sourceN tail 336.599451231
Cs sourceP sourceN 2.71937972238e-13

* Passive loads preserve bandwidth in the initial architecture.
RloadP vdd outP 997.528891026
RloadN vdd outN 997.528891026

* A common tail current establishes the differential-pair operating point.
Ibias tail 0 DC 0.000741518910019

* Generic ngspice fallback model; VTO and KP are shifted per process corner by SpiceEvaluator.
* The IHP PSP deck is selected by a separate backend.

.lib C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\cornerMOSlv.lib mos_tt
.include C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\sg13g2_moslv_mod.lib
.end
