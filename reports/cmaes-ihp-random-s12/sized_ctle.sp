* AutoAnalog-RL sized CTLE
* model source: ihp_ngspice
* W_in = 6.74007e-06
* R_load = 991.755
* I_bias = 0.000791708
* R_s = 432.138
* C_s = 2.6114e-13
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
X1 outP inP sourceP 0 sg13_lv_nmos W=6.74006611694e-06 L=0.13u
X2 outN inN sourceN 0 sg13_lv_nmos W=6.74006611694e-06 L=0.13u
RdegP sourceP tail 432.137814226
RdegN sourceN tail 432.137814226
Cs sourceP sourceN 2.61139634102e-13

* Passive loads preserve bandwidth in the initial architecture.
RloadP vdd outP 991.755175253
RloadN vdd outN 991.755175253

* A common tail current establishes the differential-pair operating point.
Ibias tail 0 DC 0.000791708309344

* Generic ngspice fallback model; VTO and KP are shifted per process corner by SpiceEvaluator.
* The IHP PSP deck is selected by a separate backend.

.lib C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\cornerMOSlv.lib mos_tt
.include C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\sg13g2_moslv_mod.lib
.end
