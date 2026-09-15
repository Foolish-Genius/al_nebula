* AutoAnalog-RL sized CTLE
* model source: ihp_ngspice
* W_in = 1.40453e-05
* R_load = 825.899
* I_bias = 0.00104292
* R_s = 295.634
* C_s = 3.42971e-13
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
X1 outP inP sourceP 0 sg13_lv_nmos W=1.40453370599e-05 L=0.13u
X2 outN inN sourceN 0 sg13_lv_nmos W=1.40453370599e-05 L=0.13u
RdegP sourceP tail 295.633565372
RdegN sourceN tail 295.633565372
Cs sourceP sourceN 3.429707964e-13

* Passive loads preserve bandwidth in the initial architecture.
RloadP vdd outP 825.899124064
RloadN vdd outN 825.899124064

* A common tail current establishes the differential-pair operating point.
Ibias tail 0 DC 0.00104292153633

* Generic ngspice fallback model; VTO and KP are shifted per process corner by SpiceEvaluator.
* The IHP PSP deck is selected by a separate backend.

.lib C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\cornerMOSlv.lib mos_tt
.include C:\Users\Hp\tools\ihp-open-pdk\ihp-sg13g2\libs.tech\ngspice\models\sg13g2_moslv_mod.lib
.end
