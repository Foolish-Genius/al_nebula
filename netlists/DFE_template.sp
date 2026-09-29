* Parameterized 1-Tap DFE Summing Amplifier.
* Analysis directives are intentionally injected by SpiceEvaluator.

.param VDD=1.2

VDD vdd 0 {VDD}

* Main signal inputs (e.g., from CTLE output)
Vinp inP 0 {VINP}
Vinn inN 0 {VINN}

* Feedback signal inputs (from slicer output representing previous bit decision)
Vfbp fbP 0 {VFBP}
Vfbn fbN 0 {VFBN}

* Main differential NMOS input pair
MmainP outP inP tailMain 0 dfe_nmos W={W_main} L=0.13u
MmainN outN inN tailMain 0 dfe_nmos W={W_main} L=0.13u

* 1-Tap DFE feedback differential NMOS pair (Current steering DAC)
* Gate inputs are crossed to subtract the previous bit's post-cursor ISI
MdfeP outP fbN tailDfe 0 dfe_nmos W={W_tap1} L=0.13u
MdfeN outN fbP tailDfe 0 dfe_nmos W={W_tap1} L=0.13u

* Shared passive loads for current summation
RloadP vdd outP {R_load}
RloadN vdd outN {R_load}

* Tail currents establish the main gain and the DFE tap weight
IbiasMain tailMain 0 DC {I_main}
IbiasDfe tailDfe 0 DC {I_tap1}

* Generic ngspice fallback model. The IHP 130nm or Skywater 130nm PSP deck is selected by backend.
.model dfe_nmos nmos level=1 vto=0.45 kp=200u lambda=0.04 gamma=0.4 phi=0.7

.end