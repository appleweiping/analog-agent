* ldo demonstrator truth template
* slice_version=ldo_v1
* truth_mode=${truth_mode}
* family=ldo
* template_id=${template_id}
* p2_hint_hz=${p2_hint_hz}

.param vdd=${vdd}
.param vref_dc=${vref_dc}
.param vref_step_high=${vref_step_high}
.param gm_err=${gm_err}
.param ro_err=${ro_err}
.param gm_pass=${gm_pass}
.param c_comp=${c_comp}
.param cload=${cload}
.param rload=${rload}
.param rfb_top=${rfb_top}
.param rfb_bot=${rfb_bot}

VDD vdd 0 {vdd}
VREF vref 0 DC {vref_dc} AC 1 PULSE({vref_dc} {vref_step_high} 20u 200n 200n 20u 60u)

RFBTOP vout fb {rfb_top}
RFBBOT fb 0 {rfb_bot}

* Error amplifier proxy and compensation pole.
GERR 0 gate vref fb {gm_err}
RERR gate 0 {ro_err}
CCOMP gate 0 {c_comp}

* Pass-device proxy driving the regulated output.
GPASS vdd vout gate 0 {gm_pass}
RLOAD vout 0 {rload}
CLOAD vout 0 {cload}

* Quiescent current model used for power extraction.
IBIAS vdd 0 DC ${quiescent_current}

.save all
