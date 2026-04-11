* two_stage_ota demonstrator truth template
* truth_mode=${truth_mode}
* family=two_stage_ota
* template_id=${template_id}
* p2_hint_hz=${p2_hint_hz}

.param vdd=${vdd}
.param vin_cm=${vin_cm}
.param vin_step_high=${vin_step_high}
.param ibias=${ibias}
.param cc=${cc}
.param cload=${cload}
.param gm1=${gm1}
.param gm2=${gm2}
.param ro1=${ro1}
.param ro2=${ro2}
.param cp1=${cp1}

VDD vdd 0 {vdd}
VCM vcm 0 {vin_cm}
VINP vinp 0 DC {vin_cm} AC 1 PULSE({vin_cm} {vin_step_high} 20n 1n 1n 80n 200n)
VINN vinn 0 DC {vin_cm}

* First stage proxy: differential transconductor into a high-gain node.
Gm1 n1 vcm vinp vinn {gm1}
R1 n1 vcm {ro1}
Cp1 n1 vcm {cp1}

* Second stage proxy: inverting gain stage with Miller compensation and output load.
Gm2 vcm vout n1 vcm {gm2}
R2 vout vcm {ro2}
Cc n1 vout {cc}
Cload vout 0 {cload}

* Supply current model used for truth-level power extraction.
Ibias vdd 0 DC {2*ibias}

.save all
