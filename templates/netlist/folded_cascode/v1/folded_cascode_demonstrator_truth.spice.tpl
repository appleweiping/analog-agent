* folded_cascode_ota demonstrator truth template
* slice_version=folded_cascode_v1
* truth_mode=${truth_mode}
* family=folded_cascode_ota
* template_id=${template_id}
* p2_hint_hz=${p2_hint_hz}

.param vdd=${vdd}
.param vin_cm=${vin_cm}
.param vin_step_high=${vin_step_high}
.param ibias=${ibias}
.param cc=${cc}
.param cload=${cload}
.param gm_in=${gm_in}
.param gm_fold=${gm_fold}
.param ro_in=${ro_in}
.param ro_fold=${ro_fold}
.param c_fold=${c_fold}

VDD vdd 0 {vdd}
VCM vcm 0 {vin_cm}
VINP vinp 0 DC {vin_cm} AC 1 PULSE({vin_cm} {vin_step_high} 20n 1n 1n 80n 200n)
VINN vinn 0 DC {vin_cm}

* Input pair proxy driving the folded node.
GMIN nfold vcm vinp vinn {gm_in}
RIN nfold vcm {ro_in}
CFOLD nfold vcm {c_fold}

* Folded cascode branch proxy driving the single-ended output node.
GFOLD vcm vout nfold vcm {gm_fold}
ROUT vout vcm {ro_fold}
CCOMP nfold vout {cc}
CLOAD vout 0 {cload}

* Supply current model used for truth-level power extraction.
IBIAS vdd 0 DC {2.4*ibias}

.save all
