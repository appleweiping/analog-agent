* bandgap demonstrator truth template
* slice_version=bandgap_v1
* truth_mode=${truth_mode}
* family=bandgap
* template_id=${template_id}
* tempco_hint_ppm_per_c=${tempco_hint_ppm_per_c}

.param vdd=${vdd}
.param vdd_step_high=${vdd_step_high}
.param area_ratio=${area_ratio}
.param r1=${r1}
.param r2=${r2}
.param gm_core=${gm_core}
.param ro_core=${ro_core}
.param c_ref=${c_ref}
.param iref=${iref}

VDD vdd 0 PULSE({vdd} {vdd_step_high} 20u 200n 200n 20u 60u)

* Reference-core proxy: current draw plus PTAT/CTAT shaping through a soft loop.
IBIAS vdd nbias DC {iref}
RPTAT nbias nsum {r1}
RCTAT nsum 0 {r2}
GCORE vdd vref nbias 0 {gm_core}
RCORE vref 0 {ro_core}
CREF vref 0 {c_ref}
ECORE nsum 0 VALUE = { log(area_ratio + 1) * 0.052 + V(vref) * 0.18 }
RLEAK vref 0 4e6

.save all
