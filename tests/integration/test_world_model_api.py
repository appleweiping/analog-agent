"""Integration tests for world-model API routes."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class WorldModelApiTests(unittest.TestCase):
    def test_compile_and_build_state_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        client = TestClient(app)
        design_task = {
            "task_id": "api-task-ota",
            "parent_spec_id": "spec-api-ota",
            "task_type": "sizing",
            "circuit_family": "two_stage_ota",
            "topology": {
                "topology_mode": "fixed",
                "template_id": "ota2_miller_basic_v1",
                "template_version": "1.0",
                "ports": [
                    {"name": "vinp", "role": "positive_input", "direction": "input"},
                    {"name": "vinn", "role": "negative_input", "direction": "input"},
                    {"name": "vout", "role": "single_output", "direction": "output"},
                ],
                "instances_schema": [
                    {"name": "m_in_pair", "role": "input_pair", "device_type": "nmos_pair", "tunable_parameters": ["w_in", "l_in"]},
                    {"name": "m_tail", "role": "tail_device", "device_type": "nmos", "tunable_parameters": ["w_tail", "l_tail", "ibias"]},
                    {"name": "c_comp", "role": "compensation", "device_type": "capacitor", "tunable_parameters": ["cc"]},
                ],
                "connectivity_schema": [],
                "topology_constraints": [{"name": "differential_input_required", "mandatory": True, "description": "keep OTA semantics"}],
            },
            "design_space": {
                "variables": [
                    {"name": "w_in", "role": "input_pair_width", "kind": "continuous", "dtype": "float", "domain": {"lower": 2e-6, "upper": 4e-4}, "scale": "log", "units": "m", "default": 8e-6, "source": "process_rule", "is_required": True, "coupling_group": "input_pair"},
                    {"name": "l_in", "role": "input_pair_length", "kind": "continuous", "dtype": "float", "domain": {"lower": 6.5e-8, "upper": 1.3e-6}, "scale": "log", "units": "m", "default": 1e-6, "source": "process_rule", "is_required": True, "coupling_group": "input_pair"},
                    {"name": "w_tail", "role": "tail_device_width", "kind": "continuous", "dtype": "float", "domain": {"lower": 2e-6, "upper": 3.2e-4}, "scale": "log", "units": "m", "default": 6e-6, "source": "process_rule", "is_required": True, "coupling_group": "tail"},
                    {"name": "l_tail", "role": "tail_device_length", "kind": "continuous", "dtype": "float", "domain": {"lower": 6.5e-8, "upper": 1.3e-6}, "scale": "log", "units": "m", "default": 1e-6, "source": "process_rule", "is_required": True, "coupling_group": "tail"},
                    {"name": "ibias", "role": "bias_current", "kind": "continuous", "dtype": "float", "domain": {"lower": 1e-6, "upper": 2e-3}, "scale": "log", "units": "A", "default": 5e-5, "source": "process_rule", "is_required": True, "coupling_group": "bias"},
                    {"name": "cc", "role": "compensation", "kind": "continuous", "dtype": "float", "domain": {"lower": 1e-13, "upper": 2e-11}, "scale": "log", "units": "F", "default": 1e-12, "source": "process_rule", "is_required": True, "coupling_group": "compensation"},
                ],
                "global_constraints": [],
                "derived_variables": [],
                "frozen_variables": [],
                "conditional_variables": [],
                "normalization_policy": {"continuous_strategy": "mixed", "categorical_strategy": "one_hot", "clip_to_domain": True},
            },
            "objective": {
                "objective_mode": "multi_objective",
                "terms": [
                    {"metric": "gbw_hz", "direction": "maximize", "weight": 1.0, "transform": "identity", "normalization": "none"},
                    {"metric": "power_w", "direction": "minimize", "weight": 1.0, "transform": "identity", "normalization": "none"},
                ],
                "scalarization": "weighted_sum",
                "reference_point": {},
                "priority_policy": "balanced",
                "reporting_metrics": ["gbw_hz", "power_w", "phase_margin_deg"],
            },
            "constraints": {
                "hard_constraints": [
                    {"name": "gbw_hz_min", "metric": "gbw_hz", "relation": ">=", "threshold": 1e8, "tolerance": 0.0, "evaluation_stage": "ac", "penalty_policy": "hard_fail", "source": "user", "criticality": "core"},
                    {"name": "phase_margin_deg_min", "metric": "phase_margin_deg", "relation": ">=", "threshold": 60.0, "tolerance": 0.0, "evaluation_stage": "ac", "penalty_policy": "hard_fail", "source": "user", "criticality": "core"},
                ],
                "soft_constraints": [],
                "feasibility_rules": [],
                "operating_region_rules": [],
                "constraint_groups": [{"name": "stability", "members": ["phase_margin_deg_min"]}, {"name": "bandwidth", "members": ["gbw_hz_min"]}],
            },
            "evaluation_plan": {
                "analyses": [
                    {"analysis_type": "op", "order": 0, "config": {"parameters": {}}, "required_metrics": ["power_w"], "estimated_cost": "cheap"},
                    {"analysis_type": "ac", "order": 1, "config": {"parameters": {}}, "required_metrics": ["gbw_hz", "phase_margin_deg"], "estimated_cost": "moderate"},
                ],
                "metric_extractors": [
                    {"metric": "power_w", "from_analysis": "op", "method": "direct"},
                    {"metric": "gbw_hz", "from_analysis": "ac", "method": "direct"},
                    {"metric": "phase_margin_deg", "from_analysis": "ac", "method": "direct"},
                ],
                "corners_policy": {"mode": "fixed", "values": ["tt"]},
                "temperature_policy": {"mode": "fixed", "values": [27.0]},
                "load_policy": {"mode": "fixed", "values": [2e-12]},
                "simulation_budget_class": "moderate",
                "fidelity_policy": "single_fidelity",
                "stop_conditions": [],
            },
            "initial_state": {
                "init_strategy": "template_default",
                "seed_candidates": [{"seed_id": "seed0", "values": {}, "source": "template"}],
                "template_defaults": {},
                "warm_start_source": None,
                "randomization_policy": {"enabled": True, "strategy": "lhs", "amplitude": 0.05},
                "reproducibility": {"seed": 17, "tag": "api"},
            },
            "task_graph": {
                "nodes": [{"node_id": "run", "operation": "simulate", "consumes": [], "produces": ["metrics"]}],
                "edges": [],
                "entrypoints": ["run"],
                "success_criteria": [],
                "failure_routes": [],
            },
            "difficulty_profile": {
                "variable_dimension": 6,
                "discrete_degree": 0.0,
                "constraint_tightness": "medium",
                "evaluation_cost": "moderate",
                "expected_feasibility": "medium",
                "sensitivity_hint": [],
                "risk_flags": [],
            },
            "solver_hint": {
                "recommended_solver_family": "bayesopt",
                "recommended_search_stage": "direct_local_refinement",
                "surrogate_friendly": True,
                "needs_feasibility_first": False,
                "parallelism_hint": "batch",
                "budget_hint": "medium",
            },
            "metadata": {
                "created_by_layer": "task_formalization_layer",
                "compile_timestamp": "2026-04-06T00:00:00+00:00",
                "schema_version": "task-schema-v1",
                "source_spec_signature": "sig",
                "assumptions": [],
                "provenance": [],
            },
            "validation_status": {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "unresolved_dependencies": [],
                "repair_history": [],
                "completeness_score": 1.0,
            },
        }

        compile_response = client.post("/world-model/compile", json=design_task)
        self.assertEqual(compile_response.status_code, 200)
        self.assertIn("world_model_bundle", compile_response.json())

        state_response = client.post(
            "/world-model/build-state",
            json={"design_task": design_task, "parameter_values": {"ibias": 7e-5}},
        )
        self.assertEqual(state_response.status_code, 200)
        state_payload = state_response.json()
        self.assertIn("state_id", state_payload)
        self.assertIn("parameter_state", state_payload)

    def test_predict_metrics_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task
        from libs.world_model.state_builder import build_world_state

        spec = DesignSpec(
            task_id="api-predict-ota",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={"gbw_hz": MetricRange(min=1e8), "phase_margin_deg": MetricRange(min=60.0)},
            environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.92,
        )
        task = compile_design_task(spec).design_task
        assert task is not None
        state = build_world_state(task)

        client = TestClient(app)
        response = client.post(
            "/world-model/predict-metrics",
            json={"design_task": task.model_dump(), "world_state": state.model_dump()},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["metrics"])
        self.assertIn("trust_assessment", payload)
