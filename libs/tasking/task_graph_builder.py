"""Task-graph construction for the task formalization layer."""

from __future__ import annotations


from libs.schema.design_task import CriterionSpec, FailureRoute, GraphEdge, GraphNode, TaskGraph


def build_task_graph(needs_feasibility_first: bool, surrogate_friendly: bool) -> TaskGraph:
    """Return the canonical execution graph for a compiled DesignTask."""

    nodes = [
        GraphNode(node_id="initialize_task", operation="initialize_task_state", produces=["initial_state"]),
        GraphNode(node_id="propose_candidate", operation="propose_candidate", consumes=["design_space"], produces=["candidate"]),
        GraphNode(node_id="precheck_candidate", operation="precheck_candidate", consumes=["candidate"], produces=["candidate_precheck"]),
        GraphNode(
            node_id="screen_with_world_model",
            operation="screen_with_world_model" if surrogate_friendly else "bypass_world_model_screen",
            consumes=["candidate_precheck"],
            produces=["screened_candidate"],
        ),
        GraphNode(
            node_id="run_evaluation_plan",
            operation="run_evaluation_plan",
            consumes=["screened_candidate", "evaluation_plan"],
            produces=["raw_metrics"],
        ),
        GraphNode(
            node_id="extract_metrics",
            operation="extract_metrics",
            consumes=["raw_metrics"],
            produces=["metric_vector"],
        ),
        GraphNode(
            node_id="check_constraints",
            operation="check_constraints",
            consumes=["metric_vector", "constraints"],
            produces=["feasibility_status"],
        ),
        GraphNode(
            node_id="update_search_state",
            operation="update_search_state",
            consumes=["candidate", "metric_vector", "feasibility_status"],
            produces=["search_state"],
        ),
        GraphNode(node_id="terminate", operation="terminate_task", consumes=["search_state"], produces=["terminal_report"]),
    ]
    edges = [
        GraphEdge(source="initialize_task", target="propose_candidate"),
        GraphEdge(source="propose_candidate", target="precheck_candidate"),
        GraphEdge(source="precheck_candidate", target="screen_with_world_model"),
        GraphEdge(source="screen_with_world_model", target="run_evaluation_plan"),
        GraphEdge(source="run_evaluation_plan", target="extract_metrics"),
        GraphEdge(source="extract_metrics", target="check_constraints"),
        GraphEdge(source="check_constraints", target="update_search_state", condition="constraints_checked"),
        GraphEdge(source="check_constraints", target="terminate", condition="all_success_criteria_met"),
        GraphEdge(source="update_search_state", target="propose_candidate", condition="continue_search"),
        GraphEdge(source="update_search_state", target="terminate", condition="budget_exhausted"),
    ]
    success_criteria = [
        CriterionSpec(name="all_hard_constraints_satisfied", relation="event"),
    ]
    if needs_feasibility_first:
        success_criteria.append(CriterionSpec(name="feasible_region_established", relation="event"))

    failure_routes = [
        FailureRoute(trigger="candidate_infeasible", route_to="propose_candidate", reason="request a new candidate proposal"),
        FailureRoute(trigger="budget_exhausted", route_to="terminate", reason="terminate after budget exhaustion"),
    ]
    if surrogate_friendly:
        failure_routes.append(
            FailureRoute(
                trigger="world_model_confidence_low",
                route_to="run_evaluation_plan",
                reason="escalate directly to simulator-grounded evaluation",
            )
        )

    return TaskGraph(
        nodes=nodes,
        edges=edges,
        entrypoints=["initialize_task"],
        success_criteria=success_criteria,
        failure_routes=failure_routes,
    )
