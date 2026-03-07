"""
Culture Persistence — Save and Load Learned Knowledge

The cumulative culture mechanism from Vibhor's framework:
  "Knowledge compounds rather than resetting."

During a training run, the solver discovers composed programs that solve
tasks. These compositions become reusable concepts (e.g., "erase_3 → crop_nonzero"
might become "learned_abc123"). This module serializes those learned
concepts so they can be loaded in a subsequent evaluation run.

What gets saved:
  - Learned concept names and their step sequences (e.g., ["erase_3", "crop_nonzero"])
  - Successful program compositions with their fitness scores
  - Task features for cross-task transfer

What doesn't need saving:
  - Base primitives (rebuilt from source code every time)
  - Predicates (not serializable, rebuilt each run)
  - Implementation closures (reconstructed from step names on load)

Usage:
    # After training run:
    save_culture(solver.toolkit, solver.archive, "culture.json")

    # Before evaluation run:
    load_culture(solver.toolkit, "culture.json")
"""
from __future__ import annotations

import json
from typing import Optional
from .concepts import Concept, Program, Toolkit, Archive, Grid


def _extract_step_names(concept: Concept) -> list[str]:
    """Extract the primitive step names from a composed concept.

    Learned concepts are compositions of base primitives. Their name
    encodes the chain: "learned_abc123" but the children or the name
    pattern "a → b → c" tells us the steps.
    """
    if concept.children:
        # Recursively flatten children
        names = []
        for child in concept.children:
            child_names = _extract_step_names(child)
            names.extend(child_names)
        return names
    else:
        return [concept.name]


def save_culture(
    toolkit: Toolkit,
    archive: Archive,
    path: str,
) -> dict:
    """Save learned culture (concepts, programs, features) to a JSON file.

    Only saves concepts that were LEARNED during the run (not base primitives).
    Base primitives are identified by having kind != "composed" and not starting
    with "learned_".

    Args:
        toolkit: The toolkit containing learned concepts
        archive: The archive with task solutions and features
        path: File path to save the JSON culture file

    Returns:
        The culture dict that was saved (for inspection/testing).
    """
    culture: dict = {
        "version": "0.9",
        "learned_concepts": [],
        "successful_programs": [],
        "task_features": {},
    }

    # Save learned concepts (composed concepts added during the run)
    for name, concept in toolkit.concepts.items():
        if name.startswith("learned_"):
            step_names = _extract_step_names(concept)
            culture["learned_concepts"].append({
                "name": name,
                "steps": step_names,
                "kind": concept.kind,
                "usage_count": concept.usage_count,
                "success_count": concept.success_count,
            })

    # Save successful programs from the archive
    for task_id, programs in archive.task_solutions.items():
        for prog in programs:
            step_names = [s.name for s in prog.steps]
            culture["successful_programs"].append({
                "task_id": task_id,
                "steps": step_names,
                "fitness": prog.fitness,
                "name": prog.name,
            })

    # Save task features for cross-task transfer
    for task_id, features in archive.task_features.items():
        # Only save serializable features
        serializable = {}
        for k, v in features.items():
            if isinstance(v, (bool, int, float, str)):
                serializable[k] = v
        culture["task_features"][task_id] = serializable

    with open(path, "w") as f:
        json.dump(culture, f, indent=2)

    return culture


def _rebuild_concept(
    step_names: list[str],
    toolkit: Toolkit,
    concept_name: str,
) -> Optional[Concept]:
    """Rebuild a composed concept from its step names.

    Looks up each step in the toolkit and chains them into a
    composed concept with a proper implementation closure.

    Returns None if any step is missing from the toolkit.
    """
    steps = []
    for sname in step_names:
        if sname in toolkit.concepts:
            steps.append(toolkit.concepts[sname])
        else:
            return None  # Can't rebuild — missing primitive

    if not steps:
        return None

    if len(steps) == 1:
        # Single-step "learned" concept — just alias the existing one
        return Concept(
            kind="composed",
            name=concept_name,
            implementation=steps[0].implementation,
            children=steps,
        )

    # Multi-step: chain implementations
    def _make_chain(step_list: list[Concept]):
        """Create a closure that chains all steps."""
        def chain_impl(grid: Grid) -> Grid:
            current = grid
            for step in step_list:
                result = step.apply(current)
                if result is None:
                    return None
                current = result
            return current
        return chain_impl

    return Concept(
        kind="composed",
        name=concept_name,
        implementation=_make_chain(steps),
        children=steps,
    )


def load_culture(
    toolkit: Toolkit,
    path: str,
    archive: Optional[Archive] = None,
) -> dict:
    """Load learned culture from a JSON file into the toolkit.

    Reconstructs learned concepts by looking up their step names in
    the current toolkit and rebuilding the composition chain.

    Args:
        toolkit: The toolkit to add learned concepts to
        path: Path to the culture JSON file
        archive: Optional archive to populate with task features and programs

    Returns:
        Dict with load statistics: concepts_loaded, programs_loaded, skipped.
    """
    with open(path) as f:
        culture = json.load(f)

    stats = {
        "concepts_loaded": 0,
        "concepts_skipped": 0,
        "programs_loaded": 0,
        "programs_skipped": 0,
        "features_loaded": 0,
    }

    # Rebuild and add learned concepts
    for entry in culture.get("learned_concepts", []):
        name = entry["name"]
        step_names = entry["steps"]

        # Skip if already in toolkit
        if name in toolkit.concepts:
            stats["concepts_skipped"] += 1
            continue

        concept = _rebuild_concept(step_names, toolkit, name)
        if concept is not None:
            concept.usage_count = entry.get("usage_count", 0)
            concept.success_count = entry.get("success_count", 0)
            toolkit.add_concept(concept)
            stats["concepts_loaded"] += 1
        else:
            stats["concepts_skipped"] += 1

    # Rebuild successful programs and add to toolkit's program list
    for entry in culture.get("successful_programs", []):
        step_names = entry["steps"]
        steps = []
        missing = False
        for sname in step_names:
            if sname in toolkit.concepts:
                steps.append(toolkit.concepts[sname])
            else:
                missing = True
                break

        if missing or not steps:
            stats["programs_skipped"] += 1
            continue

        prog = Program(steps)
        prog.fitness = entry.get("fitness", 0.0)
        toolkit.add_program(prog)
        stats["programs_loaded"] += 1

    # Load task features into archive (for cross-task transfer)
    if archive is not None:
        for task_id, features in culture.get("task_features", {}).items():
            archive.record_features(task_id, features)
            stats["features_loaded"] += 1

    return stats
