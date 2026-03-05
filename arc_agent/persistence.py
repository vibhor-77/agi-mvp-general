"""
Persistent Toolkit & Archive Serialization

Solves the "Reset Button Problem" from Vibhor's framework:
the Toolkit must survive across runs so that knowledge compounds
across sessions, not just within a single batch.

Design constraints:
- Composed concepts contain closures (lambdas), which aren't directly
  serializable. We solve this by recording the composition recipe
  (list of child concept names) and re-composing on load.
- Primitive concepts are restored by name from `build_initial_toolkit()`.
- All usage statistics (usage_count, success_count) are preserved.
"""
from __future__ import annotations
import json
from typing import Optional
from .concepts import Concept, Toolkit, Archive, Program, Grid
from .primitives import build_initial_toolkit


def save_toolkit(toolkit: Toolkit, path: str) -> None:
    """Serialize a Toolkit to a JSON file.

    Stores each concept's metadata and composition recipe.
    Primitive concepts are recorded by name; composed concepts
    record their child names so they can be re-composed on load.

    Args:
        toolkit: The Toolkit instance to save.
        path: File path to write the JSON to.
    """
    data = {
        "version": 1,
        "concepts": {},
        "programs": [],
    }

    for name, concept in toolkit.concepts.items():
        entry = {
            "kind": concept.kind,
            "name": concept.name,
            "usage_count": concept.usage_count,
            "success_count": concept.success_count,
        }
        if concept.kind == "composed" and concept.children:
            entry["children"] = [child.name for child in concept.children]
        data["concepts"][name] = entry

    # Save successful programs as name sequences
    for program in toolkit.programs:
        data["programs"].append({
            "name": program.name,
            "steps": [step.name for step in program.steps],
            "fitness": program.fitness,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_toolkit(path: str) -> Toolkit:
    """Deserialize a Toolkit from a JSON file.

    Restores primitives from `build_initial_toolkit()`, then
    re-composes any learned concepts by chaining their children.

    Args:
        path: File path to read the JSON from.

    Returns:
        A fully functional Toolkit with all concepts restored.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Start with a fresh primitive toolkit
    toolkit = build_initial_toolkit()

    # Restore usage stats for primitives
    for name, entry in data["concepts"].items():
        if name in toolkit.concepts:
            toolkit.concepts[name].usage_count = entry.get("usage_count", 0)
            toolkit.concepts[name].success_count = entry.get("success_count", 0)

    # Restore composed concepts (may depend on other composed concepts,
    # so we iterate until all are resolved)
    composed_entries = {
        name: entry for name, entry in data["concepts"].items()
        if entry["kind"] == "composed" and name not in toolkit.concepts
    }

    # Topological resolution: keep iterating until no more can be resolved
    max_iterations = len(composed_entries) + 1
    for _ in range(max_iterations):
        resolved_any = False
        for name, entry in list(composed_entries.items()):
            children_names = entry.get("children", [])
            # Check if all children are already in toolkit
            if all(cn in toolkit.concepts for cn in children_names):
                children = [toolkit.concepts[cn] for cn in children_names]
                # Build composed implementation
                impl = _build_composed_impl(children)
                concept = Concept(
                    kind="composed",
                    name=name,
                    implementation=impl,
                    children=children,
                    usage_count=entry.get("usage_count", 0),
                    success_count=entry.get("success_count", 0),
                )
                toolkit.add_concept(concept)
                del composed_entries[name]
                resolved_any = True

        if not composed_entries or not resolved_any:
            break

    # Restore programs
    for prog_data in data.get("programs", []):
        step_names = prog_data.get("steps", [])
        steps = [
            toolkit.concepts[sn] for sn in step_names
            if sn in toolkit.concepts
        ]
        if steps:
            program = Program(steps, name=prog_data.get("name", ""))
            program.fitness = prog_data.get("fitness", 0.0)
            toolkit.add_program(program)

    return toolkit


def _build_composed_impl(children: list[Concept]):
    """Build a closure that chains multiple concepts in sequence."""
    # Capture children by value via default argument
    def impl(grid: Grid, _children=children) -> Optional[Grid]:
        current = grid
        for child in _children:
            result = child.apply(current)
            if result is None:
                return None
            current = result
        return current
    return impl


def save_archive(archive: Archive, path: str) -> None:
    """Serialize an Archive to a JSON file.

    Saves task features, history, and solution metadata.
    Programs are stored as name strings (not full objects)
    since the implementations live in the Toolkit.

    Args:
        archive: The Archive instance to save.
        path: File path to write the JSON to.
    """
    data = {
        "version": 1,
        "task_features": archive.task_features,
        "history": archive.history,
        "task_solutions": {
            tid: [{"name": p.name, "fitness": p.fitness}
                  for p in programs]
            for tid, programs in archive.task_solutions.items()
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_archive(path: str) -> Archive:
    """Deserialize an Archive from a JSON file.

    Restores task features and history. Solution programs are
    stored as metadata only (names + fitness) since the actual
    implementations live in the Toolkit.

    Args:
        path: File path to read the JSON from.

    Returns:
        An Archive with features and history restored.
    """
    with open(path, "r") as f:
        data = json.load(f)

    archive = Archive()
    archive.task_features = data.get("task_features", {})
    archive.history = data.get("history", [])

    # Restore solution metadata (programs without implementations)
    for tid, solutions in data.get("task_solutions", {}).items():
        archive.task_solutions[tid] = []
        for sol_data in solutions:
            # Create a stub program with the recorded metadata
            stub = Program([], name=sol_data.get("name", ""))
            stub.fitness = sol_data.get("fitness", 0.0)
            archive.task_solutions[tid].append(stub)

    return archive
