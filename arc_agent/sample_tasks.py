"""
Sample ARC-AGI Tasks for Testing

These are representative ARC-AGI tasks that test different aspects
of the 4 pillars. Since we can't download the full dataset in this
environment, these hand-crafted tasks capture the key challenge types.

Each task has 'train' examples (with input/output pairs) and 'test' examples.
"""

SAMPLE_TASKS = {
    # ================================================================
    # Task 1: Mirror Horizontal
    # Input pattern is mirrored left-to-right
    # Tests: Single primitive (Pillar 1: fast feedback)
    # ================================================================
    "mirror_h": {
        "train": [
            {
                "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "output": [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
            },
            {
                "input": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                "output": [[0, 0, 1], [0, 2, 0], [3, 0, 0]],
            },
            {
                "input": [[5, 5, 0], [5, 0, 0]],
                "output": [[0, 5, 5], [0, 0, 5]],
            },
        ],
        "test": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[2, 1], [4, 3]],
            },
        ],
    },

    # ================================================================
    # Task 2: Rotate 90 CW
    # Input is rotated 90 degrees clockwise
    # Tests: Single primitive
    # ================================================================
    "rotate_90": {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[3, 1], [4, 2]],
            },
            {
                "input": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                "output": [[0, 0, 1], [0, 2, 0], [3, 0, 0]],
            },
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]],
                "output": [[7, 5], [8, 6]],
            },
        ],
    },

    # ================================================================
    # Task 3: Fill Enclosed Regions
    # Zero-regions fully enclosed by colored cells get filled
    # Tests: Spatial reasoning primitive
    # ================================================================
    "fill_enclosed": {
        "train": [
            {
                "input": [
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
                "output": [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
            },
            {
                "input": [
                    [2, 2, 2, 0],
                    [2, 0, 2, 0],
                    [2, 2, 2, 0],
                ],
                "output": [
                    [2, 2, 2, 0],
                    [2, 2, 2, 0],
                    [2, 2, 2, 0],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [3, 3, 3],
                    [3, 0, 3],
                    [3, 3, 3],
                ],
                "output": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
            },
        ],
    },

    # ================================================================
    # Task 4: Scale 2x
    # Each cell becomes a 2x2 block
    # Tests: Scaling primitive
    # ================================================================
    "scale_2x": {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ],
            },
            {
                "input": [[5]],
                "output": [[5, 5], [5, 5]],
            },
        ],
        "test": [
            {
                "input": [[1, 0], [0, 2]],
                "output": [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 2, 2],
                    [0, 0, 2, 2],
                ],
            },
        ],
    },

    # ================================================================
    # Task 5: Gravity Down
    # Non-zero cells fall to the bottom of their column
    # Tests: Physics-like reasoning
    # ================================================================
    "gravity_down": {
        "train": [
            {
                "input": [
                    [1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 2, 3],
                ],
            },
            {
                "input": [
                    [4, 5],
                    [0, 0],
                    [0, 0],
                ],
                "output": [
                    [0, 0],
                    [0, 0],
                    [4, 5],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [0, 1, 0],
                    [2, 0, 0],
                    [0, 0, 3],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [2, 1, 3],
                ],
            },
        ],
    },

    # ================================================================
    # Task 6: Crop + Mirror (Composition)
    # Crop to non-zero bounding box, then mirror horizontally
    # Tests: Pillar 3 — requires composing TWO primitives
    # ================================================================
    "crop_then_mirror": {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 0, 0, 0],
                ],
                "output": [[2, 1], [4, 3]],
            },
            {
                "input": [
                    [0, 0, 0],
                    [0, 5, 6],
                    [0, 0, 0],
                ],
                "output": [[6, 5]],
            },
        ],
        "test": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[9, 8, 7]],
            },
        ],
    },

    # ================================================================
    # Task 7: Outline (keep only borders of colored regions)
    # Tests: Spatial reasoning
    # ================================================================
    "outline_task": {
        "train": [
            {
                "input": [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "output": [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                ],
            },
            {
                "input": [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "output": [
                    [2, 2, 2],
                    [2, 0, 2],
                    [2, 2, 2],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3],
                ],
                "output": [
                    [3, 3, 3, 3, 3],
                    [3, 0, 0, 0, 3],
                    [3, 0, 0, 0, 3],
                    [3, 0, 0, 0, 3],
                    [3, 3, 3, 3, 3],
                ],
            },
        ],
    },

    # ================================================================
    # Task 8: Transpose + Gravity (Multi-step composition)
    # Transpose the grid then apply gravity down
    # Tests: Pillar 3 — multi-step composition
    # ================================================================
    "transpose_gravity": {
        "train": [
            {
                "input": [
                    [1, 0],
                    [0, 0],
                ],
                "output": [
                    [0, 0],
                    [1, 0],
                ],
            },
            {
                "input": [
                    [0, 2, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 2, 0],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [3, 0, 0],
                    [0, 4, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [3, 4, 0],
                ],
            },
        ],
    },

    # ================================================================
    # Task 9: Color Swap (1 → 2)
    # All cells with color 1 become color 2
    # Tests: Pattern recognition in color space
    # ================================================================
    "color_swap_1_to_2": {
        "train": [
            {
                "input": [[1, 0, 1], [0, 1, 0]],
                "output": [[2, 0, 2], [0, 2, 0]],
            },
            {
                "input": [[1, 1], [1, 1]],
                "output": [[2, 2], [2, 2]],
            },
        ],
        "test": [
            {
                "input": [[0, 1, 0, 1, 0]],
                "output": [[0, 2, 0, 2, 0]],
            },
        ],
    },

    # ================================================================
    # Task 10: Invert + Crop (Novel composition)
    # Invert colors (0↔non-0) then crop to non-zero
    # Tests: Pillar 3+4 — novel composition discovery
    # ================================================================
    "invert_then_crop": {
        "train": [
            {
                "input": [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                "output": [[1]],
            },
            {
                "input": [
                    [2, 2, 2, 2],
                    [2, 0, 0, 2],
                    [2, 2, 2, 2],
                ],
                "output": [[1, 1]],
            },
        ],
        "test": [
            {
                "input": [
                    [3, 3, 3],
                    [3, 0, 3],
                    [3, 0, 3],
                    [3, 3, 3],
                ],
                "output": [[1], [1]],
            },
        ],
    },
}
