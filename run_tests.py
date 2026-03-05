#!/usr/bin/env python3
"""
Test runner that works without pytest.
Imports all test classes and runs them using unittest.
"""
import unittest
import sys
import random

# Ensure reproducibility
random.seed(42)

# We need to convert pytest-style tests to unittest-compatible
# by wrapping them in a simple adapter

def run_all_tests():
    """Discover and run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Import all test modules
    from tests import test_concepts, test_primitives, test_scorer
    from tests import test_synthesizer, test_explorer, test_integration

    modules = [
        test_concepts, test_primitives, test_scorer,
        test_synthesizer, test_explorer, test_integration,
    ]

    for mod in modules:
        # Find all classes in the module
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and name.startswith("Test"):
                # Convert pytest-style test class to unittest
                # Create a unittest.TestCase subclass dynamically
                methods = [m for m in dir(obj) if m.startswith("test_")]
                for method_name in methods:
                    method = getattr(obj, method_name)

                    # Check if method expects fixtures
                    import inspect
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    params = [p for p in params if p != 'self']

                    # Create a test case
                    def make_test(cls, mname, fixture_params):
                        def test_fn(self):
                            random.seed(42)
                            instance = cls()
                            fn = getattr(instance, mname)
                            # Resolve fixtures
                            kwargs = {}
                            for param in fixture_params:
                                if param == 'toolkit':
                                    from arc_agent.primitives import build_initial_toolkit
                                    kwargs[param] = build_initial_toolkit()
                                elif param == 'archive':
                                    from arc_agent.concepts import Archive
                                    kwargs[param] = Archive()
                                elif param == 'synth':
                                    from arc_agent.primitives import build_initial_toolkit
                                    from arc_agent.synthesizer import ProgramSynthesizer
                                    tk = build_initial_toolkit()
                                    kwargs[param] = ProgramSynthesizer(tk, population_size=20, max_program_length=3)
                                elif param == 'explorer':
                                    from arc_agent.primitives import build_initial_toolkit
                                    from arc_agent.explorer import ExplorationEngine
                                    from arc_agent.concepts import Archive
                                    tk = build_initial_toolkit()
                                    ar = Archive()
                                    kwargs[param] = ExplorationEngine(tk, ar, epsilon=0.3)
                                elif param == 'solver':
                                    from arc_agent.solver import FourPillarsSolver
                                    kwargs[param] = FourPillarsSolver(
                                        population_size=40, max_generations=20,
                                        max_program_length=4, verbose=False
                                    )
                            fn(**kwargs)
                        test_fn.__name__ = f"test_{cls.__name__}_{mname}"
                        return test_fn

                    # Create a TestCase class for this test
                    test_case_name = f"{name}_{method_name}"
                    test_method = make_test(obj, method_name, params)

                    tc = type(test_case_name, (unittest.TestCase,), {
                        method_name: test_method,
                    })
                    suite.addTest(tc(method_name))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
