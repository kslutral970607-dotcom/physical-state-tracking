import unittest

from physical_state_tracking.state_machine import BounceStep, StateMachine


class StateMachineTest(unittest.TestCase):
    def test_moves_without_bounce(self):
        machine = StateMachine({"z": 4, "d": 1, "k": 0, "w": "blue"})

        final_state = machine.apply([BounceStep(3)])

        self.assertEqual(final_state, {"z": 7, "d": 1, "k": 0, "w": "blue"})
        self.assertEqual(machine.initial_state, {"z": 4, "d": 1, "k": 0, "w": "blue"})

    def test_diagnostic_no_boundary_transition_preserves_d_k_w(self):
        machine = StateMachine({"z": 5, "d": 1, "k": 0, "w": "blue"})

        final_state = machine.apply([BounceStep(3)])

        self.assertEqual(final_state, {"z": 8, "d": 1, "k": 0, "w": "blue"})

    def test_diagnostic_upper_boundary_reflection(self):
        machine = StateMachine({"z": 9, "d": 1, "k": 0, "w": "blue"})

        final_state = machine.apply([BounceStep(3)])

        self.assertEqual(final_state, {"z": 8, "d": -1, "k": 1, "w": "blue"})

    def test_diagnostic_lower_boundary_reflection(self):
        machine = StateMachine({"z": 1, "d": -1, "k": 0, "w": "green"})

        final_state = machine.apply([BounceStep(3)])

        self.assertEqual(final_state, {"z": 2, "d": 1, "k": 1, "w": "green"})

    def test_reflects_at_upper_boundary(self):
        machine = StateMachine({"z": 9, "d": 1, "k": 0, "w": "red"})

        final_state = machine.apply([BounceStep(3)])

        self.assertEqual(final_state, {"z": 8, "d": -1, "k": 1, "w": "red"})

    def test_reflects_at_lower_boundary(self):
        machine = StateMachine({"z": 1, "d": -1, "k": 2, "w": "green"})

        final_state = machine.apply([BounceStep(4)])

        self.assertEqual(final_state, {"z": 3, "d": 1, "k": 3, "w": "green"})

    def test_records_trajectory(self):
        machine = StateMachine({"z": 8, "d": 1, "k": 0, "w": "yellow"})

        final_state, trajectory = machine.run([BounceStep(2), BounceStep(3)])

        self.assertEqual(final_state, {"z": 7, "d": -1, "k": 1, "w": "yellow"})
        self.assertEqual(
            trajectory,
            [
                {"z": 10, "d": -1, "k": 1, "w": "yellow"},
                {"z": 7, "d": -1, "k": 1, "w": "yellow"},
            ],
        )

    def test_rejects_invalid_state(self):
        with self.assertRaises(ValueError):
            StateMachine({"z": 4, "d": 0, "k": 0, "w": "blue"})


if __name__ == "__main__":
    unittest.main()
