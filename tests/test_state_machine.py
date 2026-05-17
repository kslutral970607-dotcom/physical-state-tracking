import unittest

from physical_state_tracking.state_machine import Operation, StateMachine


class StateMachineTest(unittest.TestCase):
    def test_applies_additive_operations(self):
        machine = StateMachine({"A": 2, "B": 5})

        final_state = machine.apply(
            [
                Operation("A", 3, "A increases by 3."),
                Operation("B", -2, "B decreases by 2."),
                Operation("A", -1, "A decreases by 1."),
            ]
        )

        self.assertEqual(final_state, {"A": 4, "B": 3})
        self.assertEqual(machine.initial_state, {"A": 2, "B": 5})

    def test_rejects_unknown_variable(self):
        machine = StateMachine({"A": 1})

        with self.assertRaises(KeyError):
            machine.apply([Operation("B", 1, "B increases by 1.")])


if __name__ == "__main__":
    unittest.main()
