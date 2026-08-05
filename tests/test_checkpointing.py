import unittest

from utils.checkpointing import filter_compatible_state_dict


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


class CompatibleCheckpointStateTest(unittest.TestCase):
    def test_filters_unknown_and_shape_mismatched_weights(self):
        source = {
            'trunk.weight': FakeTensor(4, 4),
            'decoder.weight': FakeTensor(10, 4),
            'legacy.weight': FakeTensor(1),
        }
        target = {
            'trunk.weight': FakeTensor(4, 4),
            'decoder.weight': FakeTensor(12, 4),
            'target_head.bias': FakeTensor(3),
        }

        compatible, unknown, shape_mismatches = filter_compatible_state_dict(source, target)

        self.assertEqual(list(compatible), ['trunk.weight'])
        self.assertEqual(unknown, ['legacy.weight'])
        self.assertEqual(shape_mismatches, {'decoder.weight': ((10, 4), (12, 4))})


if __name__ == '__main__':
    unittest.main()
