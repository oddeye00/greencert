"""Pre-seal engineering suite; never launches a registered scientific model."""
import unittest

import test_final_scale_arb_codec_v1
import test_final_scale_numeric_replay_v1
import test_final_scale_observation_v1
import test_final_scale_seal_controller_v1
import test_final_scale_engine_v1
import test_final_scale_runtime_v1


def main():
    modules = (test_final_scale_arb_codec_v1, test_final_scale_numeric_replay_v1,
        test_final_scale_observation_v1, test_final_scale_seal_controller_v1,
        test_final_scale_engine_v1, test_final_scale_runtime_v1)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
