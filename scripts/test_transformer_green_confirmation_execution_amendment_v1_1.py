#!/usr/bin/env python3
"""Structural tests for the outcome-blind execution amendment."""
from __future__ import annotations

import unittest

from transformer_certificate_protocol import Candidate
from transformer_green_confirmation_certificate import frozen_candidates, verify_method_seal
from transformer_green_confirmation_execution_amendment_v1_1 import (
    CONSTRUCTION_ABSTENTION,
    IO_RETRIES,
    MAX_IO_ATTEMPTS,
)


class ExecutionAmendmentTests(unittest.TestCase):
    def test_original_method_seal_is_unchanged(self) -> None:
        self.assertIn("code_manifest", verify_method_seal())

    def test_amendment_coordinates_are_frozen_candidates(self) -> None:
        candidates, horizons, _ = frozen_candidates()
        self.assertIn(CONSTRUCTION_ABSTENTION, horizons)
        for candidate in IO_RETRIES:
            self.assertIn(candidate, horizons)
        self.assertEqual(len(set((*IO_RETRIES, CONSTRUCTION_ABSTENTION))), 3)
        self.assertEqual(len(candidates), 23)

    def test_amendment_is_conservative_and_bounded(self) -> None:
        self.assertEqual(CONSTRUCTION_ABSTENTION, Candidate(335, 0.70, 2440))
        self.assertEqual(MAX_IO_ATTEMPTS, 3)
        self.assertEqual(len(IO_RETRIES), 2)


if __name__ == "__main__":
    unittest.main()
