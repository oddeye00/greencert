"""Tiny complete numeric graph and cached local-arithmetic replay checks."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from flint import arb, ctx

from arb_deep_transformer_forward import arb_deep_logits
from checkpointed_green_products import StoredRows
from final_scale_arb_codec_v1 import decode_vector, encode_vector
from final_scale_artifacts_v1 import BINDINGS, PhaseReader, PhaseWriter
from final_scale_numeric_core_v1 import build, raw_sha
from final_scale_protocol_v1 import scientific_settings
from final_scale_response_v1 import propagate
from final_scale_selection_v1 import load_array_reference
from outward_green_momentum import adjoint_hvp_direction, rounded_recurrence_step
from prospective_ledger_v1 import sync_directory
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_hvp_grokking import TransformerConfig, make_template, flat_spec, flatten_parameters, make_disjoint_split


def noop(*args):
    pass


SYNC = sync_directory if os.name == "posix" else noop


class NumericCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory(prefix="greencert-numeric-graph-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name) / "numeric"
        cls.config = cfg = TransformerConfig(modulus=3, model_dim=4, hidden_dim=16, heads=2,
            depth=2, normalization="layernorm", seed=0, train_fraction=.6, learning_rate=.003,
            momentum=.9, weight_decay=.01, dtype="float64")
        cls.template = template = make_template(cfg)
        cls.spec = spec = flat_spec(template)
        cls.parameter = parameter = flatten_parameters(template).numpy()
        cls.velocity = velocity = np.zeros_like(parameter)
        cls.data = data = make_disjoint_split(cfg)
        cls.eps = eps = [(b.norm1.eps, b.norm2.eps) for b in template.blocks]
        cls.clock = clock = build_streaming_transformer_centerline(cfg, template, spec, data[0], data[1],
            torch.tensor(parameter), torch.tensor(velocity), maximum_horizon=2, sweeps=4)["scaled_center"].numpy()
        cls.options = options = scientific_settings()["construction"]
        cls.bindings = bindings = {key: hashlib.sha256(key.encode()).hexdigest() for key in BINDINGS}
        writer = PhaseWriter(cls.root, role="construction", bindings=bindings, guard=noop, sync=SYNC)
        prior = ctx.prec
        cls.result = build(writer, parameter=parameter, unscaled_velocity=velocity, clock=clock,
            train_pairs=data[0].numpy(), train_labels=data[1].numpy(), evaluation_pairs=data[4].numpy(),
            evaluation_labels=data[5].numpy(), template=template, spec=spec, config=cfg, normalization_eps=eps,
            options=options, probe_seed=3, probability_ratio=(1, 30000000), guard=noop)
        if ctx.prec != prior:
            raise AssertionError("numeric constructor leaked Arb precision")
        seal = writer.seal({"scope": "tiny_numeric_fixture_no_scientific_certificate"}, maximum_bytes=100000000)
        cls.reader = PhaseReader(cls.root, expected_manifest_sha256=seal["manifest_sha256"],
            expected_role="construction", expected_bindings=bindings, guard=noop)
        cls.reference = np.array(load_array_reference(cls.reader, cls.result["reference"],
            path="reference/scaled.npy", shape=(3, 1056), dtype="<f8"), copy=True)

    @classmethod
    def balls(cls, record):
        bits = record["codec"]["precision_bits"]
        shape = record["array"]["shape"]
        encoded = load_array_reference(cls.reader, record["array"], path=record["array"]["path"], shape=shape, dtype="<i8")
        try:
            return decode_vector(encoded, precision_bits=bits, guard=noop)
        finally:
            encoded._mmap.close()

    def test_complete_graph_is_evidence_not_an_issued_event(self):
        self.assertEqual(self.result["status"], "complete_numeric_evidence_constructed")
        self.assertFalse(self.result["event_certificate_issued"])
        self.assertTrue(self.result["independent_semantic_replay_required"])
        green = self.reader.record("green_capture.json")
        self.assertEqual(len(green["kernels"]), 16)
        self.assertEqual(green["result"]["hvp_calls"], 16)
        self.assertEqual(len(self.result["neural_rows"]), 2)
        self.assertEqual(len(self.result["point_logits"]), 3)

    def test_local_response_replayed_from_captured_intervals(self):
        n = len(self.parameter)
        checks = []
        def derivatives(j, direction):
            record = self.reader.record(f"response_kernels/step_{j:03d}.json")
            self.assertEqual(record["parameter_sha256"], raw_sha(self.reference[j, :n]))
            self.assertEqual(record["direction_sha256"], raw_sha(direction))
            self.assertEqual(record["chunks"], [[0, 5]])
            return self.balls(record["gradient"]), self.balls(record["hvp"])
        def compare(j, values, row):
            stored = self.reader.record(f"response/step_{j:03d}.json")
            actual = load_array_reference(self.reader, stored["array"], path=f"response/step_{j:03d}.npy", shape=(2*n,), dtype="<f8")
            self.assertTrue(np.array_equal(values, actual))
            self.assertEqual(row, stored["recurrence"])
            checks.append(j)
        result = propagate(self.reference, self.parameter, self.velocity, derivatives,
            learning_rate=self.config.learning_rate, momentum=self.config.momentum,
            precision_bits=self.options["response_bits"], guard=noop, persist=compare)
        stored = self.reader.record("response/summary.json")["result"]
        self.assertEqual(list(result["parameter_norms"]), stored["parameter_norms"])
        self.assertEqual(result["residual_upper"], stored["residual_upper"])
        self.assertEqual(checks, [0, 1, 2])

    def test_all_green_rows_replayed_from_captured_hvp_balls(self):
        n, H, query, rows_checked = len(self.parameter), 2, 0, 0
        old = ctx.prec
        ctx.prec = self.options["green_hvp_bits"]
        try:
            for probe in range(4):
                current = StoredRows(self.root / "green" / f"probe_{probe:03d}")
                for power in (1, 2):
                    for transpose in (False, True):
                        kind = "adjoint" if transpose else "forward"
                        target = StoredRows(self.root / "green" / f"probe_{probe:03d}_power_{power:02d}_{kind}")
                        previous = None
                        for j in (range(H-1, -1, -1) if transpose else range(H)):
                            actual, injection = target.row(j), current.row(j)
                            if previous is None:
                                self.assertTrue(np.array_equal(actual, injection))
                            else:
                                step = j+1 if transpose else j
                                direction = adjoint_hvp_direction(previous) if transpose else [arb(float(v)) for v in previous[:n]]
                                meta = self.reader.record(f"green_kernels/query_{query:06d}.json")
                                self.assertEqual((meta["query"], meta["probe"], meta["power"], meta["transpose"], meta["step"]),
                                                 (query, probe, power, transpose, step))
                                self.assertEqual(meta["encoded_direction_sha256"], raw_sha(encode_vector(direction,
                                    precision_bits=self.options["green_hvp_bits"], guard=noop)))
                                row = rounded_recurrence_step(previous, injection, self.balls(meta["hvp"]),
                                    learning_rate=self.config.learning_rate, momentum=self.config.momentum, transpose=transpose)
                                self.assertTrue(np.array_equal(actual, row["next_state"]))
                                self.assertEqual(row["local_residual_norm_upper"], target.summary["rows"][j]["local_residual_norm_upper"])
                                query += 1
                            previous = actual
                            rows_checked += 1
                        current = target
            self.assertEqual((query, rows_checked), (16, 32))
        finally:
            ctx.prec = old

    def test_point_logits_contain_independent_higher_precision_forward(self):
        old = ctx.prec
        ctx.prec = 256
        try:
            for j in range(3):
                meta = self.reader.record(f"point_logits/step_{j:03d}.json")
                captured = self.balls(meta["values"])
                independent = arb_deep_logits(self.reference[j, :len(self.parameter)], self.data[4], self.spec,
                                               self.config, normalization_eps=self.eps).entries()
                self.assertEqual(len(captured), 6)
                for lower_precision, higher_precision in zip(captured, independent):
                    self.assertTrue(lower_precision.contains(higher_precision))
        finally:
            ctx.prec = old

    def test_all_training_examples_and_terminal_output_population_present(self):
        first = self.reader.record("neural/step_001/summary.json")
        last = self.reader.record("neural/step_002/summary.json")
        self.assertEqual(first["result"]["training_count"], 5)
        self.assertEqual(last["result"]["training_count"], 0)
        self.assertEqual(first["result"]["evaluation_count"], 2)
        self.assertEqual(last["result"]["evaluation_count"], 2)
        self.assertEqual(first["result"]["parameter_sha256"], raw_sha(self.reference[1, :len(self.parameter)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
