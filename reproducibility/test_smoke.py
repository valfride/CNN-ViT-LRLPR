"""Standard-library-only smoke safety and frozen-path parity checks."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproducibility import smoke
from reproducibility.common import check_weights, new_output


def canonical_function(body):
    # Python 3.12 added FunctionDef.type_params. Ignore that empty schema field,
    # not scientific statements, so the frozen proof also works on paper Python3.9.
    def normalize(value):
        if isinstance(value, ast.AST):
            return [type(value).__name__, {k: normalize(v) for k, v in ast.iter_fields(value)
                    if k != "type_params"}]
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    function = ast.parse("def original(): pass").body[0]
    function.body = body
    value = normalize(ast.Module(body=[function], type_ignores=[]))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SmokeTests(unittest.TestCase):
    def test_frozen_inference_arithmetic_and_native_decoders(self):
        tree = ast.parse((smoke.ROOT / "test.py").read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "infer_sequence")
        # Drop only docstring, interface guards, local args construction, observer callbacks and return.
        start = next(i for i, n in enumerate(helper.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Tuple))
        body = helper.body[start:-1]
        class RemoveObservers(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == "observe is not None":
                    return None
                return self.generic_visit(node)
        body = [RemoveObservers().visit(n) for n in body]
        self.assertEqual(canonical_function(body), 'd68ac522508381483b481ed68abc5ee4668f0458cbe707e3cd2f586b89d2c214')

    def test_frozen_efficiency_forward(self):
        tree = ast.parse(smoke.BENCHMARK.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_logits_once")
        self.assertEqual(canonical_function(helper.body[1:]), '8ec519067d0abf2edd772dddbc1166dc7ac1dd033244a6a50461dc72bb93ee11')

    def test_uuid_representation(self):
        self.assertEqual(smoke.normalize_gpu_uuid("GPU-AbC-123"), smoke.normalize_gpu_uuid("abc-123"))
        self.assertNotEqual(smoke.normalize_gpu_uuid("GPU-AbC-123"), smoke.normalize_gpu_uuid("abc-124"))

    def test_matrix_scope(self):
        selected = smoke.selected_assets()
        self.assertEqual(len(selected), 14)
        self.assertEqual(sum("name" in a for a in selected), 8)
        self.assertTrue(all(a["seed"] == 42 for a in selected if "name" not in a))
        self.assertEqual(smoke.units(selected[0]), [(f, r) for f in (1,3,5) for r in smoke.RULES])
        self.assertEqual(smoke.units(selected[-1]), [(1,"bayes")])

    def test_gpu_guard(self):
        s = dict(uuid="expected", name="Quadro RTX 8000", free_mib=47000,
                 utilization=0, processes=[dict(pid=123)])
        smoke.guard(s, "expected", {123})
        for change in (dict(uuid="wrong"), dict(free_mib=39999), dict(utilization=11),
                       dict(processes=[dict(pid=456)])):
            with self.assertRaises(RuntimeError):
                smoke.guard(dict(s, **change), "expected", {123})
        smoke.guard(dict(s, utilization=100), "expected", {123}, active=True)

    def test_gpu_transient_retries_bounded(self):
        with patch.object(smoke,"gpu_snapshot",return_value=dict(uuid="wrong",name="other")), patch.object(smoke.time,"sleep"):
            with self.assertRaises(RuntimeError):
                smoke.ready("expected")
            self.assertEqual(smoke.gpu_snapshot.call_count, 3)

    def test_missing_and_mismatched_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mapping = root / "weights.json"
            mapping.write_text('{}')
            with self.assertRaisesRegex(ValueError,"Missing approved"):
                check_weights(mapping, [dict(id="one")])
            mapping.write_text('{"one":"checkpoint.pth"}')
            (root / "checkpoint.pth").write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                check_weights(mapping,[dict(id="one",checkpoint_sha256="0"*64)])

    def test_outputs_never_reuse_or_enter_checkout(self):
        with self.assertRaises(ValueError):
            new_output(smoke.ROOT / "private-log")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "new"
            self.assertEqual(new_output(p), p)
            with self.assertRaises(FileExistsError):
                new_output(p)

    def test_decoding_and_fixed_slot_interface(self):
        class Evaluator:
            @staticmethod
            def positional_match_count(prediction, target):
                return sum(a == b for a,b in zip(prediction[:7], target))
        for value in ("", "0", "ABCDEFG", "ABCDEFGH"):
            self.assertTrue(smoke.validate_decoded(value, -1., 0.5, Evaluator)["fixed_seven_position_interface"])
        for value in ("lowercase", "A-1"):
            with self.assertRaises(ValueError):
                smoke.validate_decoded(value, -1., 0.5, Evaluator)
        with self.assertRaises(ValueError):
            smoke.validate_decoded("", float('nan'), 0.5, Evaluator)


if __name__ == "__main__":
    unittest.main()
