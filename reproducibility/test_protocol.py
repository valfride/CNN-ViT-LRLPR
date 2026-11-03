"""CPU-only protocol/negative-gate tests; no torch, models or datasets imported."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

from reproducibility import common, run, verify
from temporal_votes import vote_strings
from checkpoint_state_compatibility import clean_wrapper_prefixes, require_complete_state


def aggregate_fixture(table):
    """Schema fixture from frozen aggregates, never represented as a real run."""
    manifest = common.paper_manifest()
    selected = common.assets(table)
    cohort_name = "lplcv2" if table == 3 else "test"
    cohort = manifest["datasets"][cohort_name]
    spec = {"cohort": cohort_name, "phase": cohort["phase"], "samples": cohort["samples"], "fingerprints": cohort["files"]}
    refs = verify.reference("TABLE1_EXPECTED.csv" if table == 1 else "TABLE2_EXPECTED.csv" if table == 2 else "TABLE3_COUNTS.csv") if table != 5 else []
    records = []
    for unit in run.plan(table, selected):
        asset = next(a for a in selected if a["id"] == unit["id"])
        if table == 5:
            m = dict(asset["expected"][str(unit["frames"])])
        else:
            r = next(r for r in refs if r["model"] == asset["name"] and
                     (table == 3 or int(r["F"]) == unit["frames"]) and
                     (table != 2 or r["fusion"] == unit["fusion"]))
            correct = int(r["correct"])
            m = {"sequence_correct": correct, "sequence_total": cohort["samples"],
                 "partial_6_correct": int(r["partial6_correct"]) if table != 2 else correct,
                 "partial_5_correct": int(r["partial5_correct"]) if table != 2 else correct}
        record = dict(unit, model=asset.get("name"), variant=asset.get("variant"),
                      seed=asset.get("seed"), weight_source=asset.get("weight_source"),
                      checkpoint_sha256=asset["checkpoint_sha256"], config_sha256=asset["config_sha256"],
                      dataset=spec, metrics=m)
        if table == 3:
            record["groups"] = {g: {"correct": int(r[p + "_correct"]), "total": int(r[p + "_total"])}
                                for g, p in [("Poor", "poor"), ("Illegible", "illegible")]}
        records.append(record)
    return {"table": table, "validation": False, "records": records}


class ProtocolTests(unittest.TestCase):
    def test_complete_state_loading_preserves_internal_module_names(self):
        class TensorShape:
            shape = (2, 3)
        value = TensorShape()
        name = "decoder.context.module.weight"
        cleaned = clean_wrapper_prefixes({"module.module." + name: value})
        self.assertIs(cleaned[name], value)
        self.assertIs(require_complete_state({name: value}, cleaned), cleaned)
        with self.assertRaises(ValueError):
            clean_wrapper_prefixes({name: value, "module." + name: value})
        with self.assertRaises(ValueError):
            require_complete_state({name: value, "missing": value}, cleaned)

    def test_frozen_reference_accounting(self):
        checked = verify.check_reference_integrity()
        self.assertEqual(checked["Table5_mean_std_pairs"], 18)
        self.assertEqual(len(checked["paper_display_differences"]), 3)
        self.assertEqual(checked["reported_manuscript_values"]["status"], "pass")
        self.assertEqual(checked["direct_count_arithmetic"]["direct_rounding_matches_reported_cells"], 37)
        self.assertFalse(checked["paper_reference"]["latest_commit_assumed"])

    def test_approved_displays_remain_distinct_from_raw_arithmetic(self):
        report = verify.table3_display_verification()
        differences = report["approved_historical_display_differences"]["differences"]
        self.assertEqual([r["paper_display"] for r in differences], ["80.5", "67.7", "63.8"])
        self.assertEqual([r["direct_one_decimal"] for r in differences], ["80.4", "67.6", "63.7"])
        self.assertTrue(all(r["approval_status"] == "approved_reported_display_only" for r in differences))
        document = aggregate_fixture(3)
        ours = next(r for r in document["records"] if r["model"] == "Ours")
        ours["metrics"]["partial_6_percent"] = 100 * 5650 / 7023
        verify.verify_aggregate(document)
        ours["metrics"]["partial_6_percent"] = 80.5
        with self.assertRaisesRegex(ValueError, "Incorrect percentage"):
            verify.verify_aggregate(document)

    def test_paper_display_strict_passes_but_direct_rounding_strict_fails(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify.main(["--references", "--strict-paper-display"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                verify.main(["--references", "--strict-direct-rounding"])
        self.assertEqual(error.exception.code, 1)

    def test_unapproved_display_or_changed_counts_are_rejected(self):
        original = verify.reference
        def altered(name):
            rows = copy.deepcopy(original(name))
            if name == "TABLE3_EXPECTED.csv":
                next(r for r in rows if r["model"] == "CPPD")["exact_percent"] = "34.7"
            return rows
        with patch.object(verify, "reference", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                verify.table3_display_verification()
        ledger = common.load_json(common.HERE / "reference/PAPER_DISPLAY_DISCLOSURES.json")
        for mutation in ["approval", "direct", "count", "duplicate"]:
            bad = copy.deepcopy(ledger)
            if mutation == "approval": bad["approval"]["status"] = "pending"
            if mutation == "direct": bad["differences"][0]["direct_one_decimal"] = "80.5"
            if mutation == "count": bad["differences"][0]["numerator"] += 1
            if mutation == "duplicate": bad["differences"].append(bad["differences"][0])
            with patch.object(verify, "load_json", return_value=bad):
                with self.assertRaises(ValueError): verify.table3_display_verification()

    def test_noninteger_counter_and_boolean_frame_rejected(self):
        for table in [1, 3, 5]:
            bad = aggregate_fixture(table)
            bad["records"][0]["frames"] = True
            with self.assertRaises(ValueError): verify.verify_aggregate(bad)
        bad = aggregate_fixture(5)
        bad["records"][0]["metrics"]["character_correct"] = float(bad["records"][0]["metrics"]["character_correct"])
        with self.assertRaises(ValueError): verify.verify_aggregate(bad)

    def test_exact_model_pins(self):
        models = common.paper_manifest()["models"]
        self.assertEqual(models["ours"]["loss"], "CE")
        self.assertEqual(models["ours"]["source_epoch"], 58)
        self.assertEqual(models["lister"]["source_epoch"], 113)
        self.assertEqual(models["svtrv2"]["source_epoch"], 156)
        for a in common.assets(1) + common.assets(5):
            common.require_hash(common.ROOT / a["config"], a["config_sha256"], a["id"])

    def test_plan_has_no_model_execution(self):
        self.assertEqual(len(run.plan(1, common.assets(1))), 24)
        self.assertEqual(len(run.plan(2, common.assets(2))), 64)
        self.assertEqual(len(run.plan(3, common.assets(3))), 8)
        self.assertEqual(len(run.plan(4, common.assets(4))), 8)
        self.assertEqual(len(run.plan(5, common.assets(5))), 54)
        with patch.object(run.subprocess, "run", side_effect=AssertionError("must not execute")):
            run.main(["plan", "--table", "4"])

    def test_no_execution_without_explicit_flag(self):
        with patch.object(run, "check_sources", side_effect=AssertionError("must stop before preflight")):
            with self.assertRaises(SystemExit):
                run.main(["evaluate", "--table", "1", "--dataset", "/missing", "--weights", "/missing", "--output", "/missing"])

    def test_checkpoint_tampering_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "last.pth"
            p.write_bytes(b"synthetic bytes; never a model")
            mapping = Path(temp) / "weights.json"
            mapping.write_text(json.dumps({"ours": str(p)}))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                common.check_weights(mapping, common.assets(1, ["ours"]))

    def test_output_refuses_checkout_and_existing_directory(self):
        with self.assertRaises(ValueError):
            common.new_output(common.ROOT / "outputs")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                common.new_output(temp)

    def test_all_table_aggregate_schemas_and_frozen_counts(self):
        for table in [1, 2, 3, 5]:
            doc = aggregate_fixture(table)
            checked = verify.verify_aggregate(doc)
            self.assertEqual(checked["records"], {1: 24, 2: 64, 3: 8, 5: 54}[table])
            bad = copy.deepcopy(doc)
            bad["records"][0]["metrics"]["sequence_correct"] -= 1
            with self.assertRaises(ValueError):
                verify.verify_aggregate(bad)

    def test_duplicate_missing_wrong_split_wrong_model_fail(self):
        doc = aggregate_fixture(1)
        for change in ["duplicate", "missing", "split", "checkpoint"]:
            bad = copy.deepcopy(doc)
            if change == "duplicate": bad["records"].append(bad["records"][0])
            if change == "missing": bad["records"].pop()
            if change == "split": bad["records"][0]["dataset"]["phase"] = "validation"
            if change == "checkpoint": bad["records"][0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaises(ValueError): verify.verify_aggregate(bad)

    def test_table4_reference_and_missing_ar(self):
        self.assertEqual(verify.verify_efficiency(common.HERE / "reference/TABLE4_ARCHIVED.csv")["models"], 8)
        with tempfile.TemporaryDirectory() as temp:
            rows = verify.reference("TABLE4_ARCHIVED.csv")
            p = Path(temp) / "summary.csv"
            with p.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(r for r in rows if r["model"] != "SVTRv2-AR")
            with self.assertRaises(ValueError): verify.verify_efficiency(p)

    def test_character_and_sequence_ties_preserve_native_rules(self):
        self.assertEqual(vote_strings(["AAAAAAA", "BBBBBBB"], [0.8, 0.8], "majority")[0], "AAAAAAA")
        self.assertEqual(vote_strings(["AAAAAAA", "BBBBBBB", "BAAAAAA"], [0.5, 0.5, 0.5], "char_majority")[0], "BAAAAAA")
        with self.assertRaises(ValueError): vote_strings(["AAAAAAA"], [float("nan")], "majority")

    def test_table3_private_join_rejects_wrong_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "selection.json"
            # A missing/duplicate population fails before any data can be accepted.
            manifest.write_text(json.dumps({"samples": [{"track_id": "synthetic"}]}))
            with self.assertRaises(ValueError): run.table3_groups(Path(temp) / "predictions.csv", manifest, {})

    def test_table3_complete_join_and_corrupt_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            samples = [{"track_id": f"synthetic-{i}", "ocr": "AAAAAAA",
                        "source_legibility": "Poor" if i < 6240 else "Illegible"}
                       for i in range(7023)]
            (p / "manifest.json").write_text(json.dumps({"samples": samples}))
            fields = ["track", "prediction", "ground_truth", "correct", "positional_matches"]
            def write_records(corrupt=False):
                with (p / "predictions.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                    for i, sample in enumerate(samples):
                        writer.writerow(dict(track=sample["track_id"], prediction="AAAAAAA",
                                             ground_truth="BBBBBBB" if corrupt and i == 0 else "AAAAAAA",
                                             correct="true", positional_matches=7))
            write_records()
            metrics = dict(sequence_correct=7023, partial_6_correct=7023, partial_5_correct=7023)
            groups = run.table3_groups(p / "predictions.csv", p / "manifest.json", metrics)
            self.assertEqual(groups["Illegible"], {"correct": 783, "total": 783})
            write_records(corrupt=True)
            with self.assertRaises(ValueError): run.table3_groups(p / "predictions.csv", p / "manifest.json", metrics)


if __name__ == "__main__":
    unittest.main()
