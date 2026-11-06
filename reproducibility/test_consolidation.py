"""CPU-only regression checks for consolidation and manuscript drift guards."""
import unittest

from .paper_audit import LABELS, compare, model_rows, tables, uncomment
from .run import plan


class ConsolidationTests(unittest.TestCase):
    def test_frame_subset_preserves_protocol(self):
        selected = [{"id": "ours"}]
        self.assertEqual(plan(1, selected, requested_frames=[1]),
                         [{"id": "ours", "table": 1, "frames": 1, "fusion": "bayes"}])
        for table, frames in [(2, [1]), (3, [3]), (4, [1]), (1, [1, 1]), (1, [])]:
            with self.assertRaises(ValueError):
                plan(table, selected, requested_frames=frames)

    def test_commented_optional_and_previous_tables_are_inactive(self):
        text = "\n".join("\\begin{table}\\label{tab:" + label + "}\\end{table}" for label in LABELS)
        text += "\n% \\begin{table}\\label{tab:ablation}\\end{table}"
        text += "\n% \\input{4-experiments/ablation_combined_draft}"
        self.assertEqual(set(tables(text)), set(LABELS))
        self.assertIn(r"\%", uncomment(r"RR (\%) % editorial comment"))
        with self.assertRaises(ValueError):
            tables(text + "\n\\begin{table}\\label{tab:ablation}\\end{table}")
        with self.assertRaises(ValueError):
            tables("")

    def test_native_model_names_and_nested_review_macros(self):
        block = ("SVTRv2-AR~\\cite{laroca2026competition}\n"
                 " & 3 & \\VReview{\\textbf{68.8}} & 67.3 & 63.9 & 64.8 \\\\\n"
                 "SVTRv2~\\cite{du2025svtrv2}\n"
                 " & 3 & 68.5 & 65.9 & 62.5 & 63.4 \\\\\n")
        rows = model_rows(block, 5, True)
        self.assertEqual(rows[("SVTRv2-AR", 3)][0], "68.8")
        self.assertEqual(rows[("SVTRv2", 3)][0], "68.5")
        with self.assertRaises(ValueError):
            compare(rows, {("SVTRv2-AR", 3): ["99.9"]}, "Table2")
        with self.assertRaises(ValueError):
            model_rows(block + block, 5, True)


if __name__ == "__main__":
    unittest.main()
