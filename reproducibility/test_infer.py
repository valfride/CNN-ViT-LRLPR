"""CPU-only identity and path guards for the single-state reader example."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproducibility import infer


class InferenceEntryTests(unittest.TestCase):
    def test_wrong_checkpoint_stops_before_model_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ema_weights.pth';p.write_bytes(b'not the original weights')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                infer.verified_inputs(p,Path(d)/'output')
            self.assertFalse((Path(d)/'output').exists())

    def test_input_scope_rejected_before_asset_work(self):
        for frame,rule in [(2,'bayes'),(1,'unknown')]:
            with self.assertRaises(ValueError):infer.infer('missing','unused',frames=frame,fusion=rule)

    def test_alias_mapping_is_hash_based_not_filename_based(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ema_weights.pth';path.write_bytes(b'fixture only')
            with patch.object(infer,'require_hash',return_value=path) as check,patch.object(infer,'check_sources'),patch.object(infer,'new_output',return_value=Path(d)):
                asset,checkpoint,_=infer.verified_inputs(path,Path(d)/'out')
                self.assertEqual(asset['checkpoint_filename'],'last.pth')
                self.assertEqual(check.call_args_list[0].args[1],'1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a')
                self.assertEqual(checkpoint,path)


if __name__=='__main__':unittest.main()
