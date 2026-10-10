"""Check arithmetic ports against preserved source, independently of launch code."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT.parent/'experiments/legacy-training'


def definitions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

def dump(node):return ast.dump(node,include_attributes=False)


@unittest.skipUnless(LEGACY.exists(),'Historical sources are reviewed locally before dispatch')
class ArithmeticPortTests(unittest.TestCase):
    def test_pure_head_functions_preserved(self):
        new=definitions(ROOT/'components/head.py')
        for file,names in [('target_only_head.py',['target_positions','_MaskedFrozenHead','target_logits','target_cross_entropy']),('cce_target_loss.py',['cce_target_loss']),('liger_target_loss.py',['liger_target_loss'])]:
            old=definitions(LEGACY/file)
            for name in names:self.assertEqual(dump(new[name]),dump(old[name]),name)

    def test_ple_hash_lookup_queue_arithmetic_preserved(self):
        old=definitions(LEGACY/'ple_preparation.py');new=definitions(ROOT/'components/ple.py')
        for name,node in old.items():self.assertEqual(dump(node),dump(new[name]),name)

    def test_indexer_selection_prefix_preserved(self):
        native=definitions(ROOT/'tests/fixtures/qsa_indexer_reference.py')['Qwen4ExpTextQSAIndexer']
        old=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        new=definitions(ROOT/'components/attention.py')['DirectBiasIndexer'].body[0]
        count=next(i for i,n in enumerate(old.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kv_length' for t in n.targets))
        self.assertEqual([dump(n) for n in old.body[:count]],[dump(n) for n in new.body[:count]])

if __name__ == '__main__':unittest.main()
