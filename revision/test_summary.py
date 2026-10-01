"""Scientific reporting safeguards: independent units and complete identities."""
import unittest
from summarize import statistics, support_statistics, validate_jobs


class SummaryTests(unittest.TestCase):
    def test_dependent_pairs_are_not_independent_observations(self):
        results=[{'selected':{'memory':2},'support':[['c3',0,0,i]]} for i in (0,0,1)]
        summary,frequencies=support_statistics(results)
        self.assertEqual(summary['n_runs'],3)
        self.assertEqual(summary['n_pairs'],3)
        self.assertAlmostEqual(summary['mean'],1/3)
        self.assertIsNone(summary['ci95'])
        self.assertEqual(sorted(r['frequency'] for r in frequencies),[1/3,2/3])

    def test_memory_strata_and_inclusion_denominators(self):
        results=[{'selected':{'memory':m},'support':[['c3',0,0,0]]} for m in (2,2,4)]
        summary,frequencies=support_statistics(results)
        self.assertEqual(summary['n_pairs'],1)
        self.assertEqual(summary['mean'],1)
        self.assertEqual([r['eligible_runs'] for r in frequencies],[2,1])

    def test_student_interval_uses_repetitions(self):
        s=statistics([1.,2.,3.])
        self.assertEqual(s['n'],3)
        self.assertEqual(s['mean'],2)
        self.assertEqual(s['sd'],1)
        self.assertAlmostEqual(s['ci95'][1],4.484137711719546)
        self.assertIsNone(statistics([2])['ci95'])
        with self.assertRaises(ValueError):statistics([float('nan')])

    def test_duplicate_cannot_replace_missing_repetition(self):
        cfg={'cases':[{'name':'a'}],'pnr_db':[None],'methods':['sbl'],'repeats':2}
        r={'job':{'case':{'name':'a'},'pnr':None,'method':'sbl','repeat':0},
           'status':'ok','selected':{'converged':True}}
        with self.assertRaises(ValueError):validate_jobs([r,r],cfg)


if __name__=='__main__':unittest.main()
