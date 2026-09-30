"""Verify memory-aware queueing and physical-core assignment, plus resume tests."""
import os
os.environ['SCHEDULER_MODULE'] = 'parallel_continue_v2'
from test_parallel_continue import SchedulerTests
from pathlib import Path
import tempfile
import unittest
import parallel_continue_v2 as pc


class QueueTests(unittest.TestCase):
    def test_bypass_large_waiting_job(self):
        jobs = [dict(case=dict(quintic=True), method='sbl'),
                dict(case=dict(quintic=False), method='ridge')]
        self.assertEqual(pc.next_fitting([0, 1], jobs, [], 3*pc.GIB, 2.5*pc.GIB), 1)
        self.assertIsNone(pc.next_fitting([0, 1], jobs, [], 2.9*pc.GIB, 2.5*pc.GIB))

    def test_three_sbl_plus_small_job_fit_but_four_sbl_do_not(self):
        active = [dict(reserve=4.5*pc.GIB, rss=3*pc.GIB)] * 3
        large = dict(case=dict(quintic=True), method='sbl')
        small = dict(case=dict(quintic=False), method='ridge')
        self.assertFalse(pc.can_launch(large, active, 8*pc.GIB, 2.5*pc.GIB))
        self.assertTrue(pc.can_launch(small, active, 8*pc.GIB, 2.5*pc.GIB))

    def test_hyperthreads_share_a_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cpu, core in [(0, 0), (1, 1), (4, 0), (5, 1)]:
                path = root / f'cpu{cpu}/topology'
                path.mkdir(parents=True)
                (path / 'core_id').write_text(str(core))
                (path / 'physical_package_id').write_text('0')
            self.assertEqual(pc.physical_slots([5, 4, 1, 0], root), [0, 1])

    def test_missing_completed_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = dict(case=dict(quintic=False), method='ridge')
            pc.dump(root / 'progress.json', [dict(index=0, job=job, status='ok')])
            with self.assertRaisesRegex(ValueError, 'Missing recorded job directory'):
                pc.load_completed(root, [job])


if __name__ == '__main__':
    unittest.main()
