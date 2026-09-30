"""Journal integrity, RAM admission, and isolated Linux scheduler smoke tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import parallel_continue as pc


class SchedulerTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        cfg = dict(repeats=1, cases=[dict(name='small', quintic=False)], pnr_db=[None],
                   methods=['ridge', 'sbl', 'omp'], memories=[2], max_run_hours=1,
                   max_failed_jobs=2, task_timeout_sec=30)
        pc.dump(source / 'config.json', cfg)
        worker = source / 'run_experiments.py'
        worker.write_text('import json,sys,time\nfrom pathlib import Path\n'
            'p=Path(sys.argv[sys.argv.index("--worker")+1])\n'
            'time.sleep(.2)\n'
            'p.with_name("result.json").write_text(json.dumps({"job":json.loads(p.read_text()),"status":"ok"}))\n')
        summary = source / 'summarize.py'
        summary.write_text('from pathlib import Path\nimport sys\n'
                           '(Path(sys.argv[1])/"aggregated.txt").write_text("ok")\n')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (worker, summary)}
        pc.dump(root / 'manifest.json', dict(config=cfg, source_sha256=hashes))
        pc.dump(root / 'progress.json', [])
        return cfg, pc.jobs_for(cfg)

    def test_order_is_original_nested_order(self):
        cfg = dict(repeats=2, cases=[{'quintic': False}], pnr_db=[None, 20], methods=['ridge', 'sbl'])
        jobs = pc.jobs_for(cfg)
        self.assertEqual(len(jobs), 8)
        self.assertEqual([(j['repeat'], j['pnr'], j['method']) for j in jobs[:4]],
                         [(0, None, 'ridge'), (0, None, 'sbl'), (0, 20, 'ridge'), (0, 20, 'sbl')])

    def test_remaining_ram_growth_reserved(self):
        heavy = dict(case=dict(quintic=True))
        active = [dict(reserve=4.5*pc.GIB, rss=2*pc.GIB)] * 2
        self.assertFalse(pc.can_launch(heavy, active, 11*pc.GIB, 2*pc.GIB))
        self.assertTrue(pc.can_launch(heavy, active, 12*pc.GIB, 2*pc.GIB))

    def test_tampered_worker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            pc.verify_sources(root)
            (root / 'source/run_experiments.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                pc.verify_sources(root)

    def test_atomic_result_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, jobs = self.fixture(root)
            folder = root / 'job_0000'
            folder.mkdir()
            pc.dump(folder / 'job.json', jobs[0])
            pc.dump(folder / 'result.json', dict(job=jobs[0], status='ok'))
            self.assertEqual(pc.load_completed(root, jobs)[0]['status'], 'ok')

    def test_missing_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, jobs = self.fixture(root)
            folder = root / 'job_0000'
            folder.mkdir()
            pc.dump(folder / 'job.json', jobs[0])
            pc.dump(root / 'progress.json', [dict(index=0, job=jobs[0], status='ok')])
            with self.assertRaisesRegex(ValueError, 'Missing successful result'):
                pc.load_completed(root, jobs)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process handoff')
    def test_resume_preserves_completed_and_adopts_worker(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, jobs = self.fixture(root)
            for i in (0, 1):
                (root / f'job_{i:04d}').mkdir()
                pc.dump(root / f'job_{i:04d}/job.json', jobs[i])
            preserved = root / 'job_0000/result.json'
            pc.dump(preserved, dict(job=jobs[0], status='ok'))
            before = preserved.read_bytes(), preserved.stat().st_mtime_ns
            # A short worker is adopted while a second task runs concurrently.
            worker = root / 'source/run_experiments.py'
            child = subprocess.Popen([sys.executable, str(worker), '--worker', str(root / 'job_0001/job.json')])
            process = psutil.Process(child.pid)
            pc.dump(root / 'handoff.json', dict(run=str(root), original_started=time.time(),
                coordinator=dict(pid=99999999, created=0), adopted=[dict(index=1,
                pid=child.pid, created=process.create_time(), cmdline=process.cmdline())]))
            done = subprocess.run([sys.executable, str(Path(pc.__file__)), '--run', str(root),
                '--handoff', str(root / 'handoff.json')], capture_output=True, text=True, timeout=20)
            child.wait(timeout=5)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(pc.read(root / 'completion.json')['jobs'], 3)
            self.assertEqual(before, (preserved.read_bytes(), preserved.stat().st_mtime_ns))
            rows = pc.read(root / 'progress.json')
            self.assertEqual(len({r['index'] for r in rows}), 3)
            self.assertEqual(rows[1]['timing_phase'], 'parallel_adopted')
            self.assertTrue((root / 'aggregated.txt').exists())
            self.assertFalse(pc.read(root / 'timing_provenance.json')['performance_comparison_valid'])


if __name__ == '__main__':
    unittest.main()
