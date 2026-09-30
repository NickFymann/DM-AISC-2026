import subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
cmd=[str(p/".venv/bin/python"),"-u",str(p/"run_experiments.py"),"--config",str(p/"benchmark.json"),"--output",str(p/"benchmark_20260930")]
r=subprocess.run(cmd,cwd=p)
if r.returncode==0:
 r=subprocess.run([str(p/".venv/bin/python"),str(p/"summarize.py"),str(p/"benchmark_20260930")],cwd=p)
sys.exit(r.returncode)
