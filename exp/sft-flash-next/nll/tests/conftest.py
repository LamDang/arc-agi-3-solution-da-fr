from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3] / "reap-flash-next" / "tests"))
sys.path.insert(0, str(HERE.parents[3] / "reap-flash-next"))
