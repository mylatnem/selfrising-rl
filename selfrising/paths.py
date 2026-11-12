"""工程内路径的统一入口, 避免各脚本各自拼路径。"""

from pathlib import Path

# selfrising/paths.py -> selfrising/ -> 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
MESHES_DIR = ASSETS_DIR / "meshes"
MODEL_XML = ASSETS_DIR / "robo1.xml"

REFERENCE_DIR = PROJECT_ROOT / "reference"
REFERENCE_JSON = REFERENCE_DIR / "reference-learning.json"

CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"


def runs_dir(sub: str = "") -> Path:
    """返回 runs/ 下的子目录, 不存在则创建。"""
    path = RUNS_DIR / sub if sub else RUNS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
