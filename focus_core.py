"""Import the standalone pack's one implementation; never fork the renderer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'ComfyUI-HeroFrameRig'))
from hfr_core import render, make_plan, depth_array, bounded, sample, VERSION
