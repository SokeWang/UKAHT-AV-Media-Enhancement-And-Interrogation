import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(CURRENT_DIR, "ukaht-polar-navigator", "scripts", "polar_skill_engine.py")

import importlib.util
spec = importlib.util.spec_from_file_location("polar_skill_engine", ENGINE_PATH)
polar_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polar_module)

PolarSkillEngine = polar_module.PolarSkillEngine
