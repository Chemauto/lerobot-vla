"""Load the pinned official EnvHub entry point without editing upstream sources.

Local LeRobot 0.6.2's outer factory does not forward arbitrary kwargs. The
repository's own make_env accepts them, so use that entry point explicitly.
This executes the pinned trusted upstream env.py without editing its sources.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
from huggingface_hub import hf_hub_download
REPO='lerobot/so101-tabletop-mujoco'
REVISION='12e504e860c369aad1a9a9eab3fdfbac658ed6dd'
DEFAULT_CONFIG=Path(__file__).parent/'configs/official_tabletop.json'

def make_official_env(config_path=DEFAULT_CONFIG, **overrides):
    settings=json.loads(Path(config_path).read_text());settings.update(overrides)
    # The snapshot must include its XML/STL assets, not only env.py.
    path=hf_hub_download(REPO,'env.py',revision=REVISION,local_files_only=True)
    name='so101_pinned_official_env'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name].make_env(n_envs=1,use_async_envs=False,**settings)

if __name__=='__main__':
    envs=make_official_env();env=envs['so101_tabletop'][0]
    try:
        obs,info=env.reset(seed=42)
        print('Official environment:',REPO,REVISION)
        print('Cameras:',{k:v.shape for k,v in obs['pixels'].items()})
        print('State:',obs['agent_pos'].shape,'FPS:',env.envs[0].fps)
        print('Task:',env.envs[0].task_description)
    finally:env.close()
