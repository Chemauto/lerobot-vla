"""Four local dataset variants derived from the pinned official tabletop task.

Official files are read-only. Fixed layouts use a generated external scene.
Single-cube tasks use an isolated instance of the upstream module with its
task-object constant configured to red only. This is a local task adaptation,
not an upstream-supported single-cube benchmark or changed contact physics.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from official_env import load_official_module

ROOT = Path(__file__).resolve().parents[1]
FIXED_POSITIONS = {
    'red_cube': (.18, -.09, .015), 'green_cube': (.26, -.09, .015),
    'blue_cube': (.18, -.01, .015), 'yellow_cube': (.26, -.01, .015),
}


def make_variant_env(settings):
    settings = deepcopy(settings)
    variant = settings.pop('task_variant')
    count, layout = variant['cube_count'], variant['layout']
    if count not in (1, 4) or layout not in ('fixed', 'random'):
        raise ValueError('Supported variants: cube_count1/4, layout fixed/random')
    name = f'so101_variant_{count}_{layout}'
    upstream = load_official_module(name)
    # A separate module namespace prevents changing any four-cube instance.
    # Its World selectors and Env.step both read this task-object constant.
    names = tuple(FIXED_POSITIONS) if count == 4 else ('red_cube',)
    upstream.CUBE_BODIES = names
    official_root = upstream.repo_root()
    if count == 1 or layout == 'fixed':
        tree = ET.parse(official_root/upstream.SCENE_FILE)
        world = tree.getroot().find('worldbody')
        for body in list(world.findall('body')):
            body_name = body.get('name')
            if body_name in FIXED_POSITIONS:
                if body_name not in names: world.remove(body)
                elif layout == 'fixed': body.set('pos', ' '.join(map(str, FIXED_POSITIONS[body_name])))
        # Relative upstream textures stay pinned dependencies; generated local
        # scenes live with configuration, never replace the official scene.
        for asset in tree.getroot().iter():
            file = asset.get('file')
            if file and not Path(file).is_absolute(): asset.set('file', str(official_root/file))
        folder = ROOT/'myscript/configs/scenes'
        folder.mkdir(parents=True, exist_ok=True)
        scene = folder/f'{"one" if count == 1 else "four"}_{layout}.xml'
        content = ET.tostring(tree.getroot(), encoding='unicode')
        if not scene.exists() or scene.read_text() != content:
            temp = scene.with_suffix('.xml.tmp'); temp.write_text(content);temp.replace(scene)
        settings['scene'] = str(scene)
    return upstream.make_env(n_envs=1, use_async_envs=False, **settings)
