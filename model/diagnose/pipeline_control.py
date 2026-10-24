"""Official-trainer positive control: fit one fixed observation/action, with nondegenerate stats."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class,make_pre_post_processors

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    source=LeRobotDataset('local/fixed_scene',root=ROOT/'data/fixed_scene_seed200',video_backend='pyav')
    target_root=args.output/'probe_data'
    feature_keys=['observation.images.front','observation.images.wrist','observation.state','action']
    features={key:deepcopy(source.features[key]) for key in feature_keys}
    for key in source.meta.camera_keys:
        features[key].pop('info',None)
    writer=LeRobotDataset.create('local/act_pipeline_probe',root=target_root,fps=33,features=features,
                                robot_type='so101_follower',use_videos=True,video_backend='pyav',
                                rgb_encoder=RGBEncoderConfig(vcodec='h264',crf=23,preset='fast',g=2),metadata_buffer_size=1)
    for index in [0,425]:
        item=source[index]
        frame={key:item[key].numpy().astype(np.float32) for key in ['observation.state','action']}
        for key in source.meta.camera_keys:
            frame[key]=np.rint(item[key].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        frame['task']=item['task']
        writer.add_frame(frame)
        writer.save_episode(parallel_encoding=False)
    writer.finalize()
    stats=json.loads((target_root/'meta/stats.json').read_text())
    assert np.min(stats['action']['std'])>1e-3
    assert np.min(stats['observation.state']['std'])>1e-3
    note={
        'source_dataset':str(ROOT/'data/fixed_scene_seed200'),'source_frames':[0,425],
        'training_episodes':[0], 'normalization_stats_episodes':[0,1],
        'reason':'Episode1 only prevents zero variance in normalization. It is not a validation sample. Training sees exactly one observation/action pair.',
        'limits':'Diagnostic point-regression only, not a task or temporal-chunk benchmark; does not rule out all simulator/collection/training bugs.'}
    (target_root/'README.md').write_text('# 仅用于流程正对照\n\n两个单帧episode，来源为固定场景第0帧和第425帧。只训练episode0；episode1仅让归一化统计保持非零。不是新采集数据，不用于任务成绩。\n')
    (args.output/'provenance.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
    initial=ROOT/'outputs/act20261023/train/checkpoints/015000/pretrained_model'
    command=[sys.executable,str(ROOT/'model/train/act.py'),'--config',str(ROOT/'model/configs/act_fixed_scene.json'),
             f'--policy.path={initial}','--policy.dropout=0.0',f'--dataset.root={target_root}',
             '--dataset.repo_id=local/act_pipeline_probe','--dataset.episodes=[0]',
             '--steps=300','--batch_size=1','--num_workers=0','--accelerator.mixed_precision=no',
             '--save_freq=300','--log_freq=50',f'--output_dir={args.output / "train"}']
    (args.output/'command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2)+'\n')
    print('TRAINING one pair, 300 updates, official trainer',flush=True)
    with (args.output/'train.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoint=args.output/'train/checkpoints/000300/pretrained_model'
    cfg=PreTrainedConfig.from_pretrained(checkpoint);cfg.device='cuda';cfg.pretrained_backbone_weights=None
    policy=get_policy_class('act').from_pretrained(checkpoint,config=cfg).cuda().eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=checkpoint,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    ds=LeRobotDataset('local/act_pipeline_probe',root=target_root,episodes=[0],video_backend='pyav')
    item=ds[0]
    batch={key:item[key].unsqueeze(0) for key in cfg.input_features}
    batch['task']=[item['task']]
    with torch.inference_mode():prediction=post(policy.predict_action_chunk(pre(batch))).cpu().numpy()[0,0]
    target=item['action'].numpy()
    diff=np.abs(prediction-target)
    result={'prediction':prediction.tolist(),'target':target.tolist(),'absolute_error':diff.tolist(),
            'arm_mae_degrees':float(diff[:5].mean()),'arm_max_error_degrees':float(diff[:5].max()),'gripper_error':float(diff[5]),
            'passed_point_fit':bool(diff[:5].mean()<.5 and diff[:5].max()<1 and diff[5]<1),
            'threshold_note':'Diagnostic tolerance: mean arm error <0.5deg, maximum <1deg, gripper <1/100. Not a grasp-success criterion.',**note}
    (args.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
