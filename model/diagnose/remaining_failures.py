"""Replay recorded physics exactly, inspect cube events and hand control over at expert stage boundaries."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"myscript"))

import av
import mujoco
import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class,make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from official_env import make_official_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prefixes",type=int,nargs='+',default=[330,664,960])
    parser.add_argument("--skip-replay",action='store_true',help='Skip repeated inspection of already-audited recordings')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dataset=ROOT/"data/fixed_scene_seed200"
    manifest=json.loads((dataset/"collection_manifest.json").read_text())
    data=pd.concat([pd.read_parquet(p) for p in (dataset/"data").rglob("*.parquet")],ignore_index=True)
    expert_actions=np.stack(data.action)
    expert_states=np.stack(data["observation.state"])
    envs=make_official_env(config_path=dataset/"collector_sources/configs/official_tabletop.json",
                           initial_joints=manifest["initial_joints"])
    env=envs["so101_tabletop"][0].envs[0]
    env.reset(seed=200)
    world=env._world
    site=world.model.site("gripperframe").id
    colors=["red","green","blue","yellow"]
    gripper_body=world.model.body("gripper").id
    hand_bodies=set()
    for body in range(world.model.nbody):
        cursor=body
        while cursor>0:
            if cursor==gripper_body:
                hand_bodies.add(body)
                break
            cursor=int(world.model.body_parentid[cursor])

    def advance(action):
        # The exact two physics operations from pinned SO101Env.step, skipping only rendering.
        world.data.ctrl[world.actuators]=world.to_ctrl(action)
        mujoco.mj_step(world.model,world.data,nstep=world.substeps)
        env._steps+=1

    def measure(step,action):
        contact=set()
        for k in range(world.data.ncon):
            c=world.data.contact[k]
            b1,b2=map(int,world.model.geom_bodyid[[c.geom1,c.geom2]])
            for j,body in enumerate(world.cube_bodies):
                if (b1==body and b2 in hand_bodies) or (b2==body and b1 in hand_bodies):contact.add(j)
        center=world.data.xpos[world.box_body]
        rot=world.data.xmat[world.box_body].reshape(3,3)
        points=world.data.xpos[world.cube_bodies]
        local=(points-center)@rot
        inside=(np.abs(local[:,:2])<.045).all(1)&(local[:,2]<.06)
        row={"step":step,"gripper_target":float(action[5]),"gripper_state":float(world.joint_state()[5]),
             "cubes_in_box":world.cubes_in_box()}
        for j,color in enumerate(colors):
            for axis,value in zip("xyz",points[j]):row[f"{color}_{axis}"]=float(value)
            row[f"{color}_inside"]=bool(inside[j])
            row[f"{color}_hand_contact"]=j in contact
        for axis,value in zip("xyz",world.data.site_xpos[site]):row[f"gripper_{axis}"]=float(value)
        return row

    report={"physics_replay":[],"handovers":[],"limitations":"Privileged state is used only for replay measurement and expert prefix. Handover totals include expert-placed cubes; not autonomous whole-task success."}
    traces=[]
    traces.append(("expert",expert_actions,expert_states,None))
    source=ROOT/"outputs/act20261023/fixed_scene_dropout0/eval"
    for i in range(3):
        trace=json.loads((source/f"seed_200_episode_{i}_trace.json").read_text())
        traces.append((f"episode_{i}",np.array([r["action"] for r in trace],dtype=np.float32),
                       np.array([r["state"] for r in trace]),trace))
    for name,actions,states,trace in ([] if args.skip_replay else traces):
        env.reset(seed=200)
        rows=[];max_error=0.;next_error=0.
        for i,action in enumerate(actions):
            max_error=max(max_error,float(np.abs(world.joint_state()-states[i]).max()))
            advance(action)
            if trace is not None:next_error=max(next_error,float(np.abs(world.joint_state()-np.array(trace[i]["next_state"])).max()))
            row=measure(i+1,action)
            if trace is not None:assert row['cubes_in_box']==trace[i]['cubes_in_box']
            rows.append(row)
        assert max_error<1e-5 and next_error<1e-5,(name,max_error,next_error)
        table=pd.DataFrame(rows);table.to_csv(args.output/f"{name}_physics.csv",index=False)
        cubes={}
        for color in colors:
            lifted=table[f"{color}_z"]>.05
            in_box=table[f"{color}_inside"]
            cubes[color]={"max_height_m":float(table[f"{color}_z"].max()),
                          "first_height_above_5cm":int(table.loc[lifted,'step'].iloc[0]) if lifted.any() else None,
                          "first_in_box":int(table.loc[in_box,'step'].iloc[0]) if in_box.any() else None,
                          "final_inside":bool(in_box.iloc[-1]),
                          "final_position_m":[float(table[f"{color}_{axis}"].iloc[-1]) for axis in 'xyz']}
        record={"run":name,"state_max_error":max_error,"next_state_max_error":next_error,"cubes":cubes,
                "closing_events":table.loc[(table.gripper_target<30)&~(table.gripper_target.shift(fill_value=100)<30),['step','gripper_x','gripper_y','gripper_z']].to_dict('records')}
        report['physics_replay'].append(record)
        print('REPLAY',name,{c:v['final_inside'] for c,v in cubes.items()},flush=True)

    checkpoint=ROOT/"outputs/act20261023/fixed_scene_dropout0/train/checkpoints/005000/pretrained_model"
    cfg=PreTrainedConfig.from_pretrained(checkpoint);cfg.device='cuda';cfg.pretrained_backbone_weights=None
    policy=get_policy_class('act').from_pretrained(checkpoint,config=cfg).cuda().eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=checkpoint,preprocessor_overrides={"device_processor":{"device":"cuda"}})
    for prefix in args.prefixes:
        set_seed(200);env.reset(seed=200)
        for action in expert_actions[:prefix]:advance(action)
        initial_inside=world.cubes_in_box()
        obs=env._obs(world)
        policy.reset();pre.reset();post.reset()
        folder=args.output/f"handover_{prefix:04d}";folder.mkdir()
        video=av.open(str(folder/'rollout.mp4'),'w');stream=video.add_stream('libx264',rate=33)
        stream.width=1280;stream.height=480;stream.pix_fmt='yuv420p'
        rows=[];success=False
        try:
            for step in range(1,1501):
                batch=preprocess_observation({'pixels':{'front':obs['pixels']['cam0'],'wrist':obs['pixels']['cam1']},'agent_pos':obs['agent_pos']})
                batch['task']=[env.task_description]
                with torch.inference_mode():action=post(policy.select_action(pre(batch))).cpu().numpy()[0].astype(np.float32)
                obs,_,success,truncated,info=env.step(action)
                row=measure(step,action);rows.append(row)
                frame=av.VideoFrame.from_ndarray(np.concatenate([obs['pixels']['cam0'],obs['pixels']['cam1']],axis=1),format='rgb24')
                for packet in stream.encode(frame):video.mux(packet)
                if success or truncated:break
        finally:
            for packet in stream.encode():video.mux(packet)
            video.close()
        pd.DataFrame(rows).to_csv(folder/'physics.csv',index=False)
        result={'expert_prefix_frames':prefix,'initial_inside':initial_inside,'model_steps':step,
                'final_inside':world.cubes_in_box(),'net_added_by_model':world.cubes_in_box()-initial_inside,
                'all_four_inside_after_handover':bool(success),'video':str((folder/'rollout.mp4').resolve())}
        report['handovers'].append(result)
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('HANDOVER',json.dumps(result),flush=True)
    envs['so101_tabletop'][0].close()


if __name__=='__main__':main()
