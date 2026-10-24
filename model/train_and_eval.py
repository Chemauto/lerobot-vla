"""Run an official LeRobot training config, then evaluate its saved checkpoint in our simulator."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--eval-only',action='store_true',help='Recover interrupted evaluation from a completed checkpoint; never train')
    args=parser.parse_args()
    if args.episodes<1:parser.error('episodes must be positive')
    config=args.config.resolve()
    settings=json.loads(config.read_text())
    model=settings['policy']['type']
    accumulation=settings.get('accelerator',{}).get('gradient_accumulation',{}).get('steps',1)
    budget={'micro_batch_size':settings['batch_size'],'gradient_accumulation':accumulation,
            'effective_batch_single_gpu':settings['batch_size']*accumulation,
            'micro_steps':settings['steps'],'optimizer_updates_single_gpu':settings['steps']//accumulation,
            'sample_presentations_single_gpu':settings['steps']*settings['batch_size']}
    day=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    output=(args.output or ROOT/'outputs'/f'{model}{day}').resolve()
    previous=None
    if args.eval_only:
        if args.output is None:parser.error('--eval-only requires the original --output directory')
        if not (output/'run.json').is_file():parser.error('Original run.json is missing')
        previous=json.loads((output/'run.json').read_text())
        if previous['training_config']!=settings:parser.error('Configuration differs from the original run')
        if previous['status']=='completed':parser.error('This evaluation is already complete; no files were replaced')
        checkpoint=output/'train/checkpoints/last'
        progress=checkpoint/'training_state/training_step.json'
        if not progress.is_file() or json.loads(progress.read_text())['step']!=settings['steps']:
            parser.error('Final training state is missing; evaluation recovery requires completed training')
        if not (checkpoint/'pretrained_model/model.safetensors').is_file():parser.error('Final model is missing')
        for key in ('pid','child_pid'):
            proc=Path(f"/proc/{previous.get(key,0)}/cmdline")
            try:command=proc.read_bytes().split(b'\0')
            except FileNotFoundError:continue
            if any(arg in (str(Path(__file__).resolve()).encode(),str(ROOT/'model/play/policy.py').encode()) for arg in command):
                parser.error('Original run is still active; no duplicate evaluation was started')
    elif (output/'train').exists() or (output/'run.json').exists():
        parser.error(f'Run already exists: {output}. Choose a new --output; no files were replaced.')
    output.mkdir(parents=True,exist_ok=True)
    dataset=Path(settings['dataset']['root'])
    if not dataset.is_absolute():dataset=ROOT/dataset
    train=[sys.executable,str(ROOT/'model/train/policy.py'),'--config',str(config),f'--output_dir={output/"train"}']
    evaluate=[sys.executable,str(ROOT/'model/play/policy.py'),'--checkpoint',str(output/'train/checkpoints/last/pretrained_model'),
              '--dataset',str(dataset),'--episodes',str(args.episodes),'--seed',str(args.seed),
              '--output',str(output/'eval')]
    state={'status':'starting','pid':os.getpid(),'model':model,'started':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
           'train_command':train,'eval_command':evaluate,'training_config':settings,'budget_single_gpu':budget,
           'note':'Official trainer/model, local configuration. Matching batch, updates and samples controls the training budget; architectures and algorithm presets still differ.'}
    if previous is not None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        archive=output/'interrupted'/stamp
        archive.mkdir(parents=True,exist_ok=False)
        (archive/'run.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
        for name in ('eval','eval.log'):
            if (output/name).exists():(output/name).rename(archive/name)
        state['started']=previous['started']
        state['evaluation_recovery']={'restarted':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                                      'archive':str(archive),'note':'Restart evaluation only; preserve completed training and interrupted artifacts.'}

    def save(status):
        state['status']=status
        state['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        temp=output/'run.json.tmp';temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(output/'run.json')
        report=(f'# 🧠 {model} 正式训练与评估\n\n状态：**{status}**\n\n'
                f"使用数据 `{dataset.name}`，{settings['steps']}个micro-step，batch{settings['batch_size']}。调用固定版本官方模型与训练器，具体覆盖参数见配置。\n\n"
                f"单卡有效batch {budget['effective_batch_single_gpu']}，预计{budget['optimizer_updates_single_gpu']:,}次参数更新、{budget['sample_presentations_single_gpu']:,}次样本呈现。\n\n"
                '- `train/`：checkpoint与训练状态。\n- `train.log`：训练进度与损失。\n'
                '- `eval/`：完成训练后自动生成的新场景结果与双相机视频。\n- `eval.log`：评估日志。\n'
                '- `run.json`：精确参数、状态与错误信息。\n\n'
                f'评估：{args.episodes}回合，种子{args.seed}起，最多3,600步，官方四块入盒判定。使用该模型自己的历史与动作队列，没有ACT诊断干预。\n\n'
                '注意：训练预算相同仍不代表两种模型的最优配方相同。闭环效果以结果为准，不同算法的训练损失不能直接比较。\n')
        if 'evaluation' in state:
            episodes=state['evaluation']['episodes']
            report+=f"\n完整成功：{sum(e['success'] for e in episodes)}/{len(episodes)}；最终入盒数：{[e['cubes_in_box'] for e in episodes]}。\n"
            for i,e in enumerate(episodes):report+=f"\n[回合{i+1}视频](eval/{Path(e['video']).name})\n"
        if 'error' in state:report+=f"\n错误：{state['error']}\n"
        if 'evaluation_recovery' in state:report+='\n🔄 此前评估中断，本次仅重新评估已完成的模型。旧视频与状态保留在 interrupted/，没有重新训练。\n'
        (output/'RUN.md').write_text(report)
        print(f'{status}: {output}',flush=True)

    try:
        stages=[('evaluating',evaluate,'eval.log')] if args.eval_only else [('training',train,'train.log'),('evaluating',evaluate,'eval.log')]
        for status,command,logfile in stages:
            save(status)
            with (output/logfile).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save(status)
                code=child.wait()
            if code:raise RuntimeError(f'{status} exited with code {code}; see {logfile}')
        state['evaluation']=json.loads((output/'eval/results.json').read_text())
        save('completed')
    except Exception as exc:
        state['error']=str(exc);save('failed');raise


if __name__=='__main__':main()
