"""Local artifact management; model, optimizer and validation stay in LeRobot."""
import csv
import json
import math
from pathlib import Path
import re
import shutil

STATUS = {'planned': '待训练', 'training': '训练中', 'evaluating': '评估中',
          'completed': '已完成', 'failed': '失败', 'interrupted': '已中断',
          'waiting_for_data': '等待四组数据验证完成', 'data_failed': '数据验证等待中断，未训练'}

LOSS_BEGIN = '<!-- loss-curves -->'
LOSS_END = '<!-- /loss-curves -->'


def parse_losses(text):
    """Use tqdm's exact step: the logger rounds 2800 to `step:3K`."""
    train, validation = {}, {}
    exact_step = None
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    for line in text.splitlines():
        progress = re.findall(r'(\d+)/(\d+)\s*\[', line)
        if progress:
            exact_step = int(progress[-1][0])
        match = re.search(rf'step (\d+): eval_loss=({number})', line)
        if match:
            step, loss = int(match[1]), float(match[2])
            if math.isfinite(loss):
                validation[step] = dict(step=step, loss=loss)
        if 'step:' not in line or 'loss:' not in line:
            continue
        # Plain exact logger integers also work when tqdm was disabled.
        raw = re.search(r'\bstep:(\d+)\s', line)
        step = int(raw[1]) if raw else exact_step
        loss = re.search(rf'(?<!\w)loss:({number})', line)
        if step is None or loss is None or not math.isfinite(float(loss[1])):
            continue
        row = dict(step=step, loss=float(loss[1]))
        for key in ('l1_loss', 'kld_loss'):
            value = re.search(rf'\b{key}:({number})', line)
            if value and math.isfinite(float(value[1])):
                row[key] = float(value[1])
        train[step] = row
    return dict(train=[train[k] for k in sorted(train)],
                validation=[validation[k] for k in sorted(validation)])


def write_loss_curves(run):
    """CPU-only plots from official logs; reuse the existing result directory."""
    run = Path(run)
    log = run/'metadata/train.log'
    if not log.exists():
        return None
    summary_file = run/'metadata/loss_summary.json'
    stat = log.stat()
    fingerprint = [stat.st_size, stat.st_mtime_ns]
    if summary_file.exists() and (run/'loss.png').exists() and (run/'loss.csv').exists():
        saved = json.loads(summary_file.read_text())
        if saved['fingerprint'] == fingerprint:
            return saved
    data = parse_losses(log.read_text(errors='replace'))
    if not data['train'] and not data['validation']:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    components = [key for key in ('l1_loss', 'kld_loss') if any(key in row for row in data['train'])]
    count = 3 if components else 2
    fig, axes = plt.subplots(1, count, figsize=(5*count, 4.2), constrained_layout=True)
    panels = [('train', '训练总 loss', '#2563eb'), ('validation', '验证 loss', '#ea580c')]
    for ax, (phase, title, color) in zip(axes, panels):
        points = data[phase]
        if points:
            ax.plot([p['step'] for p in points], [p['loss'] for p in points], color=color,
                    marker='o' if phase == 'validation' else None, markersize=4, linewidth=1.6)
            if all(p['loss'] > 0 for p in points):
                ax.set_yscale('log')
            if phase == 'validation':
                best = min(points, key=lambda p: p['loss'])
                ax.scatter(best['step'], best['loss'], color='#15803d', marker='*', s=130, zorder=3)
                ax.annotate(f"最低 {best['loss']:.4f} @ {best['step']:,}步", (best['step'], best['loss']),
                            xytext=(0, 13), textcoords='offset points', ha='center', fontsize=9)
        else:
            ax.text(.5, .5, '尚无记录', transform=ax.transAxes, ha='center')
        ax.set(title=title, xlabel='更新步数', ylabel='loss（正值用对数轴）')
        ax.grid(True, which='both', alpha=.2)
    if components:
        ax = axes[2]
        values = []
        for key in components:
            points = [p for p in data['train'] if key in p]
            values += [p[key] for p in points]
            ax.plot([p['step'] for p in points], [p[key] for p in points], label=key, linewidth=1.5)
        if all(value > 0 for value in values):
            ax.set_yscale('log')
        ax.set(title='训练分量（KL未加权）', xlabel='更新步数', ylabel='loss')
        ax.grid(True, which='both', alpha=.2)
        ax.legend()
    fig.suptitle(run.name+' · loss曲线', fontsize=13)
    image_temp = run/'loss.png.tmp'
    fig.savefig(image_temp, format='png', dpi=150)
    plt.close(fig)
    image_temp.replace(run/'loss.png')
    csv_temp = run/'loss.csv.tmp'
    with csv_temp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['阶段', '更新步数', 'loss', 'l1_loss', 'kld_loss'])
        for phase, label in [('train', '训练'), ('validation', '验证')]:
            for row in data[phase]:
                writer.writerow([label, row['step'], row['loss'], row.get('l1_loss', ''), row.get('kld_loss', '')])
    csv_temp.replace(run/'loss.csv')
    summary = dict(fingerprint=fingerprint, train_points=len(data['train']), validation_points=len(data['validation']),
                   latest_train=data['train'][-1] if data['train'] else None,
                   latest_validation=data['validation'][-1] if data['validation'] else None,
                   best_validation=min(data['validation'], key=lambda p:p['loss']) if data['validation'] else None)
    write_json(summary_file, summary)
    return summary


def loss_section(summary):
    if not summary:
        return ''
    return (LOSS_BEGIN+'\n\n## 📉 Loss曲线\n\n![训练与验证loss](loss.png)\n\n'
            '[原始曲线数据](loss.csv)。训练点为官方日志窗口均值，验证点为保存时的留出集损失；不额外平滑。'
            'ACT训练总loss包含加权KL，验证模式使用固定latent，不能将两条数值直接当成同一损失。'
            '不同任务、batch或验证范围的loss不能直接排名。\n\n'+LOSS_END)


def attach_loss_section(path, section):
    if not section or not path.exists():
        return
    text = path.read_text()
    text = re.sub(re.escape(LOSS_BEGIN)+r'.*?'+re.escape(LOSS_END), '', text, flags=re.S).rstrip()
    updated = text+'\n\n'+section+'\n'
    if updated != path.read_text():
        temp = path.with_suffix('.loss.tmp')
        temp.write_text(updated)
        temp.replace(path)


def batch_loss_section(batch):
    lines = [LOSS_BEGIN, '', '## 📉 Loss曲线', '',
             '| 数据集 | 最新训练loss | 最新验证loss | 最低验证loss | 曲线 |',
             '| --- | ---: | ---: | ---: | --- |']
    found = False
    for run in sorted(Path(batch).iterdir()):
        path = run/'metadata/loss_summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text()); found = True
        values = [f"{summary[key]['loss']:.4f}" if summary[key] else '—'
                  for key in ('latest_train','latest_validation','best_validation')]
        lines.append(f"| {run.name} | {' | '.join(values)} | [查看]({run.name}/loss.png) |")
    return '\n'.join([*lines, '', LOSS_END]) if found else ''


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def replace_tree(source, target):
    """Copy a completed official checkpoint, retaining a backup until promotion."""
    temp = target.with_name('.' + target.name + '.copying')
    backup = target.with_name('.' + target.name + '.previous')
    if backup.exists() and not target.exists():
        backup.rename(target)
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(source, temp)
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    temp.rename(target)
    if backup.exists():
        shutil.rmtree(backup)


def collect_checkpoints(run):
    """Observe official `last` (published after save), then retain latest + best.

    Never read a checkpoint still being written. Keep the current official last
    on disk while its trainer is alive; prune earlier numbered saves only after
    their copies have been promoted. The runner removes the staging copy on exit.
    Validation scores are the official logger's four-decimal values, ties earliest.
    """
    run = Path(run)
    staging = run / 'metadata/trainer/checkpoints'
    last = staging / 'last'
    if not last.exists():
        return
    completed = last.resolve()
    if completed.parent != staging.resolve() or not completed.name.isdigit():
        raise ValueError('Unexpected official checkpoint symlink')
    published_step = int(completed.name)
    latest_file = run / 'metadata/latest.json'
    if latest_file.exists() and json.loads(latest_file.read_text())['step'] == published_step:
        return
    text = (run / 'metadata/train.log').read_text(errors='replace')
    losses = {int(step): float(loss) for step, loss in
              re.findall(r'step (\d+): eval_loss=([\d.eE+\-]+)', text)}
    index_file = run / 'metadata/checkpoint_history.json'
    history = json.loads(index_file.read_text()) if index_file.exists() else []
    seen = {item['step'] for item in history}
    best = min(history, key=lambda item: item['validation_loss']) if history else None
    for source in sorted(staging.iterdir()):
        if not source.name.isdigit() or not source.is_dir():
            continue
        step = int(source.name)
        if step > published_step or step in seen:
            continue
        loss = losses.get(step)
        if loss is None or not math.isfinite(loss):
            raise ValueError(f'Checkpoint {step} lacks a finite official validation loss; no guessed best model')
        if not (source / 'pretrained_model/model.safetensors').is_file():
            raise ValueError(f'Incomplete checkpoint {source}')
        info = dict(step=step, validation_loss=loss, criterion='official held-out eval_loss (logged to4decimals; earliest tie)')
        replace_tree(source, run / 'checkpoints/latest')
        if best is None or loss < best['validation_loss']:
            replace_tree(source, run / 'checkpoints/best')
            best = info
            write_json(run / 'metadata/best.json', info)
        history.append(info)
        write_json(index_file, history)
        write_json(run / 'metadata/latest.json', info)
    # Keep the live trainer's current symlink valid; it may publish another save
    # during the copies above. Never delete a future/unprocessed save.
    current = last.resolve() if last.exists() else None
    processed = {item['step'] for item in history}
    for source in staging.iterdir():
        if source.name.isdigit() and int(source.name) in processed and source.resolve() != current:
            shutil.rmtree(source)


def summarize(evaluation):
    episodes = evaluation['episodes']
    n = len(episodes)
    targets = evaluation['target_cube_count'] * n
    if not n or not targets:
        raise ValueError('No completed evaluation episodes')
    return dict(episodes=n,
        grasp_rate=sum(e['grasped_cubes'] for e in episodes) / targets,
        placement_rate=sum(e['cubes_in_box'] for e in episodes) / targets,
        success_rate=sum(e['success'] for e in episodes) / n,
        official_rate=sum(e['official_success_seen'] for e in episodes) / n,
        mean_steps=sum(e['steps'] for e in episodes) / n)


def write_report(run, state):
    """Markdown is the main human report; CSV opens directly in spreadsheets."""
    run = Path(run)
    evaluation_file = run / 'metadata/evaluation/results.json'
    summary = None
    lines = [f"# 🤖 {state['model']} · {state.get('label', run.name)}", '',
             f"状态：**{STATUS.get(state['status'], state['status'])}**。数据集：`{state['dataset']}`。", '',
             '## 📂 看这几个文件就够了', '',
             '- `checkpoints/latest/`：最新模型、处理器、优化器和随机状态，可断点续训。',
             '- `checkpoints/best/`：已保存checkpoint中，官方留出验证损失最低的一份；同分保留较早者。',
             '- `videos/`：独立评估的前3回合，按种子顺序选取，不挑成功片段。',
             '- `results.csv`：汇总及逐回合结果；本README也显示表格。',
             '- `metadata/`：训练配置、日志、验证分数、精确评估记录。', '',
             '## 📏 评估怎么算', '',
             '完整成功＝目标方块全部入盒，目标和实际夹爪开度均≥95，并连续稳定2秒。'
             '首次环境terminated只单独记录，策略继续自主运行；没有专家接管或手动松爪。', '',
             '抓取完成率是仿真代理指标：目标块离开初始高度≥3cm、距离夹爪参考点<8cm、'
             '实际夹爪开度<50，持续≥0.1秒；按不同方块计数，分母为回合数×目标块数，'
             '**不是抓取尝试成功率**。入盒率按回合结束时的块数计算。', '',
             '最佳模型按验证损失选择，不使用最终测试场景挑模型；'
             '验证损失只比较同一模型同一数据集的checkpoint，不能跨算法比较。', '',
             f"训练上限：{state['steps']:,} micro-step；留出20% episode；"
             f"最终独立评估：{state['eval_episodes']}回合，seed {state['eval_seed']}起，最多{state['max_steps']}步。", '']
    best_file = run / 'metadata/best.json'
    if best_file.exists():
        best = json.loads(best_file.read_text())
        lines += [f"最佳保存步数：{best['step']:,}；验证损失：{best['validation_loss']:.4f}（官方日志精度）。", '']
    config_file = run / 'metadata/train_config.json'
    if config_file.exists():
        config = json.loads(config_file.read_text())
        lines += [f"真实batch：{config['batch_size']}；数据加载进程：{config['num_workers']}；"
                  f"验证样本：{'完整留出集' if config['max_eval_samples'] == 0 else str(config['max_eval_samples']) + '帧上限'}。", '']
    history_file = run / 'metadata/checkpoint_history.json'
    if history_file.exists():
        history = json.loads(history_file.read_text())
        if len(history) >= 3:
            tail = [point['validation_loss'] for point in history[-3:]]
            relative_drop = (tail[0] - tail[-1]) / max(abs(tail[0]), 1e-12)
            relative_range = (max(tail) - min(tail)) / max(abs(tail[0]), 1e-12)
            trend = ('仍在下降' if relative_drop >= .02 else
                     '趋平' if relative_range < .05 else '波动或反弹')
            lines += [f"最近三个保存点的验证损失：{tail}，趋势提示：**{trend}**。"
                      '这是粗略的损失趋势，不等于闭环收敛；四组均跑满统一预算，不据此提前停止。', '']
    gpu_file = run / 'metadata/gpu_usage.csv'
    if gpu_file.exists():
        with gpu_file.open() as f:
            values = [float(row['utilization_percent']) for row in csv.DictReader(f)
                      if row['stage'] == 'training']
        if values:
            lines += [f"训练GPU利用率：采样均值{sum(values)/len(values):.1f}%，峰值{max(values):.0f}%"
                      '（包含加载和验证阶段，详细记录见metadata/gpu_usage.csv）。', '']
    if evaluation_file.exists():
        evaluation = json.loads(evaluation_file.read_text())
        if evaluation['episodes']:
            summary = summarize(evaluation)
            lines += ['## 📊 评估结果', '',
                      '| 回合数 | 抓取完成率（代理） | 最终入盒率 | 完整成功率 | 曾触发环境成功 |',
                      '| ---: | ---: | ---: | ---: | ---: |',
                      f"| {summary['episodes']} | {summary['grasp_rate']:.1%} | {summary['placement_rate']:.1%} | {summary['success_rate']:.1%} | {summary['official_rate']:.1%} |", '',
                      '三段视频只用于直观检查；成功率来自全部评估回合。小样本结果不代表稳定的能力上限。', '',
                      '| seed | 抓到块数 | 最终入盒 | 完整成功 | 步数 | 视频 |',
                      '| ---: | ---: | ---: | --- | ---: | --- |']
            for e in evaluation['episodes']:
                video = f"[播放]({Path(e['video']).relative_to(run)})" if e.get('video') else '—'
                lines.append(f"| {e['seed']} | {e['grasped_cubes']} | {e['cubes_in_box']} | {'✅' if e['success'] else '❌'} | {e['steps']} | {video} |")
            with (run / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['类型', 'seed', '目标块数', '抓到块数', '最终入盒数', '完整成功', '曾触发环境成功', '步数', '抓取完成率(代理)', '最终入盒率', '完整成功率'])
                writer.writerow(['汇总', '', evaluation['target_cube_count'] * summary['episodes'],
                    sum(e['grasped_cubes'] for e in evaluation['episodes']), sum(e['cubes_in_box'] for e in evaluation['episodes']),
                    sum(e['success'] for e in evaluation['episodes']), sum(e['official_success_seen'] for e in evaluation['episodes']),
                    round(summary['mean_steps'], 1), f"{summary['grasp_rate']:.1%}", f"{summary['placement_rate']:.1%}", f"{summary['success_rate']:.1%}"])
                for e in evaluation['episodes']:
                    writer.writerow(['回合', e['seed'], evaluation['target_cube_count'], e['grasped_cubes'], e['cubes_in_box'], int(e['success']), int(e['official_success_seen']), e['steps']])
    if summary is None:
        lines += ['## 📊 评估结果', '', '尚未完成评估，暂无成功率或视频。', '']
        with (run / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['类型', 'seed', '目标块数', '抓到块数', '最终入盒数', '完整成功', '曾触发环境成功', '步数', '抓取完成率(代理)', '最终入盒率', '完整成功率'])
            writer.writerow(['待评估', '', '', '', '', '', '', '', '', '', ''])
    if state.get('error'):
        lines += [f"错误：{state['error']}；查看metadata中的日志。", '']
    curve_summary = write_loss_curves(run)
    if curve_summary:
        lines += [loss_section(curve_summary), '']
    (run / 'README.md').write_text('\n'.join(lines) + '\n')
    return summary


def write_batch_report(batch):
    rows = []
    for folder in sorted(batch.iterdir()):
        name = folder.name
        state_file = folder / 'metadata/run.json'
        if not state_file.exists():
            continue
        state = json.loads(state_file.read_text())
        summary = write_report(batch / name, state)
        rows.append((name, state.get('label', name), state, summary))
    lines = ['# 🧪 ' + batch.name, '', '每个数据集独立训练一个模型。尚未评估的项目不填成功率。', '',
             '| 数据集 | 状态 | 抓取完成率（代理） | 最终入盒率 | 完整成功率 |',
             '| --- | --- | ---: | ---: | ---: |']
    for name, label, state, s in rows:
        values = [f"{s[k]:.1%}" if s else '—' for k in ('grasp_rate', 'placement_rate', 'success_rate')]
        lines.append(f"| [{label}]({name}/README.md) | {STATUS.get(state['status'], state['status'])} | {' | '.join(values)} |")
    section = batch_loss_section(batch)
    if section:
        lines += ['', section]
    (batch / 'README.md').write_text('\n'.join(lines) + '\n')
    with (batch / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['数据集', '状态', '评估回合数', '抓取完成率(代理)', '最终入盒率', '完整成功率'])
        for name, label, state, s in rows:
            writer.writerow([label, STATUS.get(state['status'], state['status']), s['episodes'] if s else '',
                             *[f"{s[k]:.1%}" if s else '' for k in ('grasp_rate', 'placement_rate', 'success_rate')]])


def main():
    """Refresh existing results; optional CPU-only companion for a running queue."""
    import argparse
    import time
    p = argparse.ArgumentParser(description=main.__doc__)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--run', type=Path)
    target.add_argument('--batch', type=Path)
    p.add_argument('--watch-pid', type=int, help='Update until this parent process exits; never train')
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()
    if args.interval < 10:
        p.error('interval must be>=10 seconds')

    def identity():
        if args.watch_pid is None:
            return None
        try:
            fields = Path(f'/proc/{args.watch_pid}/stat').read_text().split(') ',1)[1].split()
            return None if fields[0]=='Z' else fields[19]
        except (FileNotFoundError, ProcessLookupError):
            return None
    parent = identity()
    while True:
        runs = [args.run] if args.run else [path for path in args.batch.iterdir() if path.is_dir()]
        for run in runs:
            summary = write_loss_curves(run)
            attach_loss_section(run/'README.md', loss_section(summary))
        if args.batch:
            attach_loss_section(args.batch/'README.md', batch_loss_section(args.batch))
        if args.watch_pid is None or parent is None or identity()!=parent:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
