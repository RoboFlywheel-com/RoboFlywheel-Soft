#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Human review index: all assets, visible approximations, all failures retained."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path

LABELS={'sheet':'布片','garment_cloth':'柔性衣物布面','garment_shell':'衣物／薄壁三维壳','bag_shell':'可折叠空心袋壳',
        'reinforced_shell':'有弯曲支撑的空心壳','sealed_membrane':'封闭膜／体积支撑','filled_volume':'填充体／实体'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True,help='Previous review batch used as the comparison baseline')
    parser.add_argument('--supplement',type=Path,action='append',default=[],
                        help='Use all tasks from this explicit corrective batch, retaining base results')
    args=parser.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    sample_ids=manifest.get('sample_ids') or sorted(p.parent.name for p in root.glob('*/result.json'))
    task_roots={sid:root for sid in sample_ids}
    for supplement in args.supplement:
        supplement=supplement.resolve()
        extra=json.loads((supplement/'manifest.json').read_text())
        for sid in extra['sample_ids']:
            if sid not in task_roots:raise ValueError(f'supplement sample not in base cohort: {sid}')
            task_roots[sid]=supplement
    baseline_path=args.previous/'evidence/task_evidence.jsonl'
    baseline={r['sample_id']:r for r in map(json.loads,baseline_path.read_text().splitlines())} if baseline_path.exists() else {}
    cards=[];rows=[];counts=Counter();profiles=Counter();roles=Counter();completed=0;visible=0
    for sid in sample_ids:
        task_root=task_roots[sid]
        result_path=task_root/sid/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'sample_id':sid,'probes':{}}
        asset_path=Path(result.get('asset',str(Path(manifest.get('input',args.previous))/sid/'asset.json')))
        asset=json.loads(asset_path.read_text());family=asset['geometry_family'];name=asset.get('object_type','')
        role=baseline.get(sid,{}).get('role','unknown');roles[role]+=1
        row={'sample_id':sid,'object_type':name,'family':family,'role':role,
             'execution_root':str(task_root),'probes':{}}
        cells=[];has_issue=False;profile=None
        for kind,label in [('source','原始生成网格（旋转展示，不是仿真）'),('drop','跌落'),('compression','挤压与释放')]:
            directory=task_root/sid/kind;p=directory/'report.json'
            report=json.loads(p.read_text()) if p.exists() else result['probes'].get(kind,{})
            profile=report.get('representation',{}).get('profile',profile)
            status=report.get('status','not_started');counts[kind+':'+status]+=1
            pending=status in {'not_started','preparing'}
            issue=not pending and status not in {'source_preview','simulated_for_review'}
            has_issue=has_issue or issue
            clip=directory/'render/clip.mp4';poster=directory/'render/poster.png'
            if clip.exists():
                video=f'<video controls preload="none" poster="{html.escape(os.path.relpath(poster,root))}" src="{html.escape(os.path.relpath(clip,root))}"></video>'
            else:
                video='<div class="missing">本次仿真没有生成可播放帧；请看左侧源网格及下方原因。</div>'
            error=report.get('error') or report.get('render_error')
            state='等待生成' if pending else '旋转展示' if kind=='source' and not issue else '执行中断（可用帧已保留）' if issue else '待人工审核'
            audit=report.get('render_visibility') or {}
            visible+=bool(clip.exists() and audit.get('valid'))
            warnings=report.get('geometry',{}).get('warnings',[])
            details='<details><summary>表示方式与近似记录</summary><pre>'+html.escape(json.dumps({
                'representation':report.get('representation'),
                'geometry_warnings':warnings,'appearance':report.get('appearance',{}).get('limitation'),
                'visible_frames':audit.get('visible_sampled_frames'),'sampled_frames':audit.get('sampled_frames')},ensure_ascii=False,indent=2))+'</pre></details>'
            if report.get('trajectory'):
                final_speed=report['trajectory'][-1]['max_speed']
                minimum_z=min(s['min_z'] for s in report['trajectory'])
                details+=f'<p>末帧最高粒子速度 {final_speed:.4g} m/s · 最低采样高度 {minimum_z:.4g} m</p>'
            cells.append(f'<div><h3>{label}</h3>{video}<p class="{"issue" if issue else "state"}">{state}</p>'
                         f'<p>{html.escape(str(error or ""))}</p>{details}<a href="{html.escape(os.path.relpath(p,root))}">详细记录</a></div>')
            row['probes'][kind]={k:report.get(k) for k in ('status','error','particle_count','numerics','shell_constraints','render_visibility','warnings')}
            row['probes'][kind]['status']=status
            row['probes'][kind]['geometry_warnings']=warnings
            if row['probes'][kind]['render_visibility']:
                row['probes'][kind]['render_visibility']={k:v for k,v in audit.items() if k!='samples'}
            row['probes'][kind]['video_exists']=clip.exists()
            row['probes'][kind]['report_sha256']=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            if report.get('trajectory'):
                row['probes'][kind]['first_state']=report['trajectory'][0]
                row['probes'][kind]['last_state']=report['trajectory'][-1]
                row['probes'][kind]['dynamics']={
                    'max_sampled_particle_speed_m_s':max(s['max_speed'] for s in report['trajectory']),
                    'minimum_sampled_z_m':min(s['min_z'] for s in report['trajectory']),
                    'simulated_duration_s':report['trajectory'][-1]['time_s'],
                    'scope':'descriptive trajectory statistics, not behavior acceptance'}
        completed+=all(p['status'] not in {'not_started','preparing'} for p in row['probes'].values())
        profiles[profile or 'pending']+=1;row['profile']=profile;rows.append(row)
        old=[]
        if task_root!=root:
            old.append(f'<a href="{sid}/result.json">本轮补跑前记录（保留）</a>')
        for phase in ['calibration','heldout']:
            clip=args.previous/sid/phase/'render/clip.mp4'
            if clip.exists():old.append(f'<a href="{html.escape(str(clip))}">上一版 {phase}</a>')
        tail_speed=max((p.get('last_state',{}).get('max_speed',0) for p in row['probes'].values()),default=0)
        cards.append(f'<article data-family="{family}" data-search="{html.escape((sid+" "+name).lower())}" data-issue="{int(has_issue)}" data-final-speed="{tail_speed}">'
            f'<h2>{html.escape(name)} <small>{html.escape(sid)}</small></h2><p>{family} · {LABELS.get(profile,profile) or "等待生成"} · {role} · {" · ".join(old)}</p>'
            f'<div class="videos">{"".join(cells)}</div></article>')
    summary={'purpose':'all-asset human representation review; no behavior gate filtering','tasks':len(rows),
             'completed_tasks':completed,'roles':dict(roles),'visible_videos':visible,
             'execution_roots':dict(Counter(str(p) for p in task_roots.values())),
             'profiles':dict(profiles),'statuses':dict(counts),'all_source_previews':bool(rows) and counts['source:source_preview']==len(rows),
             'limitations':['source colors are baked to a visual LOD; full PBR materials are not reproduced',
                            'shells use local rest curvature; no recovered sewing pattern or identified wall stiffness',
                            'closed membrane volume compliance is not full gas/fluid thermodynamics']}
    save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    save(root/'review_summary.json',summary)
    (root/'review_evidence.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>生成网格适配 · 第二轮人工审核</title>
<style>body{font:16px system-ui;color:#202630;background:#f3f5f7;max-width:1500px;margin:32px auto;padding:0 20px}article{background:white;border-radius:12px;padding:22px;margin:24px 0}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}video{width:100%;aspect-ratio:1}h2{font-size:22px}h3{font-size:15px}small{display:block;font-size:12px;color:#586575;margin-top:5px}input,select{font:inherit;padding:10px;margin:4px}.issue{color:#b64032}.state{color:#405968}.missing{padding:50px 15px;background:#f1e8e4}pre{white-space:pre-wrap}a{color:#1d5789}@media(max-width:800px){.videos{grid-template-columns:1fr}}</style>
<h1>生成 mesh → Genesis：第二轮人工审核</h1>
<p>全部样本展示。左侧为生成网格旋转展示，中间和右侧为新版仿真；仿真中断也保留已有帧和原因。“待人工审核”不代表物理验收通过。</p>
<p>重点查看：袋体与壳体是否保留空腔和立体形状，落地是否异常弹飞，颜色是否保留，衣物是否自然折叠。封闭膜与填充体分别标示；复杂材料采用明确的近似形式。</p>
<input id="search" placeholder="搜索任务或物体名称" oninput="filter()"><select id="family" onchange="filter()"><option value="">全部类别</option>'''
    page+=''.join(f'<option>{f}</option>' for f in sorted({r['family'] for r in rows}))
    page+='</select><label><input id="issues" type="checkbox" onchange="filter()">只看仿真中断</label>'
    page+='<button onclick="sortMotion()">按末帧速度排序，优先复核残余运动</button>'
    page+=f'<p>{len(rows)} 个样本 · {completed} 项已执行 · {sum(v for k,v in counts.items() if k.endswith(":simulated_for_review"))} 段完整仿真 · {visible} 段通过物体可见性检查</p>'
    page+=''.join(cards)
    page+='''<script>function filter(){let q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('family').value,i=document.getElementById('issues').checked;document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.search.includes(q)||(f&&a.dataset.family!==f)||(i&&a.dataset.issue!=='1'))}function sortMotion(){[...document.querySelectorAll('article')].sort((a,b)=>Number(b.dataset.finalSpeed)-Number(a.dataset.finalSpeed)).forEach(a=>document.body.appendChild(a))}</script></html>'''
    (root/'review.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
