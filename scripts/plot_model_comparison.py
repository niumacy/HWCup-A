"""
生成所有模型的对比图
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob


def main():
    # 加载所有报告
    reports = {}
    for f in glob.glob('/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/reports/*.json'):
        basename = os.path.basename(f).replace('.json', '')
        try:
            with open(f) as ff:
                d = json.load(ff)
            if 'test_metrics' in d:
                reports[basename] = d['test_metrics']
            elif 'best_metrics' in d:
                reports[basename] = d['best_metrics']
        except:
            pass

    # 也包括各个 latefusion seed (从 ensemble 报告读取)
    ensemble_report_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/reports/ensemble_v2_report.json'
    if os.path.exists(ensemble_report_path):
        with open(ensemble_report_path) as f:
            ensemble = json.load(f)
        if 'individual' in ensemble:
            for name, metrics in ensemble['individual'].items():
                reports[name] = metrics

    # 排序
    names = sorted(reports.keys(), key=lambda n: reports[n].get('F1_weighted', 0), reverse=True)

    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('模型对比 (在测试集上)', fontsize=14, fontweight='bold')

    # Acc & F1
    x = np.arange(len(names))
    accs = [reports[n].get('Accuracy', 0) for n in names]
    f1s = [reports[n].get('F1_weighted', 0) for n in names]

    axes[0].bar(x, accs, color='steelblue', alpha=0.7, label='Accuracy')
    axes[0].bar(x, f1s, color='lightcoral', alpha=0.7, label='F1')
    axes[0].set_title('Accuracy & F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # MAE (越低越好)
    maes = [reports[n].get('MAE', 0) for n in names]
    colors_mae = ['green' if m < 0.7 else 'orange' if m < 0.85 else 'red' for m in maes]
    axes[1].bar(x, maes, color=colors_mae, alpha=0.7)
    axes[1].set_title('MAE (越低越好)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='0.70 阈值')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Pearson
    pearsons = [reports[n].get('Pearson', 0) for n in names]
    colors_p = ['green' if p > 0.65 else 'orange' if p > 0.55 else 'red' for p in pearsons]
    axes[2].bar(x, pearsons, color=colors_p, alpha=0.7)
    axes[2].set_title('Pearson Correlation')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[2].axhline(y=0.65, color='r', linestyle='--', alpha=0.5, label='0.65 阈值')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    # 综合分数 (Acc*0.4 + F1*0.4 + (1-MAE)*0.2)
    composite = [
        reports[n].get('Accuracy', 0) * 0.4 +
        reports[n].get('F1_weighted', 0) * 0.4 +
        (1 - reports[n].get('MAE', 0)) * 0.2
        for n in names
    ]
    colors_c = ['gold' if c > max(composite) * 0.98 else 'steelblue' for c in composite]
    axes[3].bar(x, composite, color=colors_c, alpha=0.7)
    axes[3].set_title('Composite Score (0.4*Acc + 0.4*F1 + 0.2*(1-MAE))')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[3].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = '/mnt/data2/home/chenyu/HuaWeiCup/huaweibei-E/outputs/figures/model_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 对比图已保存: {out_path}")

    # 打印最佳模型
    best = names[0]
    print(f"\n🏆 最佳模型: {best}")
    for k in ['Accuracy', 'F1_weighted', 'MAE', 'Pearson']:
        print(f"   {k}: {reports[best][k]:.4f}")


if __name__ == '__main__':
    main()
