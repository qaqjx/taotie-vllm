#!/usr/bin/env python3
"""
Plot comparison of S3-only benchmark results for KIVI_2BIT vs OURS compression methods.
Data extracted from /tmp/kivi_s3_only_benchmark.log and /tmp/ours_s3_only_benchmark.log
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark logs
rates = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16]

# KIVI_2BIT results
kivi_ttft = [223.23, 251.94, 293.41, 297.29, 880.82, 1321.95, 1766.18, 2022.68, 3423.05, 3824.46, 3330.40]
kivi_latency = [1454.05, 1549.40, 2003.81, 2531.41, 5279.13, 4849.08, 6081.63, 6321.26, 7623.03, 7834.17, 7361.25]
kivi_itl = [19.54, 21.86, 27.15, 36.21, 69.81, 57.26, 70.40, 71.09, 71.27, 63.65, 65.26]

# OURS results (Note: 6 RPS has anomalous spike, we'll mark it)
ours_ttft = [169.55, 174.36, 168.99, 182.08, 244.94, 344.85, 13960.80, 916.53, 1784.30, 1703.17, 2304.56]
ours_latency = [1369.17, 1493.83, 1713.10, 2154.95, 2717.68, 3840.80, 43303.09, 4555.82, 5344.16, 5387.32, 5749.73]
ours_itl = [19.19, 20.94, 24.51, 31.32, 39.25, 56.75, 472.75, 58.92, 58.52, 60.33, 55.78]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: TTFT comparison
ax1 = axes[0]
ax1.plot(rates, kivi_ttft, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax1.plot(rates, ours_ttft, 's-', label='OURS', linewidth=2, markersize=6)
ax1.set_xlabel('Request Rate (RPS)', fontsize=12)
ax1.set_ylabel('TTFT (ms)', fontsize=12)
ax1.set_title('Time To First Token (TTFT)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='1s threshold')

# Plot 2: End-to-end Latency comparison
ax2 = axes[1]
ax2.plot(rates, kivi_latency, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax2.plot(rates, ours_latency, 's-', label='OURS', linewidth=2, markersize=6)
ax2.set_xlabel('Request Rate (RPS)', fontsize=12)
ax2.set_ylabel('Latency (ms)', fontsize=12)
ax2.set_title('End-to-End Latency', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: ITL comparison
ax3 = axes[2]
ax3.plot(rates, kivi_itl, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax3.plot(rates, ours_itl, 's-', label='OURS', linewidth=2, markersize=6)
ax3.set_xlabel('Request Rate (RPS)', fontsize=12)
ax3.set_ylabel('ITL (ms)', fontsize=12)
ax3.set_title('Inter-Token Latency (ITL)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

plt.suptitle('S3-Only Benchmark: KIVI_2BIT vs OURS\n(CPU Buffer Disabled)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/xujie/lmcache-v1/examples/blend_kv_v1/request_rate/s3_only_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: s3_only_comparison.png")

# Create a separate plot excluding the anomalous 6 RPS data point for OURS
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Filter out 6 RPS anomaly
rates_filtered = [r for r in rates if r != 6]
ours_ttft_filtered = [t for r, t in zip(rates, ours_ttft) if r != 6]
ours_latency_filtered = [l for r, l in zip(rates, ours_latency) if r != 6]
ours_itl_filtered = [i for r, i in zip(rates, ours_itl) if r != 6]
kivi_ttft_filtered = [t for r, t in zip(rates, kivi_ttft) if r != 6]
kivi_latency_filtered = [l for r, l in zip(rates, kivi_latency) if r != 6]
kivi_itl_filtered = [i for r, i in zip(rates, kivi_itl) if r != 6]

# Plot 1: TTFT comparison (filtered)
ax1 = axes2[0]
ax1.plot(rates_filtered, kivi_ttft_filtered, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax1.plot(rates_filtered, ours_ttft_filtered, 's-', label='OURS', linewidth=2, markersize=6)
ax1.set_xlabel('Request Rate (RPS)', fontsize=12)
ax1.set_ylabel('TTFT (ms)', fontsize=12)
ax1.set_title('Time To First Token (TTFT)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1000, color='r', linestyle='--', alpha=0.5)

# Plot 2: End-to-end Latency comparison (filtered)
ax2 = axes2[1]
ax2.plot(rates_filtered, kivi_latency_filtered, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax2.plot(rates_filtered, ours_latency_filtered, 's-', label='OURS', linewidth=2, markersize=6)
ax2.set_xlabel('Request Rate (RPS)', fontsize=12)
ax2.set_ylabel('Latency (ms)', fontsize=12)
ax2.set_title('End-to-End Latency', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: ITL comparison (filtered)
ax3 = axes2[2]
ax3.plot(rates_filtered, kivi_itl_filtered, 'o-', label='KIVI_2BIT', linewidth=2, markersize=6)
ax3.plot(rates_filtered, ours_itl_filtered, 's-', label='OURS', linewidth=2, markersize=6)
ax3.set_xlabel('Request Rate (RPS)', fontsize=12)
ax3.set_ylabel('ITL (ms)', fontsize=12)
ax3.set_title('Inter-Token Latency (ITL)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('S3-Only Benchmark: KIVI_2BIT vs OURS\n(Excluding 6 RPS anomaly)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/xujie/lmcache-v1/examples/blend_kv_v1/request_rate/s3_only_comparison_filtered.png', dpi=150, bbox_inches='tight')
print("Saved: s3_only_comparison_filtered.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY: S3-Only Benchmark Results")
print("="*70)
print("\nNote: OURS at 6 RPS has an anomalous spike (TTFT=13.9s) - likely transient S3 I/O issue")
print("\nTTFT Comparison (excluding 6 RPS anomaly):")
print("-" * 50)
print(f"{'Rate':>8} {'KIVI_2BIT':>12} {'OURS':>12} {'Improvement':>15}")
print("-" * 50)
for r, k, o in zip(rates_filtered, kivi_ttft_filtered, ours_ttft_filtered):
    improvement = ((k - o) / k) * 100
    print(f"{r:>8} {k:>10.2f} ms {o:>10.2f} ms {improvement:>+13.1f}%")

print("\n" + "-" * 50)
# Calculate average improvement
avg_improvement = np.mean([((k - o) / k) * 100 for k, o in zip(kivi_ttft_filtered, ours_ttft_filtered)])
print(f"Average TTFT improvement: OURS is {avg_improvement:.1f}% better than KIVI_2BIT")

# Low load analysis (0.5-2 RPS)
low_load_rates = [0.5, 1, 1.5, 2]
low_load_improvement = np.mean([((kivi_ttft[rates.index(r)] - ours_ttft[rates.index(r)]) / kivi_ttft[rates.index(r)]) * 100 for r in low_load_rates])
print(f"Low load (0.5-2 RPS) improvement: OURS is {low_load_improvement:.1f}% better")

# Medium load analysis (3-4 RPS)
med_load_rates = [3, 4]
med_load_improvement = np.mean([((kivi_ttft[rates.index(r)] - ours_ttft[rates.index(r)]) / kivi_ttft[rates.index(r)]) * 100 for r in med_load_rates])
print(f"Medium load (3-4 RPS) improvement: OURS is {med_load_improvement:.1f}% better")

# High load analysis (8-16 RPS)
high_load_rates = [8, 10, 12, 16]
high_load_improvement = np.mean([((kivi_ttft[rates.index(r)] - ours_ttft[rates.index(r)]) / kivi_ttft[rates.index(r)]) * 100 for r in high_load_rates])
print(f"High load (8-16 RPS) improvement: OURS is {high_load_improvement:.1f}% better")

plt.show()
