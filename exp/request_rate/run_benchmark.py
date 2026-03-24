#!/usr/bin/env python3
"""
一键执行 KV Cache 压缩方法 Benchmark 脚本

用法:
    python run_benchmark.py --num-contexts 20 --avg-prompt-tokens 4k
    python run_benchmark.py --num-contexts 50 --avg-prompt-tokens 8k --compress-methods "OURS,KIVI_2BIT"
    python run_benchmark.py --num-contexts 100 --avg-prompt-tokens 16k --request-rates "1,2,4,8"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import signal
import urllib.error
import urllib.request

# 默认配置
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_PORT = 12345
DEFAULT_DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"
DEFAULT_COMPRESS_METHODS = ["KIVI_2BIT", "SVDQ", "OURS", "NONE"]
DEFAULT_REQUEST_RATES = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16]
DEFAULT_NUM_REQUESTS = 15
DEFAULT_WARMUP_DELAY = 10.0

SCRIPT_DIR = Path(__file__).resolve().parent
BLEND_DIR = SCRIPT_DIR.parent


def parse_token_spec(spec: str) -> int:
    """解析 token 数量规格，如 '4k', '8k', '16000'"""
    spec = spec.strip().lower()
    if spec.endswith('k'):
        return int(float(spec[:-1]) * 1000)
    elif spec.endswith('m'):
        return int(float(spec[:-1]) * 1000000)
    else:
        return int(spec)


def parse_args():
    parser = argparse.ArgumentParser(
        description="一键执行 KV Cache 压缩方法 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法: 20个context, 平均4k tokens
  python run_benchmark.py --num-contexts 20 --avg-prompt-tokens 4k

  # 指定压缩方法
  python run_benchmark.py --num-contexts 50 --avg-prompt-tokens 8k --compress-methods "OURS,KIVI_2BIT,NONE"

  # 指定请求率
  python run_benchmark.py --num-contexts 30 --avg-prompt-tokens 6k --request-rates "1,2,4,8"
  # 完整配置
  python run_benchmark.py \\
      --num-contexts 100 \\
      --avg-prompt-tokens 16k \\
      --compress-methods "OURS,KIVI_2BIT,SVDQ,NONE" \\
      --request-rates "1,2,4,8,16" \\
      --num-requests 20 \\
      --output-dir ./results
        """
    )

    # 核心参数
    parser.add_argument(
        "--num-contexts", "-c",
        type=int,
        default=20,
        help="缓存的唯一 context 数量 (default: 20)"
    )
    parser.add_argument(
        "--avg-prompt-tokens", "-t",
        type=str,
        default="4k",
        help="平均 prompt token 数量，支持 k/m 后缀 (default: 4k)"
    )

    # 压缩方法和请求率
    parser.add_argument(
        "--compress-methods", "-m",
        type=str,
        default=",".join(DEFAULT_COMPRESS_METHODS),
        help=f"压缩方法，逗号分隔 (default: {','.join(DEFAULT_COMPRESS_METHODS)})"
    )
    parser.add_argument(
        "--request-rates", "-r",
        type=str,
        default=",".join(map(str, DEFAULT_REQUEST_RATES)),
        help=f"请求率 (RPS)，逗号分隔 (default: {','.join(map(str, DEFAULT_REQUEST_RATES))})"
    )

    # 其他参数
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"每个请求率下的请求数量 (default: {DEFAULT_NUM_REQUESTS})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型名称 (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"vLLM 服务端口 (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="数据集路径"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(SCRIPT_DIR),
        help="输出目录 (default: 当前目录)"
    )
    parser.add_argument(
        "--warmup-delay",
        type=float,
        default=DEFAULT_WARMUP_DELAY,
        help=f"预热后等待时间 (default: {DEFAULT_WARMUP_DELAY}s)"
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="跳过服务器启动，假设服务已运行"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU 显存利用率 (default: 0.6)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32000,
        help="最大模型长度 (default: 32000)"
    )

    return parser.parse_args()


def set_compress_type(compress_type: str):
    """通过环境变量设置压缩类型"""
    os.environ["LMCACHE_COMPRESS_TYPE"] = compress_type
    print(f"[Config] LMCACHE_COMPRESS_TYPE = {compress_type}")


def start_vllm_server(args, compress_type: str) -> subprocess.Popen:
    """启动 vLLM 服务器"""
    env = os.environ.copy()
    vllm_exe = Path(sys.executable).with_name("vllm")

    # LMCache 配置
    env.update({
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": " # # ",
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_BLEND_CHECK_LAYERS": "1",
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
        "LMCACHE_BLEND_MIN_TOKENS": "64",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
        "LMCACHE_COMPRESS_TYPE": compress_type,
        "HF_HUB_OFFLINE": "1",
    })

    kv_config = json.dumps({
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both"
    })

    cmd = [
        str(vllm_exe), "serve", args.model,
        "--kv-transfer-config", kv_config,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "-tp", "1"
    ]

    print(f"[Server] 启动 vLLM 服务器 (compress_type={compress_type})...")
    print(f"[Server] 命令: {' '.join(cmd)}")

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vllm_server_{compress_type}.log"
    log_handle = open(log_file, "ab")
    print(f"[Server] 日志输出: {log_file}")

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(BLEND_DIR),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
    except Exception:
        log_handle.close()
        raise

    proc.log_file = log_handle

    return proc


def wait_for_server(port: int, proc: Optional[subprocess.Popen] = None, timeout: int = 300) -> bool:
    """等待服务器就绪"""
    print(f"[Server] 等待服务器在端口 {port} 就绪...")
    health_url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        if proc is not None and proc.poll() is not None:
            print(f"[Server] vLLM 进程已提前退出，return code={proc.returncode}")
            return False

        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    time.sleep(3)
                    print(f"[Server] 服务器已就绪 (耗时 {time.time() - start_time:.1f}s)")
                    return True
        except urllib.error.HTTPError as http_err:
            print(f"[Server] /health 返回 {http_err.code}, 继续等待...")
        except urllib.error.URLError:
            pass

        time.sleep(2)

    print(f"[Server] 服务器启动超时 ({timeout}s)")
    return False


def stop_server(proc: subprocess.Popen):
    """停止服务器"""
    if proc:
        if proc.poll() is None:
            print("[Server] 停止服务器...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            print("[Server] 服务器已停止")

        log_handle = getattr(proc, "log_file", None)
        if log_handle and not log_handle.closed:
            log_handle.close()

    kill_existing_vllm()


def kill_existing_vllm():
    """杀死现有的 vLLM 进程"""
    print("[Server] 清理现有 vLLM 相关进程...")
    targets = [
        ["pkill", "-9", "-f", "vllm serve"],
        ["pkill", "-9", "EngineCore"],
        ["pkill", "-9", "APIServer"],
    ]
    for cmd in targets:
        subprocess.run(cmd, capture_output=True)
    time.sleep(3)


def run_benchmark(args, compress_type: str, output_json: str) -> bool:
    """运行单个压缩方法的 benchmark"""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "benchmark_reuse.py"),
        "--model", args.model,
        "--api-base", f"http://localhost:{args.port}",
        "--dataset", args.dataset,
        "--num-contexts", str(args.num_contexts),
        "--num-requests", str(args.num_requests),
        "--request-rates", args.request_rates,
        "--warmup-delay", str(args.warmup_delay),
        "--output-json", output_json,
    ]

    print(f"\n[Benchmark] 运行 {compress_type} benchmark...")
    print(f"[Benchmark] 命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"

    result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))

    return result.returncode == 0


def collect_results(output_dir: Path, compress_methods: List[str]) -> Dict:
    """收集所有结果"""
    all_results = {}

    for method in compress_methods:
        json_file = output_dir / f"benchmark_{method}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                all_results[method] = json.load(f)

    return all_results


def generate_csv(all_results: Dict, output_csv: Path):
    """生成 CSV 汇总文件"""
    rows = []
    header = "compress_method,rate_rps,ttft_avg_ms,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,latency_avg_ms,itl_avg_ms,success_rate,request_count"

    for method, data in all_results.items():
        results = data.get("results", {})
        for rate_str, rate_data in results.items():
            summary = rate_data.get("summary", {})
            ttft = summary.get("ttft", {})
            latency = summary.get("latency", {})
            itl = summary.get("itl", {})

            # 转换为 ms
            ttft_avg = (ttft.get("avg") or 0) * 1000
            ttft_p50 = (ttft.get("p50") or 0) * 1000
            ttft_p90 = (ttft.get("p90") or 0) * 1000
            ttft_p99 = (ttft.get("p99") or 0) * 1000
            latency_avg = (latency.get("avg") or 0) * 1000
            itl_avg = (itl.get("avg") or 0) * 1000
            success_rate = summary.get("success_rate", 0)
            request_count = summary.get("request_count", 0)

            rows.append(f"{method},{rate_str},{ttft_avg:.2f},{ttft_p50:.2f},{ttft_p90:.2f},{ttft_p99:.2f},{latency_avg:.2f},{itl_avg:.2f},{success_rate},{request_count}")

    with open(output_csv, 'w') as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"[Output] CSV 保存到: {output_csv}")


def generate_plot(all_results: Dict, output_dir: Path, args):
    """生成图表"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib 未安装，跳过绘图")
        return

    # 定义方法和颜色
    colors = {'KIVI_2BIT': '#4CAF50', 'SVDQ': '#F44336', 'OURS': '#2196F3', 'NONE': '#9E9E9E'}
    markers = {'KIVI_2BIT': 'o', 'SVDQ': 's', 'OURS': '^', 'NONE': 'x'}

    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, data in all_results.items():
        results = data.get("results", {})
        rates = []
        ttfts = []

        for rate_str, rate_data in sorted(results.items(), key=lambda x: float(x[0]) if x[0] != 'inf' else float('inf')):
            rate = float(rate_str) if rate_str != 'inf' else float('inf')
            if rate == float('inf'):
                continue  # 跳过 inf

            summary = rate_data.get("summary", {})
            ttft_avg = summary.get("ttft", {}).get("avg")

            if ttft_avg is not None:
                rates.append(rate)
                ttfts.append(ttft_avg)  # 已经是秒

        if rates and ttfts:
            color = colors.get(method, '#000000')
            marker = markers.get(method, 'o')
            ax.plot(rates, ttfts, marker=marker, color=color, label=method, linewidth=2, markersize=8)

    ax.set_xlabel('Request Rate (RPS)', fontsize=12)
    ax.set_ylabel('TTFT (s)', fontsize=12)
    ax.set_title(f'Time To First Token vs Request Rate\n(contexts={args.num_contexts}, avg_tokens={args.avg_prompt_tokens})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 设置 x 轴刻度
    all_rates = set()
    for data in all_results.values():
        for rate_str in data.get("results", {}).keys():
            if rate_str != 'inf':
                all_rates.add(float(rate_str))
    if all_rates:
        ax.set_xticks(sorted(all_rates))

    plt.tight_layout()

    # 保存图表
    linear_plot = output_dir / "ttft_vs_request_rate.png"
    plt.savefig(linear_plot, dpi=150, bbox_inches='tight')
    print(f"[Output] 线性图保存到: {linear_plot}")

    # Log-log 版本
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for method, data in all_results.items():
        results = data.get("results", {})
        rates = []
        ttfts = []

        for rate_str, rate_data in sorted(results.items(), key=lambda x: float(x[0]) if x[0] != 'inf' else float('inf')):
            rate = float(rate_str) if rate_str != 'inf' else float('inf')
            if rate == float('inf'):
                continue

            summary = rate_data.get("summary", {})
            ttft_avg = summary.get("ttft", {}).get("avg")

            if ttft_avg is not None:
                rates.append(rate)
                ttfts.append(ttft_avg)

        if rates and ttfts:
            color = colors.get(method, '#000000')
            marker = markers.get(method, 'o')
            ax2.loglog(rates, ttfts, marker=marker, color=color, label=method, linewidth=2, markersize=8)

    ax2.set_xlabel('Request Rate (RPS)', fontsize=12)
    ax2.set_ylabel('TTFT (s) - Log Scale', fontsize=12)
    ax2.set_title(f'Time To First Token vs Request Rate (Log-Log Scale)\n(contexts={args.num_contexts}, avg_tokens={args.avg_prompt_tokens})', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    loglog_plot = output_dir / "ttft_vs_request_rate_loglog.png"
    plt.savefig(loglog_plot, dpi=150, bbox_inches='tight')
    print(f"[Output] 对数图保存到: {loglog_plot}")

    plt.close('all')


def print_summary(all_results: Dict, args):
    """打印汇总表格"""
    print("\n" + "=" * 80)
    print("TTFT 汇总 (单位: 秒)")
    print("=" * 80)

    # 收集所有请求率
    all_rates = set()
    for data in all_results.values():
        for rate_str in data.get("results", {}).keys():
            if rate_str != 'inf':
                all_rates.add(float(rate_str))
    rates = sorted(all_rates)

    # 打印表头
    header = f"{'方法':<12}"
    for rate in rates:
        header += f"{rate} RPS".center(12)
    print(header)
    print("-" * len(header))

    # 打印数据
    for method, data in all_results.items():
        results = data.get("results", {})
        row = f"{method:<12}"
        for rate in rates:
            rate_str = str(int(rate)) if rate == int(rate) else str(rate)
            rate_data = results.get(rate_str, {})
            summary = rate_data.get("summary", {})
            ttft_avg = summary.get("ttft", {}).get("avg")

            if ttft_avg is not None:
                row += f"{ttft_avg:.3f}".center(12)
            else:
                row += "N/A".center(12)
        print(row)

    print("-" * len(header))
    print(f"\n配置: contexts={args.num_contexts}, avg_tokens={args.avg_prompt_tokens}, requests={args.num_requests}")


def main():
    args = parse_args()

    # 解析参数
    compress_methods = [m.strip() for m in args.compress_methods.split(",") if m.strip()]
    avg_tokens = parse_token_spec(args.avg_prompt_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KV Cache 压缩方法 Benchmark")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"Context 数量: {args.num_contexts}")
    print(f"平均 Prompt Tokens: {avg_tokens} ({args.avg_prompt_tokens})")
    print(f"压缩方法: {compress_methods}")
    print(f"请求率: {args.request_rates}")
    print(f"每个请求率的请求数: {args.num_requests}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    server_proc = None

    try:
        for method in compress_methods:
            print(f"\n{'='*80}")
            print(f"测试压缩方法: {method}")
            print("=" * 80)

            if not args.skip_server:
                # 清理现有进程
                kill_existing_vllm()

                # 启动服务器
                server_proc = start_vllm_server(args, method)

                # 等待服务器就绪
                if not wait_for_server(args.port, server_proc):
                    print(f"[Error] 服务器启动失败，跳过 {method}")
                    stop_server(server_proc)
                    continue

            # 运行 benchmark
            output_json = str(output_dir / f"benchmark_{method}.json")
            success = run_benchmark(args, method, output_json)

            if not success:
                print(f"[Error] {method} benchmark 失败")

            if not args.skip_server:
                # 停止服务器
                stop_server(server_proc)
                server_proc = None

        # 收集结果
        print("\n" + "=" * 80)
        print("生成汇总报告...")
        print("=" * 80)

        all_results = collect_results(output_dir, compress_methods)

        if all_results:
            # 生成 CSV
            csv_file = output_dir / "benchmark_all_results.csv"
            generate_csv(all_results, csv_file)

            # 生成图表
            generate_plot(all_results, output_dir, args)

            # 打印汇总
            print_summary(all_results, args)
        else:
            print("[Warning] 没有收集到任何结果")

        print("\n[完成] Benchmark 结束")

    except KeyboardInterrupt:
        print("\n[中断] 用户中断")
    finally:
        if server_proc:
            stop_server(server_proc)


if __name__ == "__main__":
    main()
