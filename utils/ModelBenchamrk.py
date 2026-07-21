from datetime import datetime
import torch
import time
import subprocess
import numpy as np
import os
import csv
import torch.nn as nn


class ModelBenchmark:
    """获取模型实际推理性能参数（简化可靠版）"""

    def __init__(
            self,
            model,
            model_path,
            device,
            img_size=640,
            batch=1,
            half=True
    ):
        self.model_name = model_path
        self.device = torch.device(device)
        self.img_size = img_size
        self.batch = batch
        self.half = half and self.device.type == "cuda"

        self.model = model.model.to(self.device).eval()

        if self.half:
            self.model = self.model.half()

        self.dummy = torch.randn(batch, 3, img_size, img_size).to(self.device)
        if self.half:
            self.dummy = self.dummy.half()

    # ==============================
    # GPU信息
    # ==============================
    @staticmethod
    def get_gpu_info():
        """获取GPU信息，增加错误处理"""
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits",
                shell=True,
                timeout=2
            )
            values = result.decode().strip().split(",")
            power = float(values[0].strip()) if values[0].strip() else 0.0
            memory = float(values[1].strip()) if values[1].strip() else 0.0
            util = float(values[2].strip()) if values[2].strip() else 0.0
            return power, memory, util
        except Exception as e:
            return 0.0, 0.0, 0.0

    # ==============================
    # FLOPs
    # ==============================
    def compute_flops(self):
        """计算FLOPs"""
        total_ops = 0

        def hook_fn(module, input, output):
            nonlocal total_ops
            if isinstance(module, nn.Conv2d):
                n, c, h, w = output.shape
                k_h, k_w = module.kernel_size
                # 卷积操作数
                ops = n * c * h * w * (module.in_channels // module.groups) * k_h * k_w
                total_ops += ops * 2  # 乘加算两次操作

        hooks = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(self.dummy)

        for h in hooks:
            h.remove()

        return total_ops / 1e9

    # ==============================
    # Params
    # ==============================
    def compute_params(self):
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    # ==============================
    # 单次推理时间测量（高精度）
    # ==============================
    def measure_single_latency(self):
        """单次测量，使用CUDA事件获得更高精度"""
        if self.device.type == "cuda":
            # 使用CUDA事件，精度更高
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                self.model(self.dummy)
            end_event.record()
            torch.cuda.synchronize()

            return start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        else:
            start = time.perf_counter()
            with torch.no_grad():
                self.model(self.dummy)
            return time.perf_counter() - start

    # ==============================
    # Benchmark 主函数
    # ==============================
    def run(self, warmup=50, iters=200, save_csv=True):
        """运行基准测试"""

        print(f"\n{'=' * 50}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Batch: {self.batch}, Size: {self.img_size}")
        print(f"{'=' * 50}\n")

        # 计算模型参数
        params = self.compute_params()
        flops = self.compute_flops()
        print(f"Params: {params:.2f} M")
        print(f"FLOPs: {flops:.2f} G")

        # 预热
        print(f"Warming up ({warmup} iters)...")
        for _ in range(warmup):
            with torch.no_grad():
                self.model(self.dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 正式测量
        print(f"Measuring ({iters} iters)...")
        latencies = []
        powers = []
        memories = []
        utils = []

        for i in range(iters):
            # 测量延迟
            latency = self.measure_single_latency()
            latencies.append(latency)

            # 每10次采样一次GPU信息（避免影响性能）
            if i % 10 == 0:
                power, memory, util = self.get_gpu_info()
                powers.append(power)
                memories.append(memory)
                utils.append(util)

        # 统计分析
        latencies = np.array(latencies)

        # 去除明显的异常值（3-sigma）
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        filtered_lat = latencies[np.abs(latencies - mean_lat) <= 3 * std_lat]

        latency = np.mean(filtered_lat)
        latency_std = np.std(filtered_lat)
        fps = self.batch / latency
        throughput = self.batch * fps

        # GPU统计
        power = np.mean(powers) if powers else 0
        memory = np.mean(memories) if memories else 0
        util = np.mean(utils) if utils else 0

        # 能耗（注意：nvidia-smi的功耗是平均值，这里只能估算）
        energy = power * latency if power > 0 else 0

        # 输出结果
        print(f"\n{'=' * 50}")
        print("Results:")
        print(f"{'=' * 50}")
        print(f"Latency: {latency * 1000:.2f} ± {latency_std * 1000:.2f} ms")
        print(f"CV (变异系数): {(latency_std / latency) * 100:.1f}%")  # 越小越稳定
        print(f"FPS: {fps:.2f}")
        print(f"Throughput: {throughput:.2f} img/s")

        if self.device.type == "cuda":
            print(f"\nGPU Memory: {memory:.2f} MB")
            print(f"GPU Util: {util:.2f} %")
            print(f"Power: {power:.2f} W")
            print(f"Energy (est.): {energy * 1000:.3f} mJ")

        # 稳定性评估
        if latency_std / latency < 0.05:
            print("\n✓ Stability: Good (CV < 5%)")
        elif latency_std / latency < 0.10:
            print("\n⚠ Stability: Acceptable (CV < 10%)")
        else:
            print("\n✗ Stability: Poor (CV > 10%)")

        # 保存CSV
        if save_csv:
            self.save_csv(params, flops, latency, latency_std, fps,
                          throughput, memory, util, power, energy)

        return {
            'latency': latency,
            'latency_std': latency_std,
            'fps': fps,
            'throughput': throughput,
            'power': power,
            'memory': memory,
            'util': util,
            'energy': energy
        }

    # ==============================
    # CSV保存
    # ==============================
    def save_csv(self, params, flops, latency, latency_std, fps,
                 throughput, memory, util, power, energy):

        csv_file = "benchmark_results.csv"

        header = [
            "Timestamp", "Model", "Device", "Batch", "Size",
            "Params(M)", "FLOPs(G)",
            "Latency(ms)", "Latency_Std(ms)", "CV(%)",
            "FPS", "Throughput",
            "Memory(MB)", "GPU_Util(%)", "Power(W)", "Energy(mJ)"
        ]

        cv = (latency_std / latency) * 100 if latency > 0 else 0

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.model_name,
            str(self.device),
            self.batch,
            self.img_size,
            round(params, 2),
            round(flops, 2),
            round(latency * 1000, 2),
            round(latency_std * 1000, 3),
            round(cv, 1),
            round(fps, 2),
            round(throughput, 2),
            round(memory, 2),
            round(util, 2),
            round(power, 2),
            round(energy * 1000, 3)
        ]

        write_header = not os.path.exists(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        print(f"\n✓ Results saved to {csv_file}")