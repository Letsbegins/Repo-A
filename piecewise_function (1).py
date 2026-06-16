"""
piecewise_function.py
=====================
在一维空间中构造并绘制分段复合函数。支持自动连续性缝合。

核心设计
--------
- make_*()        模块级工厂函数，返回纯函数，不含任何状态。
- PiecewiseFunction  核心类，负责段管理、求值、缝合、绘图。

连续性缝合逻辑
--------------
stitch() 从第一段的左端点出发，逐连接点计算：
    δ_{i+1} = δ_i + [ f_i(x_c) - f_{i+1}(x_c) ]
其中 x_c 为第 i 段的右端点（= 第 i+1 段的左端点）。
缝合后每段实际求值为 f_i(x) + δ_i，δ_0 = 0。
原始 PiecewiseFunction 对象不受影响；stitch() 返回新实例。

用法示例见文件末尾的 __main__ 部分。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Callable


# ──────────────────────────────────────────────────────────────────────────────
# 1. 基本函数工厂
# ──────────────────────────────────────────────────────────────────────────────

def make_exponential(a: float, b: float, c: float = 0.0) -> Callable:
    """f(x) = a·exp(b·x) + c"""
    def f(x):
        return a * np.exp(b * np.asarray(x, float)) + c
    f.__name__ = f"exp({a}·e^{{{b}x}}+{c})"
    return f


def make_power(a: float, n: float, c: float = 0.0) -> Callable:
    """f(x) = a·|x|^n·sgn(x) + c  （奇数幂保号，偶数幂取绝对值）"""
    def f(x):
        x = np.asarray(x, float)
        return a * np.power(np.abs(x), n) * np.sign(x) + c
    f.__name__ = f"power({a}·x^{n}+{c})"
    return f


def make_sinusoidal(A: float, omega: float, phi: float = 0.0,
                    c: float = 0.0) -> Callable:
    """f(x) = A·sin(ω·x + φ) + c"""
    def f(x):
        return A * np.sin(omega * np.asarray(x, float) + phi) + c
    f.__name__ = f"sin({A}·sin({omega:.3g}x+{phi:.3g})+{c})"
    return f


def make_linear(k: float, b: float = 0.0) -> Callable:
    """f(x) = k·x + b"""
    def f(x):
        return k * np.asarray(x, float) + b
    f.__name__ = f"linear({k}x+{b})"
    return f


def make_gaussian(A: float, mu: float, sigma: float,
                  c: float = 0.0) -> Callable:
    """f(x) = A·exp(-(x-μ)²/(2σ²)) + c"""
    def f(x):
        x = np.asarray(x, float)
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
    f.__name__ = f"gauss({A}·N({mu},{sigma}))"
    return f


def make_polynomial(coeffs: list[float]) -> Callable:
    """f(x) = coeffs[0] + coeffs[1]·x + coeffs[2]·x² + …  （低次到高次）"""
    def f(x):
        x = np.asarray(x, float)
        return sum(c * x ** i for i, c in enumerate(coeffs))
    terms = "+".join(f"{c}x^{i}" for i, c in enumerate(coeffs) if c != 0)
    f.__name__ = f"poly({terms})"
    return f


def make_constant(value: float) -> Callable:
    """f(x) = value"""
    def f(x):
        return np.full_like(np.asarray(x, float), value)
    f.__name__ = f"const({value})"
    return f


# ──────────────────────────────────────────────────────────────────────────────
# 2. 内部辅助：带偏移的包裹函数
# ──────────────────────────────────────────────────────────────────────────────

def _shifted(func: Callable, delta: float) -> Callable:
    """返回 g(x) = func(x) + delta，不修改原函数。"""
    if np.isclose(delta, 0.0):
        return func                          # 零偏移直接复用
    def g(x):
        return func(x) + delta
    g.__name__ = getattr(func, "__name__", "?") + f"  [+{delta:+.4g}]"
    return g


# ──────────────────────────────────────────────────────────────────────────────
# 3. PiecewiseFunction 类
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2563eb", "#dc2626", "#16a34a",
    "#d97706", "#7c3aed", "#0891b2",
    "#be185d", "#065f46",
]


class PiecewiseFunction:
    """
    一维分段复合函数。

    构造
    ----
    直接传入段列表（推荐）::

        pf = PiecewiseFunction([
            (0.0, 2.0, make_exponential(3, -0.8, 1)),
            (2.0, 5.0, make_power(0.5, 2, -1)),
        ])

    或逐段添加::

        pf = PiecewiseFunction()
        pf.add_segment(0.0, 2.0, make_exponential(3, -0.8, 1))
        pf.add_segment(2.0, 5.0, make_power(0.5, 2, -1))

    主要方法
    --------
    __call__(x)         对 NumPy 数组求值
    continuity_check()  返回各连接点的连续性信息
    summary()           打印结构与连续性报告
    stitch()            返回自动缝合后的新 PiecewiseFunction（原对象不变）
    plot(...)           绘图
    compare(other, ...) 将当前实例与另一实例叠加对比绘图
    """

    # ── 构造 ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        segments: list[tuple[float, float, Callable]] | None = None,
        *,
        _stitched: bool = False,          # 内部标记，由 stitch() 设置
    ):
        self.segments: list[tuple[float, float, Callable]] = []
        self._stitched: bool = _stitched
        if segments:
            for seg in segments:
                self.add_segment(*seg)

    def add_segment(self, x_start: float, x_end: float,
                    func: Callable) -> "PiecewiseFunction":
        """添加一段；返回 self 以支持链式调用。"""
        if x_start >= x_end:
            raise ValueError(f"区间无效: x_start={x_start} >= x_end={x_end}")
        self.segments.append((float(x_start), float(x_end), func))
        self.segments.sort(key=lambda s: s[0])
        return self

    # ── 内部校验 ──────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not self.segments:
            raise ValueError("尚未定义任何段。")
        for i in range(len(self.segments) - 1):
            end_i    = self.segments[i][1]
            start_i1 = self.segments[i + 1][0]
            if not np.isclose(end_i, start_i1):
                raise ValueError(
                    f"段 {i+1} 终点 {end_i} ≠ 段 {i+2} 起点 {start_i1}，"
                    "请检查端点设置。"
                )

    # ── 求值 ──────────────────────────────────────────────────────────────────

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """对数组 x 逐段求值；区间约定：左闭右开，最后一段双闭。"""
        self._validate()
        x = np.asarray(x, dtype=float)
        y = np.full_like(x, np.nan)
        n = len(self.segments)
        for idx, (x0, x1, func) in enumerate(self.segments):
            mask = (x >= x0) & (x < x1) if idx < n - 1 else (x >= x0) & (x <= x1)
            if np.any(mask):
                y[mask] = func(x[mask])
        return y

    # ── 连续性检查 ────────────────────────────────────────────────────────────

    def continuity_check(self, tol: float = 1e-9) -> list[dict]:
        """
        返回每个内部连接点的连续性信息字典，字段：
            x, f_left, f_right, jump, continuous
        """
        self._validate()
        results = []
        for i in range(len(self.segments) - 1):
            xc = self.segments[i][1]
            fl = float(np.squeeze(self.segments[i][2](np.array([xc], dtype=float))))
            fr = float(np.squeeze(self.segments[i+1][2](np.array([xc], dtype=float))))
            jump = abs(fr - fl)
            results.append({
                "x":          xc,
                "f_left":     fl,
                "f_right":    fr,
                "jump":       jump,
                "continuous": jump < tol,
            })
        return results

    @property
    def is_continuous(self) -> bool:
        """若所有连接点均连续则返回 True。"""
        return all(c["continuous"] for c in self.continuity_check())

    # ── 自动连续性缝合 ────────────────────────────────────────────────────────

    def stitch(self, anchor: str = "left") -> "PiecewiseFunction":
        """
        自动连续性缝合：返回一个新的 PiecewiseFunction，
        各段函数叠加常数偏移 δ_i，使所有连接点处函数值连续。
        原对象完全不变。

        参数
        ----
        anchor : "left" | "right"
            "left"  — 第一段保持不变，后续段向前对齐（默认）。
            "right" — 最后一段保持不变，前面各段向后对齐。

        返回
        ----
        新的 PiecewiseFunction 实例，标记 _stitched=True。

        数学推导（anchor="left"）
        -------------------------
            δ_0 = 0
            δ_{i+1} = δ_i + [ f_i(x_c) + δ_i ] − [ f_{i+1}(x_c) + δ_{i+1} ] = 0
            ⟹  δ_{i+1} = δ_i + f_i(x_c) − f_{i+1}(x_c)
        其中 x_c = x_{i,right} = x_{i+1,left}。
        """
        self._validate()
        n = len(self.segments)
        deltas = [0.0] * n

        def _eval(func, xc):
            """在单点 xc 处求值，安全返回 Python float。"""
            return float(np.squeeze(func(np.array([xc], dtype=float))))

        if anchor == "left":
            for i in range(n - 1):
                xc = self.segments[i][1]
                fl = _eval(self.segments[i][2],     xc) + deltas[i]
                fr = _eval(self.segments[i + 1][2], xc)
                deltas[i + 1] = fl - fr

        elif anchor == "right":
            for i in range(n - 2, -1, -1):
                xc = self.segments[i][1]
                fl = _eval(self.segments[i][2],     xc)
                fr = _eval(self.segments[i + 1][2], xc) + deltas[i + 1]
                deltas[i] = fr - fl

        else:
            raise ValueError(f"anchor 必须为 'left' 或 'right'，得到 '{anchor}'。")

        new_segments = [
            (x0, x1, _shifted(func, delta))
            for (x0, x1, func), delta in zip(self.segments, deltas)
        ]
        return PiecewiseFunction(new_segments, _stitched=True)

    # ── 报告 ──────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """打印段信息与连接点连续性报告。"""
        self._validate()
        tag = "（已缝合）" if self._stitched else ""
        print("=" * 64)
        print(f"  分段函数  {len(self.segments)} 段  {tag}")
        print("=" * 64)
        for i, (x0, x1, func) in enumerate(self.segments):
            print(f"  段 {i+1}: x ∈ [{x0}, {x1}]")
            print(f"         {getattr(func, '__name__', repr(func))}")
        print("-" * 64)
        checks = self.continuity_check()
        if checks:
            print("  连接点连续性:")
            for c in checks:
                if c["continuous"]:
                    status = "✓ 连续"
                else:
                    status = f"✗ 跳跃量 = {c['jump']:.4e}"
                print(f"    x = {c['x']:.6g}:  "
                      f"左 = {c['f_left']:.6g},  "
                      f"右 = {c['f_right']:.6g}  {status}")
        print("=" * 64)

    # ── 单图绘制 ──────────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        n_points:         int   = 1200,
        title:            str   = "分段复合函数",
        xlabel:           str   = "x",
        ylabel:           str   = "f(x)",
        show_segments:    bool  = True,
        show_breakpoints: bool  = True,
        show_joints:      bool  = True,
        figsize:          tuple = (10, 5),
        save_path:        str | None = None,
        ax:               plt.Axes | None = None,   # 外部传入坐标轴（供 compare 使用）
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        绘制分段函数。

        参数
        ----
        show_segments    各段用不同颜色绘制
        show_breakpoints 用虚线标记断点 x 坐标
        show_joints      在连接点处标记 ○（连续）或 ×（跳跃）
        ax               传入已有 Axes（用于叠加绘图），为 None 时自动创建
        """
        self._validate()
        x_min = self.segments[0][0]
        x_max = self.segments[-1][1]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#ffffff")
            standalone = True
        else:
            fig = ax.get_figure()
            standalone = False

        legend_patches = []
        pts_per_seg = max(50, n_points // len(self.segments))

        for i, (x0, x1, func) in enumerate(self.segments):
            xs = np.linspace(x0, x1, pts_per_seg)
            ys = func(xs) if show_segments else self(xs)
            color = _PALETTE[i % len(_PALETTE)]
            lw = 2.2 if show_segments else 2.0
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
            if show_segments:
                name = getattr(func, "__name__", f"段{i+1}")
                legend_patches.append(
                    mpatches.Patch(color=color, label=f"段{i+1}: {name}")
                )

        if show_breakpoints:
            bps = sorted({x0 for x0, *_ in self.segments} |
                         {x1 for _, x1, _ in self.segments})
            for xb in bps:
                ax.axvline(xb, color="#94a3b8", lw=0.8,
                           ls="--", alpha=0.65, zorder=1)

        if show_joints:
            for c in self.continuity_check():
                xc  = c["x"]
                yc  = (c["f_left"] + c["f_right"]) / 2
                col = "#16a34a" if c["continuous"] else "#e11d48"
                mk  = "o"       if c["continuous"] else "x"
                ax.plot(xc, yc, marker=mk, color=col,
                        ms=8, zorder=5, mew=2.2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, ls="--", alpha=0.45, zorder=0)
        ax.set_xlim(x_min, x_max)

        if legend_patches:
            ax.legend(handles=legend_patches, fontsize=8.5,
                      loc="best", framealpha=0.92)

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"图像已保存至: {save_path}")
            plt.show()

        return fig, ax

    # ── 对比绘图 ──────────────────────────────────────────────────────────────

    def compare(
        self,
        other: "PiecewiseFunction",
        *,
        labels:    tuple[str, str] = ("原始", "缝合后"),
        title:     str             = "缝合前后对比",
        figsize:   tuple           = (13, 5),
        save_path: str | None      = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """
        将 self 与 other 并排对比绘制（共享 y 轴范围）。

        典型用法::

            pf_stitched = pf.stitch()
            pf.compare(pf_stitched)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                       sharey=False)
        fig.patch.set_facecolor("#ffffff")
        for ax in (ax1, ax2):
            ax.set_facecolor("#f8f9fa")

        self.plot( title=labels[0], ax=ax1, show_joints=True)
        other.plot(title=labels[1], ax=ax2, show_joints=True)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图像已保存至: {save_path}")

        plt.show()
        return fig, (ax1, ax2)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        tag = " [stitched]" if self._stitched else ""
        if not self.segments:
            return f"PiecewiseFunction(empty){tag}"
        x0 = self.segments[0][0]
        x1 = self.segments[-1][1]
        return (f"PiecewiseFunction({len(self.segments)} segs, "
                f"x∈[{x0},{x1}]){tag}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. 使用示例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 示例 1: 三段，存在明显跳跃 ────────────────────────────────────────────
    pf = PiecewiseFunction([
        (0.0, 2.0, make_exponential(3.0, -0.8, 1.0)),
        (2.0, 5.0, make_power(0.5, 2.0, -1.0)),
        (5.0, 9.0, make_sinusoidal(2.0, np.pi / 2, 0.0, 3.0)),
    ])

    pf.summary()

    # 缝合（从左锚定）
    pf_s = pf.stitch(anchor="left")
    pf_s.summary()

    # 并排对比
    pf.compare(pf_s, labels=("原始（有跳跃）", "缝合后（连续）"),
               title="示例1: 指数 + 幂函数 + 正弦 — 缝合前后对比")

    # ── 示例 2: 四段，anchor="right" ──────────────────────────────────────────
    pf2 = PiecewiseFunction([
        (-5.0, -2.0, make_linear(1.0, 3.0)),
        (-2.0,  1.0, make_gaussian(4.0, 0.0, 0.8)),
        ( 1.0,  3.5, make_power(-0.3, 3.0, 2.0)),
        ( 3.5,  6.0, make_exponential(0.5, 0.4)),
    ])

    pf2_s = pf2.stitch(anchor="right")   # 最后一段保持不变
    pf2.compare(pf2_s,
                labels=("原始", "缝合后（右锚定）"),
                title="示例2: 线性+高斯+幂函数+指数 — 右锚定缝合")
