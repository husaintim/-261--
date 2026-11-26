"""Статистические функции из семинаров С7 и С8."""

import math
from statistics import NormalDist

import pandas as pd
from scipy import stats


def n_per_group(sigma: float, mde: float, alpha: float = 0.05,
                power: float = 0.8, comparisons: int = 1) -> int:
    """Размер каждой из двух равных групп для заданного эффекта."""
    return math.ceil((mde_for_n(sigma, 1, alpha, power, comparisons) / mde) ** 2)


def mde_for_n(sigma: float, n: int, alpha: float = 0.05,
              power: float = 0.8, comparisons: int = 1) -> float:
    """Минимальный обнаруживаемый эффект при n клиентах в группе."""
    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha / (2 * comparisons))
    power_quantile = normal.inv_cdf(power)
    return (critical + power_quantile) * sigma * math.sqrt(2 / n)


def diff_ci(t: pd.Series, c: pd.Series, alpha: float = 0.05) -> dict:
    """Разность средних test − control и интервал по нормальному приближению."""
    diff = float(t.mean() - c.mean())
    se = math.sqrt(t.var() / len(t) + c.var() / len(c))
    margin = NormalDist().inv_cdf(1 - alpha / 2) * se
    p = float(2 * stats.norm.sf(abs(diff) / se)) if se else float(diff == 0)
    return {"diff": diff, "lo": diff - margin, "hi": diff + margin, "p": p}


def srm_pvalue(sizes: list[int]) -> float:
    """Проверить наблюдаемые размеры групп против равного распределения."""
    return float(stats.chisquare(sizes).pvalue)


def cuped(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float]:
    """Скорректировать метрику по центрированной предэкспериментальной ковариате."""
    variance = x.var()
    theta = float(y.cov(x) / variance) if variance else 0.0
    return y - theta * (x - x.mean()), theta
