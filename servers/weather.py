"""家庭天气的 HTTP 与格式化逻辑；不依赖 MCP，不加载 .env。"""

import math
import os
from threading import RLock
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx2

Day = Literal["today", "tomorrow"]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT = "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
DAILY = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
_LOCATION_CACHE: dict[tuple[str, str], tuple[float, float, str]] = {}
_LOCATION_LOCK = RLock()


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("天气数据中的数值缺失或无效")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("天气数据中的数值无效")
    return value


def home_location(client: httpx2.Client) -> tuple[float, float, str]:
    root = os.getenv("HA_URL", "").strip().rstrip("/")
    token = os.getenv("HA_TOKEN", "").strip()
    if not root or not token:
        raise ValueError("请配置 HA_URL 和 HA_TOKEN")
    key = (root, token)
    with _LOCATION_LOCK:
        if key in _LOCATION_CACHE:
            return _LOCATION_CACHE[key]
        response = client.get(root + "/api/config", headers={"Authorization": f"Bearer {token}"}, timeout=5.0)
        response.raise_for_status()
        config = response.json()
        if not isinstance(config, dict):
            raise ValueError("HA 配置不是对象")
        lat, lon = _number(config.get("latitude")), _number(config.get("longitude"))
        if not -90 <= lat <= 90 or not -180 <= lon <= 180 or (lat, lon) == (0, 0):
            raise ValueError("家的经纬度未配置或无效")
        tz = config.get("time_zone")
        if not isinstance(tz, str) or not tz.strip():
            raise ValueError("HA 缺少 time_zone")
        ZoneInfo(tz)  # 不用 auto 或 UTC 默默替代无效的家庭时区。
        _LOCATION_CACHE.clear()  # 只保留当前配置，重启进程后重新读取。
        _LOCATION_CACHE[key] = (lat, lon, tz)
        return lat, lon, tz


def fetch_forecast(client: httpx2.Client, lat: float, lon: float, tz: str) -> dict[str, Any]:
    # Authorization 只在 HA 那一次请求上设置，绝不发给 Open-Meteo。
    response = client.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": 2,
        "current": CURRENT, "daily": DAILY,
    }, timeout=5.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("天气服务返回的数据无效")
    return data


def describe(code: int) -> str:
    if code == 0:
        return "晴"
    for codes, text in ((range(1, 4), "多云"), ((45, 48), "雾"),
                        (range(51, 58), "毛毛雨"), (range(61, 68), "雨"),
                        (range(71, 78), "雪"), (range(80, 83), "阵雨"),
                        ((85, 86), "阵雪"), (range(95, 100), "雷雨")):
        if code in codes:
            return text
    return "天气情况未知"


def forecast_details(data: dict[str, Any], day: Day) -> dict[str, Any]:
    """只选取用于回答的天气字段，不返回 Open-Meteo 回显的坐标。"""
    if day not in ("today", "tomorrow"):
        raise ValueError("day 只能是 today 或 tomorrow")
    index = 0 if day == "today" else 1
    daily = data.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("天气服务缺少 daily 预报")
    selected: dict[str, Any] = {}
    for key in DAILY.split(","):
        values = daily.get(key)
        if not isinstance(values, list) or len(values) <= index:
            raise ValueError("天气服务缺少所请求日期的预报")
        value = values[index]
        # null 表示没有降雨概率数据，不能把它当作零降雨。
        selected[key] = None if key == "precipitation_probability_max" and value is None else _number(value)
    if selected["temperature_2m_min"] > selected["temperature_2m_max"]:
        raise ValueError("天气服务返回的温度范围无效")
    probability = selected["precipitation_probability_max"]
    if probability is not None and not 0 <= probability <= 100:
        raise ValueError("天气服务返回的降雨概率无效")
    result: dict[str, Any] = {"day": day, "source": "Open-Meteo", "daily": selected}
    if day == "today":
        current = data.get("current")
        if not isinstance(current, dict):
            raise ValueError("天气服务缺少 current 数据")
        result["current"] = {key: _number(current.get(key)) for key in CURRENT.split(",")}
    return result


def summarize(data: dict[str, Any], day: Day) -> str:
    details = forecast_details(data, day)
    daily = details["daily"]
    probability = daily["precipitation_probability_max"]
    rain = f"降雨概率 {probability:g}%" if probability is not None else "降雨概率暂无数据"
    text = (f"{'今天' if day == 'today' else '明天'}{describe(daily['weather_code'])}，"
            f"{daily['temperature_2m_min']:g} 到 {daily['temperature_2m_max']:g} 度，{rain}")
    if day == "today":
        current = details["current"]
        text += f"，现在 {current['temperature_2m']:g} 度，体感 {current['apparent_temperature']:g} 度"
    return text
