from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService
import random, os
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from utils.config_handler import agent_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger


vector_store = VectorStoreService()
rag = RagSummarizeService(vector_store)

USER_ID_POOL = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
DEFAULT_CITY_POOL = ["深圳", "合肥", "杭州"]
DEFAULT_TIMEZONE = "Asia/Shanghai"

_USER_CONTEXT = {
    "user_id": None,
    "user_city": None,
}

external_data = {}


def set_user_context(user_id: str | None = None, city: str | None = None):
    if user_id is not None:
        _USER_CONTEXT["user_id"] = user_id or None
    if city is not None:
        _USER_CONTEXT["user_city"] = city or None


def _get_city_from_ip() -> str | None:
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=2)
        if resp.ok:
            data = resp.json()
            city = data.get("city")
            if city:
                return city
    except Exception as e:
        logger.warning(f"[get_user_location]IP定位失败: {e}")
    return None


def _geocode_city(city: str) -> tuple[float, float] | None:
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=3,
        )
        if geo_resp.ok:
            geo_data = geo_resp.json()
            results = geo_data.get("results") or []
            if results:
                lat = results[0].get("latitude")
                lon = results[0].get("longitude")
                if lat is not None and lon is not None:
                    return float(lat), float(lon)
    except Exception as e:
        logger.warning(f"[geocode]Open-Meteo地理编码失败: {e}")

    try:
        geo_resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "heima-agent-proj/1.0"},
            timeout=3,
        )
        if not geo_resp.ok:
            return None
        geo_data = geo_resp.json() or []
        if not geo_data:
            return None
        lat = geo_data[0].get("lat")
        lon = geo_data[0].get("lon")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception as e:
        logger.warning(f"[geocode]Nominatim地理编码失败: {e}")
        return None


def _get_weather_from_api(city: str) -> str | None:
    try:
        coords = _geocode_city(city)
        if not coords:
            return None
        lat, lon = coords

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=3,
        )
        if not weather_resp.ok:
            return None
        weather_data = weather_resp.json().get("current") or {}
        if not weather_data:
            return None

        temp = weather_data.get("temperature_2m")
        humidity = weather_data.get("relative_humidity_2m")
        rain = weather_data.get("precipitation")
        wind = weather_data.get("wind_speed_10m")
        if temp is None:
            return None

        return (
            f"城市{city}当前气温{temp}摄氏度"
            f"，相对湿度{humidity}%"
            f"，降水量{rain}毫米"
            f"，风速{wind}米/秒"
        )
    except Exception as e:
        logger.warning(f"[get_weather]天气API调用失败: {e}")
        return None

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符形式返回")
def get_weather(city: str) -> str:
    result = _get_weather_from_api(city)
    if result:
        return result

    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时内降雨概率极低"


@tool(description="获取用户所在城市名称，以纯字符形式返回")
def get_user_location() -> str:
    if _USER_CONTEXT["user_city"]:
        return _USER_CONTEXT["user_city"]

    env_city = os.getenv("AGENT_USER_CITY") or os.getenv("USER_CITY")
    if env_city:
        _USER_CONTEXT["user_city"] = env_city
        return env_city

    city = _get_city_from_ip()
    if city:
        _USER_CONTEXT["user_city"] = city
        return city

    fallback = random.choice(DEFAULT_CITY_POOL)
    _USER_CONTEXT["user_city"] = fallback
    return fallback


@tool(description="获取用户ID，以纯字符形式返回")
def get_user_id() -> str:
    if _USER_CONTEXT["user_id"]:
        return _USER_CONTEXT["user_id"]

    env_user_id = os.getenv("AGENT_USER_ID") or os.getenv("USER_ID")
    if env_user_id:
        _USER_CONTEXT["user_id"] = env_user_id
        return env_user_id

    fallback = random.choice(USER_ID_POOL)
    _USER_CONTEXT["user_id"] = fallback
    return fallback


@tool(description="获取当前月份，以纯字符形式返回")
def get_current_month() -> str:
    tz_name = os.getenv("AGENT_TIMEZONE", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).strftime("%Y-%m")


def generate_external_data():
    if not external_data:
        if "external_data_path" not in agent_conf:
            raise KeyError("配置中缺少 external_data_path 字段")

        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }



@tool(description="检索指定用户在指定月份的扫地/扫拖机器人完整使用记录，以纯字符形式返回，如未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warn(f"[fetch_external_data]未能检索到用户:{user_id}在{month}的数据。")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    return "fill_context_for_report已调用"
