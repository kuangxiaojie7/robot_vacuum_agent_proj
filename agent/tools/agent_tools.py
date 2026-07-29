from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService
from utils.config_handler import agent_conf
import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import re
from zoneinfo import ZoneInfo

from utils.request_context import (
    clear_request_context,
    get_request_user_city,
    get_request_user_id,
    set_request_context,
)
from utils.sqlite_store import sqlite_store
import os

'''
urlencode：将字典或类似字典的对象转换为 URL 查询字符串，例如将 {"name": "test", "age": 10} 转换为 name=test&age=10
urlopen：打开一个 URL 并返回一个文件类对象，可以读取 URL 内容
URLError 和 HTTPError：用于捕获和处理 URL 相关的错误，如网络连接失败、HTTP 错误状态码等
'''

vector_store = VectorStoreService()
rag = RagSummarizeService(vector_store)

_IPV4_RE = re.compile(
    r"^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)# ？代表0个或者1个，*代表0个或者多个，+代表1个或者多个

def _is_valid_ipv4(ip: str) -> bool:
    return bool(_IPV4_RE.match(ip or ""))

#从前端设置用户上下文，包括user_id和user_city
def set_user_context(user_id: str | None = None, city: str | None = None):
    set_request_context(user_id=user_id, city=city)


def clear_user_context():
    clear_request_context()

def _get_public_ip() -> str:
    # 可在agent.yml里覆盖
    ip_sources = agent_conf.get("public_ip_sources", [
        "https://ipv4.icanhazip.com",
    ])
    timeout = float(agent_conf.get("public_ip_timeout", 3))
    for source in ip_sources:
        try:
            with urlopen(source, timeout=timeout) as resp:
                ip = resp.read().decode("utf-8").strip()
                if _is_valid_ipv4(ip):
                    return ip
        except Exception:
            continue

    return ""

GAODE_BASE_URL = agent_conf.get("gaode_base_url")
GAODE_TIMEOUT = float(agent_conf.get("gaode_timeout"))

# 调用高德地图的API(包括天气接口，逆地理编码接口)，返回JSON数据
def _gaode_get(path: str, params: dict) -> dict:
    gaode_key = os.getenv("GAODE_API_KEY", "").strip()
    if not gaode_key:
        raise ValueError("未配置 GAODE_API_KEY 环境变量")

    query = dict(params)
    query["key"] = gaode_key
    url = f"{GAODE_BASE_URL}{path}?{urlencode(query)}"

    try:
        with urlopen(url, timeout=GAODE_TIMEOUT) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as e:
        # e.code是HTTP状态码，e.reason是错误信息
        raise RuntimeError(f"高德HTTP错误: {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"高德网络错误: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"高德请求异常: {str(e)}") from e


# 返回城市名称和adcode，adcode是高德地图的唯一标识符
#该函数用于将城市名称转换为adcode，adcode是高德地图的唯一标识符，用于查询天气等信息，以及标准化城市名称格式
def _resolve_city_to_adcode(city: str) -> tuple[str, str]:
    #调用高德地图的逆地理编码接口，返回城市名称和adcode
    geo = _gaode_get("/v3/geocode/geo", {"address": city})
    #1表示成功，geocodes是返回的城市信息列表
    if geo.get("status") != "1" or not geo.get("geocodes"):
        raise RuntimeError(f"城市解析失败: {geo.get('info', 'unknown')}")

    first = geo["geocodes"][0]
    adcode = first.get("adcode")
    if not adcode:
        raise RuntimeError("城市解析成功但未返回adcode")

    resolved_city = first.get("city") or first.get("district") or city
    if isinstance(resolved_city, list):
        resolved_city = "".join(resolved_city)

    return str(resolved_city), str(adcode)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    if not city or not city.strip():
        return "未提供城市名称，无法查询天气"

    try:
        resolved_city, adcode = _resolve_city_to_adcode(city.strip())
        weather = _gaode_get(
            "/v3/weather/weatherInfo",
            {"city": adcode, "extensions": "base"},
        )# 调用高德地图的天气接口，返回城市的天气信息,extensions=base表示返回基础天气信息

        #1表示成功，lives是返回的天气信息列表
        if weather.get("status") != "1" or not weather.get("lives"):
            return f"城市{resolved_city}天气查询失败：{weather.get('info', 'unknown')}"

        # 取[0]是因为返回的是一个列表，列表里只有一个元素，元素是一个字典，字典里包含了天气信息
        live = weather["lives"][0]
        condition = live.get("weather", "未知")
        temperature = live.get("temperature", "未知")
        humidity = live.get("humidity", "未知")
        wind_direction = live.get("winddirection", "未知")
        wind_power = live.get("windpower", "未知")
        report_time = live.get("reporttime", "未知")

        return (
            f"城市{resolved_city}天气为{condition}，气温{temperature}摄氏度，"
            f"空气湿度{humidity}%，{wind_direction}风{wind_power}级，"
            f"数据发布时间{report_time}。"
        )
    except Exception as e:
        logger.error(f"[get_weather]天气查询失败 city={city} err={str(e)}")
        return f"城市{city}天气查询失败，请稍后重试"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    request_city = get_request_user_city()
    if request_city:
        return request_city
    try:
        public_ip = _get_public_ip()
        params = {"ip": public_ip} if public_ip else {}
        # 调用高德地图的IP接口，返回用户的IP信息
        ip_info = _gaode_get("/v3/ip", params)

        if ip_info.get("status") != "1":
            logger.warning(
                f"[get_user_location]高德返回失败 info={ip_info.get('info')} "
                #当public_ip为None或者空字符串时，用'none'代替
                f"infocode={ip_info.get('infocode')} ip={public_ip or 'none'}"
            )
            return "未知城市"

        city = ip_info.get("city", "")
        province = ip_info.get("province", "")

        if isinstance(city, list):
            city = "".join(city)
        if isinstance(province, list):
            province = "".join(province)

        city = str(city).strip()
        province = str(province).strip()

        if city:
            return city
        if province:
            return province

        logger.warning(
            f"[get_user_location]空城市信息 info={ip_info.get('info')} "
            f"infocode={ip_info.get('infocode')} ip={public_ip or 'none'} raw={ip_info}"
        )
        return "未知城市"

    except Exception as e:
        logger.error(f"[get_user_location]定位失败 err={str(e)}")
        return "未知城市"



@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    request_user_id = get_request_user_id()
    if request_user_id:
        return request_user_id
    return "未提供用户ID"


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    record = sqlite_store.get_usage_record(user_id, month)
    if not record:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


# if __name__ == '__main__':
#     print(get_weather(get_user_location()))
