import unittest
from unittest.mock import patch

from agent.tools.agent_tools import get_weather


class WeatherFallbackTests(unittest.TestCase):
    @patch("agent.tools.agent_tools._resolve_city_to_adcode", side_effect=RuntimeError("mock gaode failure"))
    def test_weather_returns_user_friendly_fallback_when_gaode_fails(self, _mock_resolve):
        answer = get_weather.invoke({"city": "合肥"})
        self.assertEqual(answer, "城市合肥天气查询失败，请稍后重试")

