"""假 HA / Open-Meteo 响应；不读取真实配置、不发真实请求。"""

import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx2

from servers import weather
with patch("dotenv.load_dotenv"):
    from servers import personal_server as server

from integrations.operations import OperationStore
from integrations.tool_executor import make_executor
from integrations.tool_gateway import ToolGateway
from integrations.tool_policy import get_policy


class WeatherTests(unittest.IsolatedAsyncioTestCase):
    def test_home_location_is_internal_weather_logic_not_a_model_tool(self):
        self.assertFalse(hasattr(server, "get_home_location"))
        self.assertIsNone(get_policy("get_home_location"))

    def setUp(self):
        self.requests = []
        self.location = {"latitude": 34.123456, "longitude": -118.654321,
                         "time_zone": "America/Los_Angeles"}
        self.forecast = {
            "latitude": self.location["latitude"], "longitude": self.location["longitude"],
            "timezone": "America/Los_Angeles",
            "current": {"temperature_2m": 15, "apparent_temperature": 13,
                        "precipitation": 0, "weather_code": 3, "wind_speed_10m": 6},
            "daily": {"time": ["2026-10-15", "2026-10-16"], "weather_code": [3, 63],
                      "temperature_2m_min": [12, 10], "temperature_2m_max": [19, 17],
                      "precipitation_probability_max": [30, 80]},
        }
        self.ha_status = self.weather_status = 200
        self.weather_error = None
        weather._LOCATION_CACHE.clear()
        self.addCleanup(weather._LOCATION_CACHE.clear)
        self.env = patch.dict(weather.os.environ, {"HA_URL": "http://ha/", "HA_TOKEN": "fake"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.real_client = httpx2.Client

    def handler(self, request):
        self.requests.append(request)
        if request.url.host == "ha":
            self.assertEqual(request.url.path, "/api/config")
            self.assertEqual(request.headers["Authorization"], "Bearer fake")
            return httpx2.Response(self.ha_status, json=self.location)
        self.assertEqual(str(request.url).split("?")[0], weather.FORECAST_URL)
        self.assertNotIn("Authorization", request.headers)
        if self.weather_error:
            raise self.weather_error
        return httpx2.Response(self.weather_status, json=self.forecast)

    def client(self, **kwargs):
        return self.real_client(transport=httpx2.MockTransport(self.handler), **kwargs)

    async def call(self, day="today"):
        with patch.object(server.httpx2, "Client", side_effect=self.client):
            return await server.mcp.call_tool("get_weather", {"day": day})

    def test_home_location_is_cached_but_forecast_is_fresh(self):
        with self.client() as client:
            first = weather.home_location(client)
            self.assertEqual(first, (34.123456, -118.654321, "America/Los_Angeles"))
            self.assertEqual(weather.home_location(client), first)
            self.assertEqual(len(self.requests), 1)
            weather.fetch_forecast(client, *first)
            weather.fetch_forecast(client, *first)
        self.assertEqual(len(self.requests), 3)
        params = self.requests[-1].url.params
        self.assertEqual(params["timezone"], first[2])
        self.assertEqual(params["forecast_days"], "2")
        self.assertEqual(params["current"], weather.CURRENT)
        self.assertEqual(params["daily"], weather.DAILY)
        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 5.0)

    def test_missing_zero_out_of_range_coordinates_and_bad_timezone_are_rejected(self):
        for location in ({}, {"latitude": 0, "longitude": 0, "time_zone": "UTC"},
                         {"latitude": True, "longitude": 10, "time_zone": "UTC"},
                         {"latitude": 91, "longitude": 10, "time_zone": "UTC"},
                         {"latitude": 1, "longitude": 2, "time_zone": "Invalid/Zone"}):
            self.location = location
            with self.subTest(location=location), self.client() as client, self.assertRaises((ValueError, KeyError)):
                weather.home_location(client)
            self.assertEqual(weather._LOCATION_CACHE, {})

    def test_describe_wmo_codes(self):
        for code, expected in [(0, "晴"), (3, "多云"), (45, "雾"), (55, "毛毛雨"),
                               (63, "雨"), (75, "雪"), (81, "阵雨"), (86, "阵雪"), (95, "雷雨")]:
            self.assertEqual(weather.describe(code), expected)

    def test_today_and_tomorrow_summary(self):
        self.assertEqual(weather.summarize(self.forecast, "today"),
                         "今天多云，12 到 19 度，降雨概率 30%，现在 15 度，体感 13 度")
        tomorrow = weather.summarize(self.forecast, "tomorrow")
        self.assertEqual(tomorrow, "明天雨，10 到 17 度，降雨概率 80%")
        self.assertNotIn("现在", tomorrow)

    async def test_tool_and_gateway_never_return_coordinates(self):
        async def worker_call(name, arguments):
            return await self.call(**arguments)
        worker = Mock(closed=False, call=AsyncMock(side_effect=worker_call))
        store = OperationStore(":memory:")
        self.addCleanup(store.close)
        gateway = ToolGateway(store, make_executor(ha_url="", http_client=None, worker=worker))
        op = await gateway.submit("weather", "get_weather", {"day": "today"})
        self.assertEqual(op.status, "succeeded")
        serialized = json.dumps(op.result, ensure_ascii=False)
        for private in ("34.123456", "-118.654321", "latitude", "longitude"):
            self.assertNotIn(private, serialized)
        self.assertEqual(op.result["structured_content"]["source"], "Open-Meteo")

    async def test_weather_500_and_timeout_return_error_without_request_url(self):
        self.weather_status = 500
        result = await self.call()
        self.assertTrue(result.is_error)
        self.assertIn("天气服务暂时不可用", result.content[0].text)
        self.assertNotIn("34.123456", result.content[0].text)
        self.weather_error = httpx2.ReadTimeout("https://example.com/?latitude=34.123456")
        result = await self.call()
        self.assertTrue(result.is_error)
        self.assertNotIn("34.123456", result.content[0].text)

    async def test_ha_failure_stops_before_weather_request(self):
        self.ha_status = 401
        result = await self.call()
        self.assertTrue(result.is_error)
        self.assertIn("读取家庭位置失败", result.content[0].text)
        self.assertEqual(len(self.requests), 1)

    async def test_missing_forecast_day_is_reported_and_null_probability_is_not_zero(self):
        self.forecast["daily"]["weather_code"] = [3]
        result = await self.call("tomorrow")
        self.assertTrue(result.is_error)
        self.forecast["daily"]["weather_code"] = [3, 63]
        self.forecast["daily"]["precipitation_probability_max"][0] = None
        self.assertIn("降雨概率暂无数据", weather.summarize(self.forecast, "today"))
