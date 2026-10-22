"""同一轮里的设备操作防护：来回开关、重复调整、无目标、次数上限。纯逻辑，不访问 Home Assistant。"""

import unittest

from brain.device_guard import MAX_DEVICE_WRITES, DeviceTurnGuard, device_target

ON, OFF, SET = "intent__HassTurnOn", "intent__HassTurnOff", "light__HassLightSet"


class DeviceGuardTests(unittest.TestCase):
    def allow(self, guard, tool, args):
        self.assertIsNone(guard.check(tool, args), (tool, args))
        guard.record(tool, args)

    def test_turning_the_same_target_on_and_off_in_one_turn_is_refused(self):
        guard = DeviceTurnGuard()
        self.allow(guard, OFF, {"name": "餐厅灯", "domain": ["light"]})
        self.assertIn("没有执行", guard.check(ON, {"name": "餐厅灯", "domain": ["light"]}))
        self.assertIn("没有执行", guard.check(OFF, {"name": "餐厅灯", "domain": ["light"]}))

    def test_target_matching_ignores_case_spacing_and_list_order(self):
        self.assertEqual(device_target({"name": " Kitchen Light ", "domain": ["switch", "light"]}),
                         device_target({"name": "kitchen light", "domain": ["light", "switch"]}))
        guard = DeviceTurnGuard()
        self.allow(guard, ON, {"area": "Living Room", "domain": ["light"]})
        self.assertIsNotNone(guard.check(OFF, {"area": " living room", "domain": ["light"]}))

    def test_different_targets_and_a_different_kind_of_change_are_allowed(self):
        guard = DeviceTurnGuard()
        self.allow(guard, ON, {"name": "餐厅灯"})
        self.allow(guard, ON, {"name": "厨房灯"})
        self.allow(guard, SET, {"name": "餐厅灯", "brightness": 30})  # 开灯后调亮度是正常的组合
        self.assertIsNotNone(guard.check(SET, {"name": "餐厅灯", "brightness": 80}))

    def test_power_and_brightness_tools_need_a_target_but_other_tools_do_not(self):
        guard = DeviceTurnGuard()
        for tool in (ON, OFF, SET):
            self.assertIn("没有指定设备目标", guard.check(tool, {}))
            self.assertIn("没有指定设备目标", guard.check(tool, {"name": "", "domain": []}))
        self.assertIsNone(guard.check(ON, {"domain": ["light"]}))  # 全部灯：有明确的类型
        self.allow(guard, "script__goodnight", {})                 # 脚本本来就没有目标参数
        self.assertIsNotNone(guard.check("script__goodnight", {}))

    def test_writes_stop_at_the_per_turn_limit(self):
        guard = DeviceTurnGuard()
        for index in range(MAX_DEVICE_WRITES):
            self.allow(guard, ON, {"name": f"灯{index}"})
        self.assertIn("上限", guard.check(ON, {"name": "新的灯"}))

    def test_a_new_turn_starts_clean(self):
        first = DeviceTurnGuard()
        self.allow(first, OFF, {"name": "餐厅灯"})
        self.assertIsNone(DeviceTurnGuard().check(ON, {"name": "餐厅灯"}))


if __name__ == "__main__":
    unittest.main()
