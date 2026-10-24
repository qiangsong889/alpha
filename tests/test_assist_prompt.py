"""HA Assist prompt 只保留设备列表：HA 自带的说明里有和 Alpha 规则冲突的一条。不连接 HA。"""

import unittest

from integrations.mcp_client import condense_assist

ASSIST = """You ARE equipped to answer questions about the current state of
the home by retrieving live context. This is a primary function.
For general knowledge questions not about the home: Answer truthfully from internal knowledge.

Static Context: An overview of the areas and the devices in this smart home:
- names: Dining Room Light, 餐厅灯
  domain: light
  areas: Dining Room
- names: Blinds1, 餐厅左窗帘
  domain: cover

When controlling Home Assistant always call the intent tools. Use intent__HassTurnOn to lock and intent__HassTurnOff to unlock a lock.
When a user asks to turn on all devices of a specific type, ask the user to specify an area, unless there is only one device of that type.
This device is not able to start timers."""


class CondenseAssistTests(unittest.TestCase):
    def test_keeps_the_device_list_and_drops_ha_instructions(self):
        text = condense_assist(ASSIST)
        self.assertIn('- names: Dining Room Light, 餐厅灯', text)  # 中英文名对应靠这份列表
        self.assertIn('- names: Blinds1, 餐厅左窗帘', text)
        self.assertIn('GetLiveContext', text)
        # 和“把灯关了就关所有灯，不先问”冲突；门锁、定时器与这个家无关。
        self.assertNotIn('ask the user to specify an area', text)
        self.assertNotIn('lock', text)
        self.assertNotIn('timers', text)
        self.assertLess(len(text), len(ASSIST))

    def test_unknown_format_is_kept_whole_so_devices_are_never_lost(self):
        changed = 'Devices in this home:\n- 客厅灯 (light)'
        self.assertEqual(condense_assist(changed), changed)

    def test_device_list_at_the_end_without_trailing_instructions(self):
        text = condense_assist('intro\nStatic Context: devices\n- names: 客厅灯\n  domain: light')
        self.assertTrue(text.endswith('- names: 客厅灯\n  domain: light'))


if __name__ == '__main__':
    unittest.main()
