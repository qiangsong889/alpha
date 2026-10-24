"""当前 TurnCollector 使用的执行口令、未完句规则，以及播报期间的回声判断。"""
from difflib import SequenceMatcher
import re

EXECUTE = {"执行", "執行", "执行吧", "執行吧", "开始执行", "開始執行", "立即执行", "就这些", "就這些", "没了", "沒了"}
TRAILING = re.compile(r"(?:然后再|然後再|再帮我|再幫我|还有|還有|然后|然後|或者|等一下|等等|先别执行|先別執行|把|给|給|和|跟)$")


def unfinished(text: str) -> bool:
    """只识别明显未完的句尾；这是保守规则，不是完整的语义理解。"""
    text = text.rstrip(" ，。！？、,;；!?…\t\n")
    tail = re.split(r"[，。！？,;；!?]", text)[-1].strip()
    return bool(TRAILING.search(text) or (
        tail.startswith("把") and not re.search(
            r"加|放|买|買|删|刪|关|關|开|開|调|調|设|設|标|標|完成|拿|移|换|換|改|提醒|取消|撤", tail
        )
    ))



# 播报期间转写出的文字，至少这个比例能在 Alpha 正在说的话里按顺序找到，就当作回声。
# 实测回声转写：“可以听到”100%、“购物清单还有您”86%、“还有牛奶鸡蛋”100%；
# 用户插话“厨房灯关了”对“客厅灯开了，厨房灯也开了”是 80%，不能被当成回声。
ECHO_RATIO = 0.85
_NOT_TEXT = re.compile(r"[\W_]+")


def echo_ratio(heard: str, said: str) -> float:
    """转写里有多少比例的字，能在 Alpha 说的话里按原顺序找到（忽略标点、空格和大小写）。"""
    heard, said = _NOT_TEXT.sub("", heard).casefold(), _NOT_TEXT.sub("", said).casefold()
    if not heard or not said:
        return 0.0
    blocks = SequenceMatcher(None, heard, said, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks) / len(heard)


def is_echo(heard: str, said: str) -> bool:
    # 太短的转写（一两个字）匹配不可靠，交给后面的判断。
    return len(_NOT_TEXT.sub("", heard)) >= 3 and echo_ratio(heard, said) >= ECHO_RATIO
