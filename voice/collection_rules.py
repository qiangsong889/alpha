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


# 播报结束后不久开始的声音，也可能是回声的尾巴（房间混响、设备延迟）：实测结束后 118ms
# 转写出“资料。”，正是刚说完那句话的最后两个字。人听完再开口通常要几百毫秒，所以窗口很短，
# 只和刚说完那句话的结尾比对。
ECHO_TAIL_S = 0.5
ECHO_TAIL_CHARS = 20


def is_echo(heard: str, said: str, *, min_chars: int = 3) -> bool:
    # 太短的转写匹配不可靠，交给后面的判断；回声尾巴只和句尾比对，所以允许两个字。
    return len(_NOT_TEXT.sub("", heard)) >= min_chars and echo_ratio(heard, said) >= ECHO_RATIO


# 播报期间（QWEN_ALPHA_INTERRUPT_MODE=call）只接受这些：叫名字，或者明确的“停”。
# 名字的写法包括 ASR 实际认错过的“奥特”“阿发”和几种同音写法。
WAKE = re.compile(r"^\s*(?:(?:嘿|你好|hey)[，,\s]*)?(?:alpha(?![A-Za-z0-9_])|alfa(?![A-Za-z0-9_])|"
                  r"阿尔法|阿爾法|阿尔发|阿法|阿发|奥法|奥尔法|艾尔法|奥特(?!曼))[，,。.!！\s]*", re.I)
STOP = {"停", "停一下", "停下", "停下来", "暂停", "别说了", "不要说了", "不用说了", "等一下", "等等", "安静", "闭嘴", "够了"}
# 回声里也可能出现 Alpha 的名字（“我是 Alpha，你的家居……”）：去掉名字后的部分只要有这么多能在
# Alpha 正在说的话里找到，就仍然当作回声。比全句的门槛低，因为名字后面通常紧跟着走样的转写。
WAKE_ECHO_RATIO = 0.6
# 而且要有连续几个字和原话一样：回声是原话的一段（“你的家居”），真心提问只是零散地碰到几个字
# （“Alpha，现在几点了”对“现在下午五点二十一分”：重合 60%，但最长只连续 2 个字）。
WAKE_ECHO_RUN = 3


def playback_command(heard: str, said: str) -> str | None:
    """播报期间这句话算什么：'stop'（只停下）、'call'（叫 Alpha，交给大脑）或 None（忽略）。"""
    wake = WAKE.match(heard)
    rest = heard[wake.end():] if wake else heard
    plain = _NOT_TEXT.sub("", rest)
    if plain in STOP:
        return "stop"
    if not wake:
        return None
    if not plain:
        return "call"
    if echo_ratio(rest, said) >= WAKE_ECHO_RATIO and _longest_run(rest, said) >= WAKE_ECHO_RUN:
        return None
    return "call"


def _longest_run(heard: str, said: str) -> int:
    heard, said = _NOT_TEXT.sub("", heard).casefold(), _NOT_TEXT.sub("", said).casefold()
    if not heard or not said:
        return 0
    return SequenceMatcher(None, heard, said, autojunk=False).find_longest_match(0, len(heard), 0, len(said)).size
