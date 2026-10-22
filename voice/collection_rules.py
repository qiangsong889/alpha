"""当前 TurnCollector 使用的执行口令与未完句规则。"""
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

