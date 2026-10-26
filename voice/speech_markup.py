"""大脑回答里给 TTS 的朗读标记（语音模式的 VOICE_RULES 允许写）。

声音事件：[笑] [叹气] [咳嗽] [清嗓子]，英文 (laugh) (sigh) (cough) (clears throat)。Breeze TTS 2 克隆模式会真的
笑、叹气；云端 TTS 不认识，合成前去掉。回声比对、打断记录、“刚问过用户”的判断都用去掉标记的文字。

语气：回答里出现 [严肃] 表示整段用严肃语气（[关键时刻]那类情况），由 Breeze 的音色指导模式实现；
标记本身不朗读，云端 TTS 忽略。
"""
import re

EVENT = re.compile(r'\[(?:笑|叹气|咳嗽|清嗓子)\]\s*|\((?:laugh|sigh|cough|clears throat)\)\s*', re.I)
STYLES = {'严肃': 'serious'}
STYLE = re.compile(r'\[(' + '|'.join(STYLES) + r')\]\s*')


def split_style(text):
    """去掉语气标记，返回 (带声音事件的朗读文字, 语气)。标记可能不在开头（大脑会在前面补“查到了，”）。"""
    styles = STYLE.findall(text)
    return STYLE.sub('', text), STYLES[styles[0]] if styles else None


def plain_speech(text):
    """去掉所有朗读标记，得到用户听到的文字内容。"""
    return EVENT.sub('', STYLE.sub('', text))
