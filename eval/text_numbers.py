"""仅供评分使用的数字等价比较；不修改原话、证据或存储内容。"""

import re

_DIGITS = dict(zip('零〇一二两三四五六七八九', '001223456789'))
_SMALL = {'十': 10, '百': 100, '千': 1000}
_NUMBERS = re.compile(r'负?[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二两三四五六七八九]+)?')


def _integer(text: str) -> str:
    for char, unit in (('亿', 100000000), ('万', 10000)):
        if char in text:
            left, right = text.split(char, 1)
            return str(int(_integer(left) or '1') * unit + int(_integer(right) or '0'))
    if not any(char in _SMALL for char in text):
        return ''.join(_DIGITS[char] for char in text)
    total = number = 0
    for char in text:
        if char in _DIGITS:
            number = int(_DIGITS[char])
        else:
            total += (number or 1) * _SMALL[char]
            number = 0
    return str(total + number)


def normalize_numbers(text: str) -> str:
    text = text.translate(str.maketrans('０１２３４５６７８９％', '0123456789%'))
    def convert(match):
        value = match.group()
        sign = '-' if value.startswith('负') else ''
        whole, _, decimal = value.removeprefix('负').partition('点')
        return sign + _integer(whole) + ('.' + ''.join(_DIGITS[c] for c in decimal) if decimal else '')
    text = text.replace('百分之', '\x00PERCENT\x00')
    text = _NUMBERS.sub(convert, text)
    return re.sub(r'\x00PERCENT\x00(-?\d+(?:\.\d+)?)', r'\1%', text)


def contains_equivalent(text: str, snippet: str) -> bool:
    """双向归一化，并避免把 26 度误匹配到 126 度或 26.5 度。"""
    pattern = ''.join('(?<![0-9.+-])' + re.escape(part) + '(?![0-9.])'
                      if re.fullmatch(r'-?\d+(?:\.\d+)?', part) else re.escape(part)
                      for part in re.split(r'(-?\d+(?:\.\d+)?)', normalize_numbers(snippet)) if part)
    return re.search(pattern, normalize_numbers(text)) is not None
