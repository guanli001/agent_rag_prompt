"""
count_vowels(s) — 统计字符串里元音字母的数量。
a e i o u，大小写都算
"""

a = "aeiouAEIOU"


def count_vowels(s):
    sum_s = 0
    for i in s:
        if i in a:
            sum_s += 1
    return sum_s
