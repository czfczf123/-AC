import epitran

# 创建一个日语的 Epitran 对象
epi = epitran.Epitran('jpn-Hrgn')

# 将日语文本转换为国际音标
ipa = epi.transliterate('こんにちは')

print(ipa)


epi = epitran.Epitran('eng-Latn')
epi.transliterate('Berkeley')
