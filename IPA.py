import epitran

epi = epitran.Epitran('eng-Latn')
import epitran
epi = epitran.Epitran('eng-Latn')
print(epi.transliterate(u'Berkeley'))

#print(epi.trans_list('it will be the see you again'))

epi = epitran.Epitran('cmn-Hans', cedict_file='g2p/cedict_1_0_ts_utf-8_mdbg.txt')
print(epi.trans_list("过去，人们常说\n"))

from epitran.backoff import Backoff
backoff = Backoff(['eng-Latn', 'cmn-Hans'], cedict_file='g2p/cedict_1_0_ts_utf-8_mdbg.txt')
print(backoff.trans_list('打架，大家'))
print(backoff.trans_list("it will be the see you again"))