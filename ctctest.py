from ctcdecode import CTCBeamDecoder

labels="_ i ɪ e ɛ æ ɑ ɔ o ʊ u ʌ ə ɚ ɝ ɾ t d n s z ʃ ʒ θ ð k ɡ ŋ p b m f v h r j l w ʍ x ʔ tʃ dʒ aɪ aʊ ɔɪ eɪ oʊ ɪər ɛər ʊər aɪər aʊər ɔɪər pʰ tʰ kʰ bʷ tʷ dʷ kʷ ɡʷ tʃʰ dʒʰ mː nː ŋː pː tː kː bː dː ɡː tʃː dʒː tʰː kʰː pːʰ fː θː sː ʃː hː rː lː wː jː eː iː uː ɑː ɔː oː ʌː əː aɪː aʊː ɔɪː eɪː oʊː ɪərː ɛərː ʊərː aɪərː aʊərː ɔɪərː jʊ ɑɪːɹ"
decoder = CTCBeamDecoder(
    labels,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=100,
    num_processes=4,
    blank_id=0,
    log_probs_input=False
)
#beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)

p, b, t, d, ʈ, ɖ, c, ɟ, k, ɡ, q, ɢ, ʔ, m, ɱ, n, ɳ, ɲ, ŋ, ɴ, ʙ, r, ʀ, ɾ, ɽ, ʃ, ʒ, ʂ, ʐ, ɕ, ʑ, ɬ, ɮ, l, ɭ, ʎ, ʟ, ɺ, ɺ̢, ɓ, ɗ, ʄ, ɠ, ʛ, ʘ, ǀ, ǃ, ǂ, ǁ, ɨ, ʉ, ɯ, ɪ, ʏ, ʊ, ɪ̈, ʊ̈, ə, ɵ, ɘ, ɤ, ɚ, ɝ, ɛ, ɜ, ɞ, ʌ, ɔ, ɐ, æ, ɑ, ɒ, ʕ, ʡ, ʢ, ʢ̆, i, y, ɨ̞, ʉ̞, ɯ̞, u, ɪ̞, ʏ̞, ʊ̞, e̞, ø̞, ɘ̞, ɵ̞, ɤ̞, o̞, ə̞, ɛ̞, œ̞, ɜ̞, ɞ̞, ʌ̞, ɔ̞, æ̞, ɐ̞, a̠, ɑ̠, ɒ̠, ɑ̟, ɒ̟, ɑ̃, ɒ̃, ɑ̰, ɑ̰̃