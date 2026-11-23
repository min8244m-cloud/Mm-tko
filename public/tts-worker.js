/**
 * MM TTS FREE — Background Synthesis Web Worker (tts-worker.js)
 *
 * Runs Burmese G2P phonemization, BigInt64Array tensor preparation, and
 * ONNX Runtime Web (WASM) encoder/decoder inference in a dedicated worker thread
 * so the main UI thread stays responsive at 60fps.
 *
 * Model Architecture: YeeNyunn/nara-mini-v1 (Kokoro-based Burmese + English TTS)
 *   - encoder.onnx: { tokens: int64[1, T+2], style: float32[1, 128] }
 *                   -> { durations: int64[1, T+2], prosody: float32[1, Dp, T+2], text: float32[1, Dt, T+2] }
 *   - decoder.onnx: { prosody: float32[1, Dp, Frames], text: float32[1, Dt, Frames], style: float32[1, 128] }
 *                   -> { waveform: float32[1, Samples] } @ 24,000 Hz mono
 */

import * as ort from '/vendor/ort/ort.wasm.min.mjs';

const MODEL_CACHE_NAME = 'mm-tts-models-v1';

// Configure ONNX Runtime Web for multi-threaded SIMD WASM when crossOriginIsolated,
// gracefully falling back to 1 thread with SIMD enabled inside restricted iframes.
ort.env.wasm.wasmPaths = '/vendor/ort/';
ort.env.wasm.simd = true;
ort.env.wasm.numThreads =
  self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined'
    ? (self.navigator && self.navigator.hardwareConcurrency) || 4
    : 1;

let activeVoiceId = null;
let encoderSession = null;
let decoderSession = null;
let voiceMetadata = null;
let loadingVoiceId = null;
let loadingPromise = null;
let minedVoicingSet = new Set();
let minedSchwaSet = new Set();
let minedRulesLoaded = false;

// ============================================================================
// 1. Burmese Number Verbalization (Ported from YeeNyunn/nara-mini-v1 my_numbers.py)
// ============================================================================
const MY_DIGITS = ['သုည', 'တစ်', 'နှစ်', 'သုံး', 'လေး', 'ငါး', 'ခြောက်', 'ခုနစ်', 'ရှစ်', 'ကိုး'];
const MY_PLACES = { 1: 'ဆယ်', 2: 'ရာ', 3: 'ထောင်', 4: 'သောင်း' };
const MY_CREAKY = { 'ဆယ်': 'ဆယ့်', 'ရာ': 'ရာ့', 'ထောင်': 'ထောင့်' };
const MY_DIGIT_MAP = {
  '၀': '0', '၁': '1', '၂': '2', '၃': '3', '၄': '4',
  '၅': '5', '၆': '6', '၇': '7', '၈': '8', '၉': '9',
};

function myDigitsToAscii(str) {
  return str.replace(/[၀-၉]/g, (ch) => MY_DIGIT_MAP[ch] || ch);
}

function belowLakh(n) {
  const s = String(n);
  const parts = [];
  for (let i = 0; i < s.length; i++) {
    const d = Number(s[i]);
    const place = s.length - 1 - i;
    if (d) parts.push([d, place]);
  }
  const words = [];
  for (let j = 0; j < parts.length; j++) {
    const [d, place] = parts[j];
    if (place === 0) {
      words.push(MY_DIGITS[d]);
      continue;
    }
    let unit = MY_PLACES[place];
    if (j < parts.length - 1) {
      unit = MY_CREAKY[unit] || unit;
    }
    const lead = d === 1 && place === 1 && j === 0 ? '' : MY_DIGITS[d];
    words.push(lead + unit);
  }
  return words.join('');
}

function numberWords(n) {
  if (n === 0) return MY_DIGITS[0];
  if (n < 100000) return belowLakh(n);
  const lakhs = Math.floor(n / 100000);
  const rest = n % 100000;
  return numberWords(lakhs) + 'သိန်း' + (rest ? belowLakh(rest) : '');
}

function digitByDigit(str) {
  return [...str].map((c) => MY_DIGITS[Number(c)]).join(' ');
}

function verbalizeNumbers(text) {
  if (!/[0-9၀-၉]/.test(text)) return text;

  // Normalize Myanmar letter ဝ mistaken as zero ၀ inside digit runs
  text = text.replace(/(?:[၀-၉]|ဝ(?![ါ-ှ]))+/g, (m) =>
    /[၀-၉]/.test(m) ? m.replace(/ဝ/g, '၀') : m
  );

  const numRegex =
    /([0-9၀-၉]{1,3}(?:,[0-9၀-၉]{3})+|[0-9၀-၉]+)(?:\.([0-9၀-၉]+))?([kK](?![A-Za-z]))?(\s?%)?(\s?(?:Ks|KS|ks|MMK|mmk|Kyats?|kyats?)\b)?/g;

  text = text.replace(numRegex, (_match, intPart, fracPart, kSuffix, pctSuffix, curSuffix) => {
    const raw = myDigitsToAscii(intPart).replace(/,/g, '');
    let out = '';
    if ((raw.length > 1 && raw[0] === '0' && !fracPart) || raw.length > 10) {
      out = digitByDigit(raw);
    } else {
      let n = parseInt(raw, 10);
      if (kSuffix) n *= 1000;
      out = numberWords(n);
      if (fracPart) {
        out += ' ဒသမ ' + digitByDigit(myDigitsToAscii(fracPart));
      }
    }
    if (pctSuffix) out += ' ရာခိုင်နှုန်း';
    if (curSuffix) out += ' ကျပ်';
    return ` ${out} `;
  });

  return text.replace(/[ \t]{2,}/g, ' ').replace(/ (?=[!?.,။၊])/g, '').trim();
}

// ============================================================================
// 2. Burmese Romanized Name Replacement (from nara_names.py)
// ============================================================================
const NAME_MAP = {
  shwe: 'ʃwe→', aung: 'ʔaʊɴ→', kyaw: 'ʧɔ→', zaw: 'zɔ→', hla: 'ʰla→',
  htet: 'tʰɛʔ', naing: 'naɪɴ→', thu: 'θu→', moe: 'mo↗', myat: 'mjaʔ',
  min: 'miɴ→', nyunt: 'ɲʊɴ↓', khin: 'kʰiɴ→', mya: 'mja↗', hnin: 'ʰniɴ→',
  su: 'su→', phyo: 'pʰjo↗', htun: 'tʰʊɴ→', tun: 'tʊɴ→', oo: 'ʔu↗',
  zin: 'ziɴ→', thet: 'θɛʔ', nay: 'ne→', lin: 'liɴ→', kaung: 'kaʊɴ→',
  shein: 'ʃeɪɴ→', hein: 'heɪɴ→', myo: 'mjo↗', thant: 'θaɴ↓', san: 'saɴ↗',
  win: 'wɪɴ→', may: 'me→', soe: 'so↗', ye: 'jɛ↗', wai: 'we→',
  thida: 'θi→ də ɡa→',
};
const AMBIG_NAMES = new Set(['win', 'may', 'san', 'u', 'so', 'soe', 'ye', 'tin', 'ma']);

function replaceNames(text) {
  const regex = /\b[A-Z][a-z]+\b|\bU\b/g;
  const matches = [];
  let m;
  while ((m = regex.exec(text)) !== null) {
    matches.push({ word: m[0], start: m.index, end: m.index + m[0].length });
  }
  const keep = new Set();
  for (let i = 0; i < matches.length; i++) {
    const w = matches[i].word.toLowerCase();
    if (!(w in NAME_MAP)) continue;
    if (AMBIG_NAMES.has(w)) {
      const prevOk =
        i > 0 &&
        matches[i - 1].word.toLowerCase() in NAME_MAP &&
        text.slice(matches[i - 1].end, matches[i].start).trim() === '';
      const nxtOk =
        i + 1 < matches.length &&
        matches[i + 1].word.toLowerCase() in NAME_MAP &&
        text.slice(matches[i].end, matches[i + 1].start).trim() === '';
      if (!prevOk && !nxtOk) continue;
    }
    keep.add(i);
  }
  let out = '';
  let last = 0;
  for (let i = 0; i < matches.length; i++) {
    if (keep.has(i)) {
      out += text.slice(last, matches[i].start) + NAME_MAP[matches[i].word.toLowerCase()];
      last = matches[i].end;
    }
  }
  return out + text.slice(last);
}

// ============================================================================
// 3. Lightweight Browser English -> IPA Phonemizer (for mixed Burmese+English)
// ============================================================================
const EN_COMMON = {
  hello: 'həloʊ', world: 'wɜːld', free: 'fɹiː', text: 'tɛkst', speech: 'spiːtʃ',
  voice: 'vɔɪs', audio: 'ɔːdioʊ', offline: 'ɔːflaɪn', online: 'ɒnlaɪn',
  app: 'æp', local: 'loʊkəl', browser: 'bɹaʊzə', device: 'dɪvaɪs',
  music: 'mjuːzɪk', studio: 'stjuːdioʊ', model: 'mɒdəl', thank: 'θæŋk',
  you: 'juː', welcome: 'wɛlkəm', good: 'ɡʊd', morning: 'mɔːnɪŋ',
  evening: 'iːvnɪŋ', night: 'naɪt', myanmar: 'mjænmɑː', burmese: 'bɜːmiːz',
  yangon: 'jæŋɡɒn', mandalay: 'mændəleɪ', download: 'daʊnloʊd',
  player: 'pleɪə', fast: 'fæst', neural: 'njʊəɹəl', ai: 'eɪaɪ',
  tts: 'tiːtiːɛs', mm: 'ɛmɛm', pwa: 'piːdʌbəljuːeɪ',
};

function englishWordToIPA(word) {
  const lower = word.toLowerCase().replace(/[^a-z']/g, '');
  if (!lower) return '';
  if (EN_COMMON[lower]) return EN_COMMON[lower];

  let s = lower;
  s = s
    .replace(/tion$/g, 'ʃən')
    .replace(/sion$/g, 'ʒən')
    .replace(/ing$/g, 'ɪŋ')
    .replace(/ment$/g, 'mənt')
    .replace(/ness$/g, 'nəs')
    .replace(/able$/g, 'eɪbəl')
    .replace(/ee/g, 'iː')
    .replace(/oo/g, 'uː')
    .replace(/sh/g, 'ʃ')
    .replace(/ch/g, 'tʃ')
    .replace(/th/g, 'θ')
    .replace(/ph/g, 'f')
    .replace(/ng/g, 'ŋ')
    .replace(/qu/g, 'kw')
    .replace(/ai|ay/g, 'eɪ')
    .replace(/ea/g, 'iː')
    .replace(/oa/g, 'oʊ')
    .replace(/ou|ow/g, 'aʊ')
    .replace(/oi|oy/g, 'ɔɪ')
    .replace(/ar/g, 'ɑː')
    .replace(/or/g, 'ɔː')
    .replace(/er|ir|ur/g, 'ɜː')
    .replace(/ck/g, 'k')
    .replace(/c(?=[eiy])/g, 's')
    .replace(/c/g, 'k')
    .replace(/x/g, 'ks')
    .replace(/y$/g, 'i')
    .replace(/y/g, 'j')
    .replace(/a/g, 'æ')
    .replace(/e/g, 'ɛ')
    .replace(/i/g, 'ɪ')
    .replace(/o/g, 'ɒ')
    .replace(/u/g, 'ʌ')
    .replace(/r/g, 'ɹ')
    .replace(/g/g, 'ɡ');
  return s;
}

function phonemizeEnglishRuns(text) {
  if (!/[A-Za-z]/.test(text)) return text;
  // Preserve already-converted IPA name runs containing tone arrows
  return text.replace(/\b[A-Za-z][A-Za-z']*\b/g, (word) => englishWordToIPA(word));
}

// ============================================================================
// 4. Burmese G2P & Phonemizer (Ported from g2p_my.py + char_map.py + lexicon)
// ============================================================================
const LEXICON_RAW = {
  'ကျွန်ုပ်': 'tɕ-ə n-oʊʔ',
  'ယောက်ျား': 'j-aʊʔ tɕ-a-˥',
  'ပညာ': 'p-j-ɪɴ-˧ ɲ-a-˧',
  'တရုတ်': 't-ə j-oʊʔ',
  'မလေးရှား': 'm-ə l-e-˥ ʃ-a-˥',
  'သင်္ဘော': 'θ-ɪɴ-˥ b-ɔ-˥',
  'အင်္ကျီ': 'ʔ-eɪɴ-˥ dʑ-i-˧',
  'ရေခဲ': 'j-e-˧ g-ɛ-˥',
  'အရည်': 'ʔ-a-ˀ j-e-˧',
  'ပြဿနာ': 'p-j-aʔ θ-ə n-a-˧',
  'မေးလ်': 'm-e-˥',
  'ဆော့ဖ်': 's-ɔ-ˀ',
  'ဝဲလ်': 'w-ɛ-˥',
  'ပို့စ်': 'p-oʊ-ˀ',
  'ဝက်ဘ်': 'w-ɛʔ',
  'မင်္ဂလာ': 'm-ɪɴ-˧ g-ə l-a-˧',
  'မလေး': 'm-a-˧ l-e-˥',
  'မကြီး': 'm-a-˧ dʑ-i-˥',
  'မမကြီး': 'm-a-ˀ m-a-ˀ dʑ-i-˥',
  'မမ': 'm-a-ˀ m-a-ˀ',
  'တံတား': 't-ə d-a-˥',
  'ချောကလက်': 'tɕʰ-ɔ-˧ k-ə l-ɛʔ',
  'ကလေး': 'k-ə l-e-˥',
  'မည်း': 'm-ɛ-˥',
  'သည်း': 'θ-ɛ-˥',
  'လည်း': 'l-ɛ-˥',
  'ဆည်းလည်း': 'sʰ-ɛ-˥ l-ɛ-˥',
  'ရည်': 'j-e-˧',
  'ဒုက္ခ': 'd-oʊʔ kʰ-a-˧',
  'နာမည်': 'n-a-˧ m-e-˧',
  'လည်': 'l-e-˧',
  'ရည်းစား': 'j-i-˥ z-a-˥',
  'မြွေ': 'm-w-e-˧',
  'ဘုရား': 'pʰ-ə j-a-˥',
  'အစ်ကို': 'ʔ-ə k-o-˧',
  'လည်းကောင်း': 'l-ə g-aʊɴ-˥',
  'ပရိသတ်': 'p-ə j-eɪʔ θ-aʔ',
  'စကား': 's-ə g-a-˥',
  'စည်း': 's-i-˥',
  'စည်': 's-i-˧',
  'ပြည်': 'p-j-i-˧',
  'ကြည့်': 'tɕ-i-ˀ',
  'ကြည်': 'tɕ-i-˧',
  'မြည်': 'm-j-i-˧',
  'တည်': 't-i-˧',
  'ပြည့်': 'p-j-i-ˀ',
  'ချည်': 'tɕʰ-i-˧',
  'ခြည်': 'tɕʰ-i-˧',
  'အမည်': 'ʔ-ə m-j-i-˧',
  'စည်းကမ်း': 's-i-˥ g-aɴ-˥',
  'တံခါး': 't-ə g-a-˥',
  'တံဆိပ်': 't-ə z-eɪʔ',
  'တံထွေး': 't-ə tʰ-w-e-˥',
  'ခုနစ်': 'kʰ-ə n-ɪʔ',
  'သူခိုး': 'θ-ə kʰ-o-˥',
  'သူငယ်ချင်း': 'θ-ə ŋ-ɛ-˧ dʑ-ɪɴ-˥',
  'သူနာပြု': 'θ-ə n-a-˧ p-j-u-ˀ',
  'တကူးတက': 't-ə g-u-˥ d-ə g-a-ˀ',
  'သူဌေး': 'θ-ə tʰ-e-˥',
  'သူတောင်းစား': 'θ-ə d-aʊɴ-˥ z-a-˥',
  'သူပုန်': 'θ-ə b-oʊɴ-˧',
  'သူရဲ': 'θ-ə j-ɛ-˥',
  'သူလျှို': 'θ-ə ʃ-o-ˀ',
  'သူစိမ်း': 'θ-ə z-eɪɴ-˥',
  'ခဲလုံး': 'g-ɛ-˥ l-oʊɴ-˥',
  'သတ္တဝါ': 'θ-aʔ d-ə w-a-˧',
  'ငါးခြောက်': 'ŋ-ə tɕʰ-aʊʔ',
  'ငါးရံ့': 'ŋ-ə j-aɴ-ˀ',
  'ငါးရှဉ့်': 'ŋ-ə ʃ-ɪɴ-ˀ',
  'ငါးပိ': 'ŋ-ə p-i-˧',
  'ကျွန်မ': 'tɕ-ə m-a-ˀ',
  'ကျွန်တော်': 'tɕ-ə n-ɔ-˧',
  'ကျွန်တော့်': 'tɕ-ə n-ɔ-ˀ',
  'ပလက်ဖောင်း': 'p-l-ɛʔ f-aʊɴ-˥',
  'ပလတ်ဖောင်း': 'p-l-ɛʔ f-aʊɴ-˥',
  'တိရစ္ဆာန်': 't-ə r-eɪʔ sʰ-aɴ-˧',
  'တိရိစ္ဆာန်': 't-ə r-eɪʔ sʰ-aɴ-˧',
  'မုဆိုး': 'm-oʊʔ sʰ-o-˥',
  'ဂျပု': 'tɕ-ə p-u-ˀ',
  'တန်ခိုး': 't-ə kʰ-o-˥',
  'သည်းခံ': 'θ-i-˥ kʰ-aɴ-˧',
  'စင်္ကာပူ': 's-ɪɴ-˧ k-ə p-u-˧',
  'ပါရစေ': 'p-ə j-a-ˀ z-e-˧',
};

function normalizeBurmese(text) {
  text = text.replace(/[\u200b\ufeff]/g, '');
  text = text.replace(/့်/g, '့်');
  text = text.replace(/သြော်/g, 'ဪ');
  text = text.replace(/သြ/g, 'ဩ');
  const indep = [
    ['ဣ', 'အိ'], ['ဤ', 'အီ'], ['ဥ', 'အု'], ['ဦ', 'အူ'],
    ['ဧ', 'အေ'], ['ဩ', 'အော'], ['ဪ', 'အော်'],
  ];
  for (const [a, b] of indep) {
    text = text.split(a).join(b);
  }
  text = text.split('င်္').join('င်');
  text = text.replace(/([\u1000-\u1021])\u1039/g, '$1်');
  text = text.split('ဿ').join('သ်သ');
  text = text.split('၎င်း').join('လည်းကောင်း');
  text = text.split('၌').join('နှိုက်').split('၍').join('ရွေ့').split('၏').join('အိ');
  return text;
}

// Build normalized lexicon sorted longest-first and indexed by first char for O(1) lookup
const LEXICON = {};
for (const [k, v] of Object.entries(LEXICON_RAW)) {
  const nk = normalizeBurmese(k).replace(/ော်/g, '\ue001').replace(/ေါ်/g, '\ue001');
  LEXICON[nk] = v;
}
const LEX_KEYS = Object.keys(LEXICON).sort((a, b) => b.length - a.length);
const LEX_BY_FIRST_CHAR = new Map();
for (const k of LEX_KEYS) {
  const ch = k[0];
  if (!LEX_BY_FIRST_CHAR.has(ch)) {
    LEX_BY_FIRST_CHAR.set(ch, []);
  }
  LEX_BY_FIRST_CHAR.get(ch).push(k);
}

const CONS_RE = /^[\u1000-\u1021\u1023-\u1027\u1029\u102A]/;
const VOWEL_MED_SET = new Set('ါာိီုူေဲံ့း်ျြွှ\ue001');
const MIDSYL_SET = new Set('ါာိီုူေဲံ့း်ျြွှ');
const ASAT = '်';

function splitSyllables(text) {
  const out = [];
  let cur = '';
  let i = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (CONS_RE.test(ch)) {
      const isFinal = i + 1 < n && text[i + 1] === ASAT;
      if (cur && !isFinal) {
        out.push(cur);
        cur = ch;
      } else {
        cur += ch;
      }
      i += 1;
    } else if (VOWEL_MED_SET.has(ch)) {
      cur += ch;
      i += 1;
    } else {
      if (cur) {
        out.push(cur);
        cur = '';
      }
      const rest = text.slice(i);
      const m = rest.match(/^[A-Za-zæðθŋʒɐ-ʯᵻˈˌː→↗↓]+/);
      if (m) {
        out.push(m[0]);
        i += m[0].length;
        continue;
      }
      out.push(ch);
      i += 1;
    }
  }
  if (cur) out.push(cur);
  return out;
}

const ONSET = {
  'က': 'k', 'ခ': 'kʰ', 'ဂ': 'g', 'ဃ': 'g', 'င': 'ŋ',
  'စ': 's', 'ဆ': 'sʰ', 'ဇ': 'z', 'ဈ': 'z', 'ည': 'ɲ', 'ဉ': 'ɲ',
  'ဋ': 't', 'ဌ': 'tʰ', 'ဍ': 'd', 'ဎ': 'd', 'ဏ': 'n',
  'တ': 't', 'ထ': 'tʰ', 'ဒ': 'd', 'ဓ': 'd', 'န': 'n',
  'ပ': 'p', 'ဖ': 'pʰ', 'ဗ': 'b', 'ဘ': 'b', 'မ': 'm',
  'ယ': 'j', 'ရ': 'j', 'လ': 'l', 'ဝ': 'w', 'သ': 'θ', 'ဟ': 'h', 'ဠ': 'l', 'အ': 'ʔ',
};
const VELAR_PAL = { k: 'tɕ', 'kʰ': 'tɕʰ', g: 'dʑ' };
const ASPIRATE = { m: 'm̥', n: 'n̥', ɲ: 'ɲ̥', ŋ: 'ŋ̊', l: 'l̥', j: 'ʃ', w: 'w̥' };
const INDEP_SYL = {
  'ဣ': ['ʔ', 'i', 'ˀ'], 'ဤ': ['ʔ', 'i', '˧'], 'ဥ': ['ʔ', 'u', 'ˀ'],
  'ဦ': ['ʔ', 'u', '˧'], 'ဧ': ['ʔ', 'e', '˧'], 'ဩ': ['ʔ', 'ɔ', '˥'], 'ဪ': ['ʔ', 'ɔ', '˧'],
};
const OPEN_VOWELS = {
  '': ['a', 'ˀ'], 'ာ': ['a', '˧'], 'ိ': ['i', 'ˀ'], 'ီ': ['i', '˧'],
  'ု': ['u', 'ˀ'], 'ူ': ['u', '˧'], 'ေ': ['e', '˧'], 'ဲ': ['ɛ', '˥'],
  'ော': ['ɔ', '˥'], 'ို': ['o', '˧'],
};
const RHYME_CLOSED = {
  '|F_K': 'ɛʔ', '|F_S': 'ɪʔ', '|F_T': 'aʔ', 'ိ|F_T': 'eɪʔ', 'ု|F_T': 'oʊʔ',
  'ွ|F_T': 'ʊʔ', 'ို|F_K': 'aɪʔ', 'ော|F_K': 'aʊʔ', 'ွ|F_S': 'ɪʔ', 'ိ|F_S': 'ɪʔ',
  'ု|F_S': 'ʊʔ', '|F_NG': 'ɪɴ', '|F_NY': 'ɪɴ', '|F_N': 'aɴ', 'ာ|F_T': 'aʔ',
  'ာ|F_N': 'aɴ', 'ေ|F_T': 'ɪʔ', 'ု|F_K': 'oʊʔ', 'ာ|F_NY': 'ɪɴ', 'ိ|F_K': 'eɪʔ',
  'ိ|F_NG': 'eɪɴ', 'ို|F_N': 'aɪɴ', 'ိ|F_N': 'eɪɴ', 'ု|F_N': 'oʊɴ', 'ွ|F_N': 'ʊɴ',
  'ို|F_NG': 'aɪɴ', 'ော|F_NG': 'aʊɴ',
};
const VOICE_MAP = {
  k: 'g', 'kʰ': 'g', s: 'z', 'sʰ': 'z', t: 'd', 'tʰ': 'd',
  p: 'b', 'pʰ': 'b', tɕ: 'dʑ', 'tɕʰ': 'dʑ', θ: 'ð',
};
const CLASSIFIERS = new Set([
  'ယောက်', 'ခု', 'ခါ', 'နေ့', 'ချက်', 'ချောင်း', 'လုံး', 'ဖက်', 'ကောင်', 'ရက်', 'မျိုး', 'ပတ်',
  'ခေါက်', 'စု', 'ပါး', 'စီး', 'ဆူ', 'ရွက်', 'စောင်', 'လက်', 'ပင်', 'အုပ်', 'ထည်', 'ခွန်း',
  'လှမ်း', 'ဝိုင်း', 'သျှူး', 'ချောင်', 'နာရီ', 'မိနစ်', 'လ', 'နှစ်', 'ကြိမ်', 'ဦး', 'ပုဒ်', 'ခန်း', 'တန်း',
]);
const PUNCT_MAP = { '၊': ',', '။': '.', ',': ',', '.': '.', '?': '?', '!': '!', ' ': ' ' };

function finalClass(c) {
  if ('ကဂ'.includes(c)) return 'F_K';
  if ('စဇ'.includes(c)) return 'F_S';
  if ('တဒပဗဋဍသ'.includes(c)) return 'F_T';
  if ('နမဏ'.includes(c)) return 'F_N';
  if (c === 'င') return 'F_NG';
  if (c === 'ဉ') return 'F_NY';
  return null;
}

const SYL_RE =
  /^([\u1000-\u1021])([ျြွှ]*)([ါာိီုူေဲ]*)(?:([\u1000-\u1021])်)?(ံ?)(့?)(း?)(့?)$/;

function parseSyllable(s) {
  if (INDEP_SYL[s]) {
    return INDEP_SYL[s].join('-');
  }
  const m = s.match(SYL_RE);
  if (!m) return null;
  const [, base, meds, vowsRaw, fin, anusvara, dot1, visarga, dot2] = m;
  const dot = dot1 || dot2;
  let onset = ONSET[base];
  if (!onset) return null;
  let medial = '';

  if (meds.includes('ှ')) {
    onset = ASPIRATE[onset] || onset;
  }
  if (meds.includes('ျ') || meds.includes('ြ')) {
    if (VELAR_PAL[onset]) {
      onset = VELAR_PAL[onset];
    } else if (onset === 'ŋ') {
      onset = 'ɲ';
    } else if (onset !== 'ʃ' && onset !== 'j') {
      medial = 'j';
    }
  }
  let hasW = meds.includes('ွ');
  const vows = vowsRaw.replace(/ါ/g, 'ာ');
  const order = 'ေိီုူာဲ';
  let vkey = [...vows].sort((a, b) => order.indexOf(a) - order.indexOf(b)).join('');
  if (vkey === 'ော') vkey = 'ော';
  else if (vkey === 'ို') vkey = 'ို';

  let rhyme = null;
  let tone = null;

  if (fin || anusvara) {
    const fc = anusvara ? 'F_N' : finalClass(fin);
    if (fin && fin === 'ယ') {
      [rhyme, tone] = vkey === 'ို' ? ['o', '˧'] : ['ɛ', '˧'];
    } else if (fin && 'လဟ'.includes(fin)) {
      const ov = OPEN_VOWELS[vkey];
      if (!ov) return null;
      rhyme = ov[0];
      tone = '˧';
    } else if (fin && fin === 'ည') {
      [rhyme, tone] = vkey === 'ေ' ? ['ɛ', '˧'] : ['e', '˧'];
    } else if (fc) {
      let wkey = hasW && RHYME_CLOSED[`ွ|${fc}`] && !vkey ? 'ွ' : vkey;
      if (wkey === 'ွ') hasW = false;
      rhyme = RHYME_CLOSED[`${wkey}|${fc}`];
      if (!rhyme) return null;
      if (base === 'ဝ' && !vkey) {
        if (rhyme === 'aʔ') rhyme = 'ʊʔ';
        else if (rhyme === 'aɴ') rhyme = 'ʊɴ';
      }
      tone = rhyme.endsWith('ʔ') ? null : '˧';
    } else {
      return null;
    }
  } else {
    const ov = OPEN_VOWELS[vkey];
    if (!ov) return null;
    [rhyme, tone] = ov;
  }

  if (tone !== null) {
    if (visarga) tone = '˥';
    else if (dot) tone = 'ˀ';
  }

  const parts = [onset];
  if (hasW) medial = medial ? medial + 'w' : 'w';
  if (medial) parts.push(medial);
  parts.push(rhyme);
  if (tone) parts.push(tone);
  return parts.join('-');
}

function titReduces(chunk, i) {
  const nxt = chunk[i + 1];
  if (CLASSIFIERS.has(nxt)) return true;
  if (i > 0) {
    const b0 = chunk[i - 1];
    const b1 = chunk[i];
    if (i + 2 < chunk.length && chunk[i + 1] === b0 && chunk[i + 2] === b1) return false;
    if (i >= 3 && chunk[i - 3] === b0 && chunk[i - 2] === b1) return false;
    if (chunk[i - 1] === nxt) return false;
  }
  return true;
}

function applyMinedRules(chunk, parses) {
  const out = [...parses];
  for (let i = 0; i < chunk.length; i++) {
    const syl = chunk[i];
    const p = out[i];
    if (!p) continue;
    const parts = p.split('-');
    if (syl === 'တစ်' && i + 1 < chunk.length && titReduces(chunk, i)) {
      out[i] = 't-ə';
      continue;
    }
    if (i > 0 && minedVoicingSet.has(`${chunk[i - 1]}\t${syl}`) && VOICE_MAP[parts[0]]) {
      parts[0] = VOICE_MAP[parts[0]];
      out[i] = parts.join('-');
    }
    if (i + 1 < chunk.length && minedSchwaSet.has(`${chunk[i + 1]}\t${syl}`)) {
      out[i] = `${parts[0]}-ə`;
    }
  }
  return out;
}

const CHAR_MAP = {
  k: 'k', g: 'ɡ', 'ŋ': 'ŋ', s: 's', z: 'z', 'ɲ': 'ɲ', t: 't', d: 'd', n: 'n',
  p: 'p', b: 'b', m: 'm', j: 'j', r: 'r', l: 'l', w: 'w', 'θ': 'θ', 'ð': 'ð',
  h: 'h', 'ʔ': 'ʔ', 'ʃ': 'ʃ', f: 'f', 'kʰ': 'kʰ', 'sʰ': 'sʰ', 'tʰ': 'tʰ', 'pʰ': 'pʰ',
  'tɕ': 'ʧ', 'tɕʰ': 'ʧʰ', 'dʑ': 'ʤ', 'm̥': 'ʰm', 'n̥': 'ʰn', 'ɲ̥': 'ʰɲ',
  'ŋ̊': 'ʰŋ', 'l̥': 'ʰl', 'w̥': 'w', a: 'a', i: 'i', u: 'u', e: 'e', 'ɛ': 'ɛ',
  'ɔ': 'ɔ', o: 'o', 'ə': 'ə', 'ɴ': 'ɴ', 'ʔfin': 'ʔ', 'eɪ': 'A', 'oʊ': 'O',
  'aɪ': 'I', 'aʊ': 'W', '˧': '→', '˥': '↗', 'ˀ': '↓',
};

const RHYME_SPLIT = {
  'ɪɴ': ['i', 'ɴ'], 'aɴ': ['a', 'ɴ'], 'eɪɴ': ['A', 'ɴ'], 'oʊɴ': ['O', 'ɴ'],
  'ʊɴ': ['u', 'ɴ'], 'aɪɴ': ['I', 'ɴ'], 'aʊɴ': ['W', 'ɴ'], 'ɪʔ': ['i', 'ʔ'],
  'ʊʔ': ['u', 'ʔ'], 'aʔ': ['a', 'ʔ'], 'ɛʔ': ['ɛ', 'ʔ'], 'eɪʔ': ['A', 'ʔ'],
  'oʊʔ': ['O', 'ʔ'], 'aɪʔ': ['I', 'ʔ'], 'aʊʔ': ['W', 'ʔ'], a: ['a'], i: ['i'],
  u: ['u'], e: ['e'], 'ɛ': ['ɛ'], 'ɔ': ['ɔ'], o: ['o'], 'ə': ['ə'],
};

function sylToChars(syl) {
  const parts = syl.split('-');
  const last = parts[parts.length - 1];
  const tone = last === '˧' || last === '˥' || last === 'ˀ' ? last : null;
  const core = tone ? parts.slice(0, -1) : parts;
  if (core.length < 2) return '';
  const onset = core[0];
  const meds = core.slice(1, -1);
  const rhyme = core[core.length - 1];
  const rParts = RHYME_SPLIT[rhyme];
  if (!CHAR_MAP[onset] || !rParts) return '';
  let res = CHAR_MAP[onset];
  for (const m of meds) res += CHAR_MAP[m] || m;
  for (const c of rParts) res += CHAR_MAP[c] || c;
  if (tone) res += CHAR_MAP[tone];
  return res;
}

function burmeseG2P(rawText) {
  let text = normalizeBurmese(rawText).replace(/ော်/g, '\ue001').replace(/ေါ်/g, '\ue001');
  const chunks = [];
  let i = 0;
  const n = text.length;

  while (i < n) {
    let hit = null;
    const candidates = LEX_BY_FIRST_CHAR.get(text[i]);
    if (candidates) {
      for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
        const k = candidates[cIdx];
        if (text.startsWith(k, i)) {
          const j = i + k.length;
          if (j < n && MIDSYL_SET.has(text[j])) continue;
          hit = k;
          break;
        }
      }
    }
    if (hit) {
      chunks.push(['PHON', LEXICON[hit]]);
      i += hit.length;
    } else {
      chunks.push(['RAW', text[i]]);
      i += 1;
    }
  }

  const out = [];
  const flush = (buf) => {
    const syls = splitSyllables(buf);
    let runsyl = [];
    const closeRun = () => {
      if (!runsyl.length) return;
      const parses = runsyl.map((s) => {
        if (s === '\ue001') return 'ʔ-ɔ-˧';
        if (s.includes('\ue001')) {
          const p = parseSyllable(s.replace(/\ue001/g, 'ော'));
          return p ? p.replace(/ɔ-˥$/, 'ɔ-˧') : null;
        }
        return parseSyllable(s);
      });
      for (const p of applyMinedRules(runsyl, parses)) {
        if (p) out.push(p);
      }
      runsyl = [];
    };

    for (const syl of syls) {
      if (CONS_RE.test(syl) || syl.includes('\ue001')) {
        runsyl.push(syl);
      } else {
        closeRun();
        if (PUNCT_MAP[syl]) out.push(PUNCT_MAP[syl]);
        else if (syl.trim()) out.push(syl);
        else out.push(' ');
      }
    }
    closeRun();
  };

  let buf = '';
  for (const [kind, val] of chunks) {
    if (kind === 'RAW') {
      buf += val;
    } else {
      if (buf) {
        flush(buf);
        buf = '';
      }
      out.push(val);
    }
  }
  if (buf) flush(buf);

  const isEn = (tok) => !tok.includes('-') && /[A-Za-zæðθŋʒɐ-ʯᵻˈˌː]/.test(tok);
  const merged = [];
  for (let idx = 0; idx < out.length; idx++) {
    const tok = out[idx];
    if (tok === ' ') {
      const nxt = out.slice(idx + 1).find((x) => x !== ' ');
      const prev = merged[merged.length - 1];
      if (merged.length && ![',', '.', '?', '!'].includes(prev) && !isEn(prev) && (!nxt || !isEn(nxt))) {
        merged.push(',');
      }
    } else if (['.', '?', '!', ','].includes(tok)) {
      const prev = merged[merged.length - 1];
      if (prev === ',') merged[merged.length - 1] = tok;
      else if (['.', '?', '!'].includes(prev)) {
        if (tok !== prev) merged.push(tok);
      } else {
        merged.push(tok);
      }
    } else {
      merged.push(tok);
    }
  }
  while (merged.length && merged[merged.length - 1] === ',') merged.pop();
  while (merged.length && [',', '.', '?', '!'].includes(merged[0])) merged.shift();

  // Convert syllable tuples to compact Kokoro IPA char stream
  const finalTokens = [];
  for (const tok of merged.join(' ').split(/\s+/)) {
    if (!tok) continue;
    if ([',', '.', '?', '!'].includes(tok)) {
      finalTokens.push(tok);
    } else if (tok.includes('-')) {
      const mapped = sylToChars(tok);
      if (mapped) finalTokens.push(mapped);
    } else if (/[A-Za-zæðθŋʒɐ-ʯᵻˈˌː→↗↓]/.test(tok)) {
      finalTokens.push(tok);
    }
  }
  return finalTokens.join(' ');
}

function splitSpeech(text, limit = 140) {
  const units = [];
  const rawSentences = text.split(/(?<=[။…\n])\s*|(?<=[.!?])(?:\s+|$)/);
  const sentences = [];

  for (const seg of rawSentences) {
    const trimmed = seg.trim();
    if (!trimmed) continue;
    // Keep tiny list markers (e.g., "1.", "၁။") attached to the following sentence
    if (sentences.length > 0 && sentences[sentences.length - 1].length <= 4) {
      sentences[sentences.length - 1] = `${sentences[sentences.length - 1]} ${trimmed}`;
    } else {
      sentences.push(trimmed);
    }
  }

  for (const sentence of sentences) {
    if (sentence.length <= limit) {
      units.push(sentence);
      continue;
    }
    const clauses = sentence.split(/(?<=၊)\s*/);
    let current = '';
    for (let clause of clauses) {
      clause = clause.trim();
      if (!clause) continue;
      if (clause.length > limit) {
        if (current) {
          units.push(current);
          current = '';
        }
        while (clause.length > limit) {
          let cut = clause.lastIndexOf(' ', limit);
          if (cut <= 0) cut = limit;
          units.push(clause.slice(0, cut).trim());
          clause = clause.slice(cut).trim();
        }
        if (clause) units.push(clause);
      } else if (current && current.length + 1 + clause.length > limit) {
        units.push(current);
        current = clause;
      } else {
        current = current ? `${current} ${clause}` : clause;
      }
    }
    if (current) units.push(current);
  }
  return units;
}

function splitSubtitleLines(text, maxChars = 56) {
  const clean = text.trim();
  if (!clean) return [];
  if (clean.length <= maxChars) return [clean];

  const lines = [];
  let rem = clean;

  while (rem.length > maxChars) {
    const partsLeft = Math.ceil(rem.length / maxChars);
    const target = Math.ceil(rem.length / partsLeft);
    let bestCut = -1;
    let bestScore = Infinity;

    for (let idx = 12; idx <= Math.min(maxChars, rem.length - 6); idx++) {
      if (rem[idx] === ' ') {
        const tailLen = rem.length - idx - 1;
        const fitsRemaining = tailLen <= maxChars * (partsLeft - 1);
        const penalty = fitsRemaining ? 0 : 100;
        const score = Math.abs(idx - target) + penalty;
        if (score < bestScore) {
          bestScore = score;
          bestCut = idx;
        }
      }
    }

    if (bestCut === -1) {
      for (let idx = 12; idx <= Math.min(maxChars, rem.length - 6); idx++) {
        const ch = rem[idx];
        const prev = rem[idx - 1];
        const next = rem[idx + 1];
        if (/[\u1000-\u1021\u1023-\u1027\u1029\u102A]/.test(ch) && prev !== '\u1039' && next !== '\u103A') {
          const score = Math.abs(idx - target);
          if (score < bestScore) {
            bestScore = score;
            bestCut = idx;
          }
        }
      }
    }

    if (bestCut === -1) bestCut = maxChars;
    const head = rem.slice(0, bestCut).trim();
    if (head) lines.push(head);
    rem = rem.slice(bestCut).trim();
  }

  if (rem) lines.push(rem);
  return lines;
}

// ============================================================================
// 5. Model Loading from Cache Storage & Tensor Synthesis
// ============================================================================
async function loadCachedMinedRules(minedUrl) {
  if (minedRulesLoaded || !minedUrl) return;
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    let resp = await cache.match(minedUrl);
    if (!resp && navigator.onLine) {
      resp = await fetch(minedUrl);
      if (resp.ok) await cache.put(minedUrl, resp.clone());
    }
    if (resp && resp.ok) {
      const text = await resp.text();
      for (const line of text.split(/\r?\n/)) {
        if (!line || line.startsWith('#')) continue;
        const [kind, ctx, syl] = line.split('\t');
        if (!ctx || !syl) continue;
        if (kind === 'voicing') minedVoicingSet.add(`${ctx}\t${syl}`);
        else minedSchwaSet.add(`${ctx}\t${syl}`);
      }
      minedRulesLoaded = true;
    }
  } catch {
    // Graceful offline fallback using core built-in rules
  }
}

async function ensureVoiceLoaded(voiceConfig, silent = false) {
  if (activeVoiceId === voiceConfig.id && encoderSession && decoderSession && voiceMetadata) {
    return;
  }

  if (loadingPromise && loadingVoiceId === voiceConfig.id) {
    return loadingPromise;
  }

  if (loadingPromise) {
    await loadingPromise.catch(() => {});
    if (activeVoiceId === voiceConfig.id && encoderSession && decoderSession && voiceMetadata) {
      return;
    }
  }

  loadingVoiceId = voiceConfig.id;
  loadingPromise = (async () => {
    // Release previous ONNX sessions to keep mobile RAM low
    if (encoderSession && typeof encoderSession.release === 'function') {
      await encoderSession.release().catch(() => {});
    }
    if (decoderSession && typeof decoderSession.release === 'function') {
      await decoderSession.release().catch(() => {});
    }
    encoderSession = null;
    decoderSession = null;
    voiceMetadata = null;
    activeVoiceId = null;

    if (!silent) {
      self.postMessage({ type: 'status', phase: 'loading_model', message: 'Reading cached model weights...' });
    }

    const cache = await caches.open(MODEL_CACHE_NAME);
    const [voiceResp, encResp, decResp] = await Promise.all([
      cache.match(voiceConfig.voiceJsonUrl),
      cache.match(voiceConfig.encoderUrl),
      cache.match(voiceConfig.decoderUrl),
    ]);

    if (!voiceResp || !encResp || !decResp) {
      throw new Error('VOICE_NOT_CACHED');
    }

    const [meta, encBuf, decBuf] = await Promise.all([
      voiceResp.json(),
      encResp.arrayBuffer(),
      decResp.arrayBuffer(),
    ]);

    // Precompute mean style vector once per voice load
    if (Array.isArray(meta.styles) && meta.styles.length > 0) {
      const dim = meta.styles[0].length;
      const meanVec = new Float32Array(dim);
      for (let s = 0; s < meta.styles.length; s++) {
        const sv = meta.styles[s];
        for (let k = 0; k < dim; k++) {
          meanVec[k] += sv[k];
        }
      }
      for (let k = 0; k < dim; k++) {
        meanVec[k] /= meta.styles.length;
      }
      meta._meanStyle = meanVec;
    }

    if (voiceConfig.minedRulesUrl) {
      await loadCachedMinedRules(voiceConfig.minedRulesUrl);
    }

    if (!silent) {
      self.postMessage({ type: 'status', phase: 'compiling_wasm', message: 'Initializing WASM inference session...' });
    }

    const sessionOpts = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
    };

    const [enc, dec] = await Promise.all([
      ort.InferenceSession.create(new Uint8Array(encBuf), sessionOpts),
      ort.InferenceSession.create(new Uint8Array(decBuf), sessionOpts),
    ]);

    encoderSession = enc;
    decoderSession = dec;
    voiceMetadata = meta;
    activeVoiceId = voiceConfig.id;
  })();

  try {
    await loadingPromise;
  } finally {
    loadingPromise = null;
    loadingVoiceId = null;
  }
}

/**
 * Repeats [1, channels, seqLen] tensor along axis 2 according to integer durations[seqLen]
 */
function repeatAlongAxis2(srcTensor, durations, totalFrames) {
  const [, channels, seqLen] = srcTensor.dims;
  const srcData = srcTensor.data;
  const out = new Float32Array(channels * totalFrames);

  for (let c = 0; c < channels; c++) {
    const srcRowOffset = c * seqLen;
    const dstRowOffset = c * totalFrames;
    let cursor = 0;
    for (let t = 0; t < seqLen; t++) {
      const val = srcData[srcRowOffset + t];
      const rep = durations[t];
      if (rep === 1) {
        out[dstRowOffset + cursor++] = val;
      } else if (rep <= 4) {
        for (let r = 0; r < rep; r++) {
          out[dstRowOffset + cursor++] = val;
        }
      } else {
        out.fill(val, dstRowOffset + cursor, dstRowOffset + cursor + rep);
        cursor += rep;
      }
    }
  }
  return new ort.Tensor('float32', out, [1, channels, totalFrames]);
}

function buildStyleTensor(meta, tokenLength, expressiveness) {
  const centers = meta.centers || [100];
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centers.length; i++) {
    const d = Math.abs(centers[i] - tokenLength);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }

  const styleVec = meta.styles[bestIdx];
  const dim = styleVec.length;
  const meanVec = meta._meanStyle || styleVec;

  // expressiveness in [0.0, 1.0] -> modulation scale [0.55 .. 1.45]
  const exprScale = 0.55 + expressiveness * 0.9;
  const out = new Float32Array(dim);
  for (let k = 0; k < dim; k++) {
    out[k] = meanVec[k] + (styleVec[k] - meanVec[k]) * exprScale;
  }
  return new ort.Tensor('float32', out, [1, dim]);
}

function safeDisposeTensor(tensor) {
  if (tensor && typeof tensor.dispose === 'function') {
    tensor.dispose();
  }
}

function appendSubCues(cues, displayText, startSec, endSec) {
  const subLines = splitSubtitleLines(displayText, 56);
  if (subLines.length <= 1) {
    cues.push({
      startSec,
      endSec,
      text: displayText,
    });
    return;
  }

  const gapSec = 0.01;
  const totalSegDur = Math.max(0.2 * subLines.length, endSec - startSec);
  const usableDur = Math.max(0.15 * subLines.length, totalSegDur - gapSec * (subLines.length - 1));
  const weights = subLines.map((line) => Math.max(1, line.replace(/\s+/g, '').length));
  const totalW = weights.reduce((acc, w) => acc + w, 0);
  let cueCursor = startSec;

  for (let sIdx = 0; sIdx < subLines.length; sIdx++) {
    const isLast = sIdx === subLines.length - 1;
    const dur = isLast
      ? Math.max(0.15, endSec - cueCursor)
      : usableDur * (weights[sIdx] / totalW);
    const cueEnd = isLast ? endSec : Math.min(endSec, cueCursor + dur);
    cues.push({
      startSec: cueCursor,
      endSec: cueEnd,
      text: subLines[sIdx],
    });
    cueCursor = cueEnd + gapSec;
  }
}

async function synthesizeSpeech({ text, voiceConfig, speed = 1.0, expressiveness = 0.65, normalizeNumbers = true }) {
  await ensureVoiceLoaded(voiceConfig, false);

  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error('EMPTY_TEXT');
  }

  // Split input text into individual sentences by ။, \n, ?, ! (and sentence-ending .)
  const rawPieces = splitSpeech(trimmed, 140);
  const vocab = voiceMetadata.vocab;
  const sampleRate = voiceMetadata.sample_rate || 24000;
  const safeSpeed = Math.max(0.7, Math.min(1.5, speed));

  // Phonemize & validate all sentence chunks upfront so progress counts reflect active sentences
  const sentenceUnits = [];
  for (let pIdx = 0; pIdx < rawPieces.length; pIdx++) {
    const displayText = rawPieces[pIdx];
    const spokenText = normalizeNumbers ? verbalizeNumbers(displayText) : displayText;
    const prepared = phonemizeEnglishRuns(replaceNames(spokenText));
    const phonemes = burmeseG2P(prepared);

    const tokenIds = [];
    for (const ch of phonemes) {
      if (Object.prototype.hasOwnProperty.call(vocab, ch)) {
        tokenIds.push(vocab[ch]);
      }
    }
    if (tokenIds.length === 0) continue;

    if (tokenIds.length <= 480) {
      sentenceUnits.push({ displayText, tokenIds });
    } else {
      const parts = Math.ceil(tokenIds.length / 480);
      const subLines = splitSubtitleLines(displayText, Math.ceil(displayText.length / parts));
      const stride = Math.ceil(tokenIds.length / Math.max(1, subLines.length));
      for (let i = 0; i < subLines.length; i++) {
        const slice = tokenIds.slice(i * stride, (i + 1) * stride);
        if (slice.length > 0) {
          sentenceUnits.push({ displayText: subLines[i], tokenIds: slice });
        }
      }
    }
  }

  if (sentenceUnits.length === 0) {
    throw new Error('NO_PHONEMES');
  }

  const audioSegments = [];
  const cues = [];
  const silenceGapSamples = new Float32Array(Math.round(sampleRate * 0.25)); // 0.25s silence gap between sentences
  let sampleCursor = 0;

  for (let sIdx = 0; sIdx < sentenceUnits.length; sIdx++) {
    const unit = sentenceUnits[sIdx];
    self.postMessage({
      type: 'progress',
      completedChunks: sIdx,
      totalChunks: sentenceUnits.length,
      percent: Math.round((sIdx / sentenceUnits.length) * 100),
    });

    const paddedTokens = new BigInt64Array(unit.tokenIds.length + 2);
    paddedTokens[0] = 0n;
    for (let i = 0; i < unit.tokenIds.length; i++) {
      paddedTokens[i + 1] = BigInt(unit.tokenIds[i]);
    }
    paddedTokens[paddedTokens.length - 1] = 0n;

    const tokensTensor = new ort.Tensor('int64', paddedTokens, [1, paddedTokens.length]);
    const styleTensor = buildStyleTensor(voiceMetadata, unit.tokenIds.length, expressiveness);

    let encOutputs = null;
    let expandedProsody = null;
    let expandedText = null;
    let decOutputs = null;

    try {
      encOutputs = await encoderSession.run({
        tokens: tokensTensor,
        style: styleTensor,
      });

      const rawDurations = encOutputs.durations.data;
      const seqLen = paddedTokens.length;
      const scaledDurations = new Int32Array(seqLen);

      let totalFrames = 0;
      for (let t = 0; t < seqLen; t++) {
        const d = Math.max(1, Math.round(Number(rawDurations[t]) / safeSpeed));
        scaledDurations[t] = d;
        totalFrames += d;
      }

      if (totalFrames > 20000) {
        const shrink = 20000 / totalFrames;
        totalFrames = 0;
        for (let t = 0; t < seqLen; t++) {
          const d = Math.max(1, Math.floor(scaledDurations[t] * shrink));
          scaledDurations[t] = d;
          totalFrames += d;
        }
      }

      expandedProsody = repeatAlongAxis2(encOutputs.prosody, scaledDurations, totalFrames);
      expandedText = repeatAlongAxis2(encOutputs.text, scaledDurations, totalFrames);

      decOutputs = await decoderSession.run({
        prosody: expandedProsody,
        text: expandedText,
        style: styleTensor,
      });

      const firstKey = Object.keys(decOutputs)[0];
      const segment = new Float32Array(decOutputs[firstKey].data);

      for (let s = 0; s < segment.length; s++) {
        const v = segment[s];
        segment[s] = Number.isFinite(v) ? (v > 1 ? 1 : v < -1 ? -1 : v) : 0;
      }

      if (audioSegments.length > 0) {
        audioSegments.push(silenceGapSamples);
        sampleCursor += silenceGapSamples.length;
      }

      const startSec = sampleCursor / sampleRate;
      audioSegments.push(segment);
      sampleCursor += segment.length;
      const endSec = Math.max(startSec + 0.15, sampleCursor / sampleRate);

      appendSubCues(cues, unit.displayText, startSec, endSec);

      self.postMessage({
        type: 'progress',
        completedChunks: sIdx + 1,
        totalChunks: sentenceUnits.length,
        percent: Math.round(((sIdx + 1) / sentenceUnits.length) * 100),
      });
    } finally {
      safeDisposeTensor(tokensTensor);
      safeDisposeTensor(styleTensor);
      safeDisposeTensor(expandedProsody);
      safeDisposeTensor(expandedText);
      if (encOutputs) {
        Object.values(encOutputs).forEach(safeDisposeTensor);
      }
      if (decOutputs) {
        Object.values(decOutputs).forEach(safeDisposeTensor);
      }
    }
  }

  let totalLength = 0;
  for (const seg of audioSegments) totalLength += seg.length;

  const fullBuffer = new Float32Array(totalLength);
  let offset = 0;
  for (const seg of audioSegments) {
    fullBuffer.set(seg, offset);
    offset += seg.length;
  }

  return { audioBuffer: fullBuffer, sampleRate, cues };
}

self.addEventListener('message', async (event) => {
  const { id, type, payload } = event.data || {};
  if (type === 'warmup') {
    try {
      if (payload && payload.voiceConfig) {
        await ensureVoiceLoaded(payload.voiceConfig, true);
      }
      self.postMessage({ id, type: 'warmed' });
    } catch {
      // Ignore background warmup errors silently (e.g. uncached voice)
    }
    return;
  }

  if (type === 'unload') {
    if (loadingPromise) {
      await loadingPromise.catch(() => {});
    }
    activeVoiceId = null;
    if (encoderSession && typeof encoderSession.release === 'function') {
      await encoderSession.release().catch(() => {});
    }
    if (decoderSession && typeof decoderSession.release === 'function') {
      await decoderSession.release().catch(() => {});
    }
    encoderSession = null;
    decoderSession = null;
    voiceMetadata = null;
    self.postMessage({ id, type: 'unloaded' });
    return;
  }

  if (type === 'synthesize') {
    try {
      const { audioBuffer, sampleRate, cues } = await synthesizeSpeech(payload);
      self.postMessage(
        {
          id,
          type: 'result',
          audioBuffer,
          sampleRate,
          cues,
        },
        [audioBuffer.buffer]
      );
    } catch (err) {
      self.postMessage({
        id,
        type: 'error',
        error: err && err.message ? err.message : String(err),
      });
    }
  }
});
