/**
 * MM TTS FREE — Main Client Application (app.js)
 *
 * Modular ES6+ controller for:
 *   1. Bilingual UI (Burmese / English) & Dark/Light Mode persistence
 *   2. ModelCacheManager with Cache API ('mm-tts-models-v1'), localStorage LRU
 *      eviction (max 2 voice models), streaming download progress, & persistent storage
 *   3. Background Web Worker ('/tts-worker.js') orchestration for ONNX inference
 *   4. Custom Audio Player (Web Audio / Blob URL playback with Play/Pause/Seek/Replay)
 *   5. Dual Audio Export: Pure JS 44-byte RIFF/WAVE PCM WAV encoder + lamejs 128kbps MP3
 *   6. Full PWA support (Service Worker registration & in-app Install Prompt)
 */

// ============================================================================
// 1. Hugging Face Model Configuration (YeeNyunn/nara-mini-v1)
// ============================================================================
const HF_BASE_URL = 'https://huggingface.co/YeeNyunn/nara-mini-v1/resolve/main';
const MODEL_CACHE_NAME = 'mm-tts-models-v1';
const META_STORAGE_KEY = 'mm-tts-cache-meta-v1';
const MAX_CACHED_VOICES = 2;

const VOICE_PRESETS = [
  {
    id: 'shoon',
    nameEn: 'Shoon',
    nameMy: 'ရွှန်း (Shoon)',
    descEn: 'Warm, clear female voice · Natural narration',
    descMy: 'ကြည်လင်ပြတ်သားသော အမျိုးသမီးအသံ · ပုံမှန်စကားပြော',
    gender: 'Female',
    approxBytes: 189995436, // ~181.2 MiB (encoder + decoder + voice.json)
    encoderUrl: `${HF_BASE_URL}/onnx/shoon/encoder.onnx`,
    decoderUrl: `${HF_BASE_URL}/onnx/shoon/decoder.onnx`,
    voiceJsonUrl: `${HF_BASE_URL}/onnx/shoon/voice.json`,
    minedRulesUrl: `${HF_BASE_URL}/frontend/mined_rules.tsv`,
  },
  {
    id: 'thazin',
    nameEn: 'Thazin',
    nameMy: 'သဇင် (Thazin)',
    descEn: 'Soft, expressive female voice · Storytelling',
    descMy: 'နူးညံ့သိမ်မွေ့သော အမျိုးသမီးအသံ · ပုံပြင်နှင့် စာဖတ်',
    gender: 'Female',
    approxBytes: 189995436,
    encoderUrl: `${HF_BASE_URL}/onnx/thazin/encoder.onnx`,
    decoderUrl: `${HF_BASE_URL}/onnx/thazin/decoder.onnx`,
    voiceJsonUrl: `${HF_BASE_URL}/onnx/thazin/voice.json`,
    minedRulesUrl: `${HF_BASE_URL}/frontend/mined_rules.tsv`,
  },
  {
    id: 'ko-banyar',
    nameEn: 'Ko Banyar',
    nameMy: 'ကိုဗညား (Ko Banyar)',
    descEn: 'Calm, authoritative male voice · News & docs',
    descMy: 'တည်ငြိမ်ခန့်ညားသော အမျိုးသားအသံ · သတင်းနှင့် မှတ်တမ်း',
    gender: 'Male',
    approxBytes: 189995436,
    encoderUrl: `${HF_BASE_URL}/onnx/ko-banyar/encoder.onnx`,
    decoderUrl: `${HF_BASE_URL}/onnx/ko-banyar/decoder.onnx`,
    voiceJsonUrl: `${HF_BASE_URL}/onnx/ko-banyar/voice.json`,
    minedRulesUrl: `${HF_BASE_URL}/frontend/mined_rules.tsv`,
  },
  {
    id: 'htet-myat',
    nameEn: 'Htet Myat',
    nameMy: 'ထက်မြက် (Htet Myat)',
    descEn: 'Crisp, energetic male voice · Tutorials & clips',
    descMy: 'သွက်လက်တက်ကြွသော အမျိုးသားအသံ · သင်ခန်းစာနှင့် ဗီဒီယို',
    gender: 'Male',
    approxBytes: 189995436,
    encoderUrl: `${HF_BASE_URL}/onnx/htet-myat/encoder.onnx`,
    decoderUrl: `${HF_BASE_URL}/onnx/htet-myat/decoder.onnx`,
    voiceJsonUrl: `${HF_BASE_URL}/onnx/htet-myat/voice.json`,
    minedRulesUrl: `${HF_BASE_URL}/frontend/mined_rules.tsv`,
  },
];

// ============================================================================
// 2. Bilingual Dictionary (Burmese + English) & Sample Texts
// ============================================================================
const I18N = {
  en: {
    online: 'Online',
    offline: 'Offline',
    clearCache: 'Clear Cache',
    installApp: 'Install App',
    heroTitle: 'Burmese Neural Voice Studio',
    heroSubtitle: '100% client-side Myanmar & English speech synthesis running locally in WebAssembly.',
    voicesHeading: 'Voice Presets',
    voicesSub: 'YeeNyunn/nara-mini-v1 · Max 2 cached (LRU)',
    lruNote: 'To save device storage, up to 2 voice models (~181 MB each) remain cached offline. Downloading a 3rd automatically removes the least recently used voice.',
    composerHeading: 'Text Input',
    copyText: 'Copy',
    copied: 'Copied!',
    clearText: 'Clear',
    samplesLabel: 'Samples:',
    sampleGreeting: 'မင်္ဂလာပါ (Greeting)',
    sampleNews: 'သတင်း (News)',
    sampleNumbers: 'ကိန်းဂဏန်း (Numbers)',
    sampleBilingual: 'Mixed MY/EN',
    controlsHeading: 'Voice Controls',
    defaultBtn: 'Default (1.0x · 65%)',
    lock: 'Lock',
    locked: 'Locked',
    speedLabel: 'Speech Speed',
    expressLabel: 'Expressiveness',
    calm: 'Calm',
    balanced: 'Balanced',
    vivid: 'Vivid',
    normalizeNumbers: 'Convert Burmese & currency numbers into spoken words (e.g. 2,500 Ks)',
    generateBtn: 'ဖတ်ပါ (Generate & Play)',
    generatingBtn: 'Synthesizing locally...',
    downloadingBtn: 'Downloading voice model...',
    noAudioYet: 'No audio generated yet',
    downloadWav: 'Download WAV',
    downloadMp3: 'Download MP3',
    downloadSrt: 'Download SRT',
    historyHeading: 'Synthesis History',
    historySub: 'Saved locally in your browser · Replay audio or export WAV / MP3 / SRT',
    clearHistory: 'Clear History',
    historyEmpty: 'No synthesis history yet. Generated speeches will appear here automatically.',
    loadAndPlay: 'Play',
    loadText: 'Load Text',
    mp3Short: 'MP3',
    deleteItem: 'Delete',
    privacyNote: '100% Local Processing — Your text never leaves your device.',
    offlineReady: 'Offline Ready',
    downloadRequired: 'Download Required',
    cancel: 'Cancel',
    close: 'Close',
    iosModalTitle: 'Install MM TTS FREE on iPhone / iPad',
    iosStep1: 'Tap the Share button in Safari toolbar.',
    iosStep2: 'Scroll down and tap "Add to Home Screen".',
    iosStep3: 'Tap "Add" to launch MM TTS FREE offline anytime.',
    storagePersisted: 'Persistent Storage',
    storageTemporary: 'Standard Storage',
  },
  my: {
    online: 'အွန်လိုင်း',
    offline: 'အော့ဖ်လိုင်း',
    clearCache: 'Cache ဖျက်မည်',
    installApp: 'App ထည့်သွင်းမည်',
    heroTitle: 'မြန်မာ AI အသံထွက် စတူဒီယို',
    heroSubtitle: 'ဆာဗာမလိုဘဲ သင့်စက်ထဲတွင်သာ ၁၀၀% အခမဲ့ အသံထွက်ဖန်တီးပေးသည့် စနစ်။',
    voicesHeading: 'အသံရွေးချယ်ရန်',
    voicesSub: 'YeeNyunn/nara-mini-v1 · အများဆုံး ၂ ခု သိမ်းဆည်းမည် (LRU)',
    lruNote: 'ဖုန်း/စက် Storage မပြည့်စေရန် အသံမော်ဒယ် ၂ ခု (~181 MB စီ) အထိသာ အော့ဖ်လိုင်း သိမ်းဆည်းပေးပါသည်။ ၃ ခုမြောက် ဒေါင်းလုဒ်လုပ်ပါက အသုံးနည်းဆုံး ၁ ခုကို အလိုအလျောက် ဖျက်ပေးပါမည်။',
    composerHeading: 'စာသားရိုက်ထည့်ရန်',
    copyText: 'ကူးယူမည်',
    copied: 'ကူးပြီး!',
    clearText: 'ရှင်းလင်းမည်',
    samplesLabel: 'နမူနာစာသား:',
    sampleGreeting: 'မင်္ဂလာပါ',
    sampleNews: 'သတင်းတို',
    sampleNumbers: 'ကိန်းဂဏန်းများ',
    sampleBilingual: 'မြန်မာ + English',
    controlsHeading: 'အသံအမြန်နှုန်း ထိန်းချုပ်မှု',
    defaultBtn: 'ပုံမှန် (1.0x · 65%)',
    lock: 'လော့ခ်ချ',
    locked: 'လော့ခ်ချထား',
    speedLabel: 'စကားပြောနှုန်း (Speed)',
    expressLabel: 'အသံခံစားချက် (Expressiveness)',
    calm: 'တည်ငြိမ်',
    balanced: 'ပုံမှန်',
    vivid: 'သွက်လက်',
    normalizeNumbers: 'ဂဏန်းနှင့် ငွေကြေးများကို မြန်မာအသံထွက်အဖြစ် ပြောင်းဖတ်မည် (ဥပမာ- ၂,၅၀၀ ကျပ်)',
    generateBtn: 'ဖတ်ပါ (Generate & Play)',
    generatingBtn: 'စက်ထဲတွင် အသံဖန်တီးနေပါသည်...',
    downloadingBtn: 'အသံမော်ဒယ် ဒေါင်းလုဒ်ဆွဲနေပါသည်...',
    noAudioYet: 'အသံဖိုင် မထုတ်ရသေးပါ',
    downloadWav: 'WAV ဒေါင်းမည်',
    downloadMp3: 'MP3 ဒေါင်းမည်',
    downloadSrt: 'SRT ဒေါင်းမည်',
    historyHeading: 'ဖန်တီးခဲ့သည့် မှတ်တမ်း (History)',
    historySub: 'ဘရောက်ဇာထဲတွင်သာ သိမ်းဆည်းထားသည် · ပြန်နားဆင်ရန် သို့မဟုတ် WAV / MP3 / SRT ထုတ်ယူရန်',
    clearHistory: 'မှတ်တမ်းအားလုံးဖျက်မည်',
    historyEmpty: 'မှတ်တမ်း မရှိသေးပါ။ အသံထုတ်လုပ်ပြီးပါက ဤနေရာတွင် အလိုအလျောက် သိမ်းဆည်းပေးပါမည်။',
    loadAndPlay: 'နားဆင်မည်',
    loadText: 'စာသားထည့်မည်',
    mp3Short: 'MP3',
    deleteItem: 'ဖျက်မည်',
    privacyNote: '၁၀၀% စက်တွင်းလုပ်ဆောင်မှု — သင့်စာသားများ အပြင်သို့ လုံးဝမထွက်ပါ။',
    offlineReady: 'အော့ဖ်လိုင်း အသင့်ရှိ',
    downloadRequired: 'ဒေါင်းလုဒ်ဆွဲရန် လိုအပ်',
    cancel: 'ပယ်ဖျက်မည်',
    close: 'ပိတ်မည်',
    iosModalTitle: 'iPhone / iPad တွင် MM TTS FREE ထည့်သွင်းနည်း',
    iosStep1: 'Safari အောက်ခြေရှိ Share ခလုတ်ကို နှိပ်ပါ။',
    iosStep2: 'အောက်သို့ဆွဲချပြီး "Add to Home Screen" ကို နှိပ်ပါ။',
    iosStep3: '"Add" ကို နှိပ်၍ အော့ဖ်လိုင်း အသုံးပြုနိုင်ပါပြီ။',
    storagePersisted: 'အမြဲတမ်း Storage',
    storageTemporary: 'ပုံမှန် Storage',
  },
};

const SAMPLE_TEXTS = {
  greeting:
    'မင်္ဂလာပါ။ MM TTS FREE မှ ကြိုဆိုပါတယ်။ သင့်စက်ထဲတွင်သာ အင်တာနက်မလိုဘဲ မြန်မာအသံထွက်များကို တိုက်ရိုက် ဖန်တီးနားဆင်နိုင်ပါသည်။',
  news:
    'ယနေ့ နံနက်ပိုင်းတွင် ရန်ကုန်မြို့နှင့် မန္တလေးမြို့တို့၌ ရာသီဥတု သာယာကြည်လင်လျက်ရှိပြီး ပြည်သူများ အေးချမ်းစွာ သွားလာလုပ်ကိုင်နေကြပါသည်။',
  numbers:
    'မြန်မာ့လက်ဖက်ရည် တစ်ခွက်လျှင် 1,500 Ks ဖြစ်ပြီး မုန့်ဟင်းခါး နှစ်ပွဲအတွက် စုစုပေါင်း 4,500 ကျပ် ကျသင့်ပါမည်။ ဝန်ဆောင်ခ 5% ထပ်ဆောင်းပါမည်။',
  bilingual:
    'မင်္ဂလာပါ။ ကျွန်တော်တို့ရဲ့ MM TTS FREE app ဟာ ONNX Runtime Web ကို အသုံးပြုထားပြီး browser ထဲမှာတင် offline အလုပ်လုပ်ပါတယ်။',
};

// ============================================================================
// 3. ModelCacheManager (Browser Cache API + localStorage LRU Eviction max 2)
// ============================================================================
class ModelCacheManager {
  constructor(cacheName = MODEL_CACHE_NAME, maxVoices = MAX_CACHED_VOICES) {
    this.cacheName = cacheName;
    this.maxVoices = maxVoices;
    this.activeController = null;
  }

  getMetadata() {
    try {
      const raw = localStorage.getItem(META_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }

  saveMetadata(meta) {
    try {
      localStorage.setItem(META_STORAGE_KEY, JSON.stringify(meta));
    } catch {
      // Ignore storage quota warning for small metadata JSON
    }
  }

  async requestPersistence() {
    if (navigator.storage && typeof navigator.storage.persist === 'function') {
      try {
        const persisted = await navigator.storage.persist();
        return persisted;
      } catch {
        return false;
      }
    }
    return false;
  }

  async verifyCachedVoices() {
    const meta = this.getMetadata();
    const cache = await caches.open(this.cacheName);
    let changed = false;

    for (const voiceId of Object.keys(meta)) {
      const preset = VOICE_PRESETS.find((v) => v.id === voiceId);
      if (!preset) {
        delete meta[voiceId];
        changed = true;
        continue;
      }
      const [vJson, enc, dec] = await Promise.all([
        cache.match(preset.voiceJsonUrl),
        cache.match(preset.encoderUrl),
        cache.match(preset.decoderUrl),
      ]);
      if (!vJson || !enc || !dec) {
        delete meta[voiceId];
        changed = true;
      }
    }

    if (changed) this.saveMetadata(meta);
    return meta;
  }

  isVoiceCached(voiceId) {
    const meta = this.getMetadata();
    return Boolean(meta[voiceId]);
  }

  getCachedList() {
    const meta = this.getMetadata();
    return Object.entries(meta)
      .map(([id, info]) => ({ id, ...info }))
      .sort((a, b) => a.lastUsedAt - b.lastUsedAt);
  }

  touchVoice(voiceId) {
    const meta = this.getMetadata();
    if (meta[voiceId]) {
      meta[voiceId].lastUsedAt = Date.now();
      this.saveMetadata(meta);
    }
  }

  async evictLRUIfNeeded(incomingVoiceId) {
    const meta = this.getMetadata();
    if (meta[incomingVoiceId]) return null;

    const entries = Object.entries(meta).sort(
      ([, a], [, b]) => (a.lastUsedAt || 0) - (b.lastUsedAt || 0)
    );

    const evictedNames = [];
    const cache = await caches.open(this.cacheName);

    while (entries.length >= this.maxVoices) {
      const [oldestId] = entries.shift();
      const preset = VOICE_PRESETS.find((v) => v.id === oldestId);
      if (preset) {
        await Promise.all([
          cache.delete(preset.voiceJsonUrl),
          cache.delete(preset.encoderUrl),
          cache.delete(preset.decoderUrl),
        ]);
        evictedNames.push(preset.nameEn);
      }
      delete meta[oldestId];
    }

    this.saveMetadata(meta);
    return evictedNames.length ? evictedNames : null;
  }

  async downloadVoice(preset, onProgress) {
    if (!navigator.onLine) {
      throw new Error('Cannot download voice model while offline.');
    }

    await this.requestPersistence();
    const evicted = await this.evictLRUIfNeeded(preset.id);

    const cache = await caches.open(this.cacheName);
    const controller = new AbortController();
    this.activeController = controller;

    const files = [
      { url: preset.voiceJsonUrl, type: 'application/json', fallbackSize: 41000 },
      { url: preset.encoderUrl, type: 'application/octet-stream', fallbackSize: 78268090 },
      { url: preset.decoderUrl, type: 'application/octet-stream', fallbackSize: 111686677 },
    ];

    // Opportunistically fetch & cache mined_rules.tsv for enhanced voicing rules
    if (preset.minedRulesUrl) {
      fetch(preset.minedRulesUrl, { signal: controller.signal })
        .then((r) => (r.ok ? cache.put(preset.minedRulesUrl, r) : null))
        .catch(() => {});
    }

    let totalEstimatedBytes = files.reduce((acc, f) => acc + f.fallbackSize, 0);
    let cumulativeLoaded = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const response = await fetch(file.url, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} while fetching ${file.url}`);
        }

        const contentLenHeader = response.headers.get('content-length');
        const fileTotal = contentLenHeader ? parseInt(contentLenHeader, 10) : file.fallbackSize;
        if (contentLenHeader) {
          totalEstimatedBytes =
            totalEstimatedBytes - file.fallbackSize + fileTotal;
        }

        const reader = response.body.getReader();
        const chunks = [];
        let fileLoaded = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          fileLoaded += value.byteLength;

          const currentLoaded = cumulativeLoaded + fileLoaded;
          const pct = Math.min(99, Math.round((currentLoaded / totalEstimatedBytes) * 100));
          onProgress({
            percent: pct,
            loadedMB: currentLoaded / (1024 * 1024),
            totalMB: totalEstimatedBytes / (1024 * 1024),
            evicted,
          });
        }

        cumulativeLoaded += fileLoaded;
        const blob = new Blob(chunks, { type: file.type });
        const cacheResp = new Response(blob, {
          status: 200,
          statusText: 'OK',
          headers: {
            'Content-Type': file.type,
            'Content-Length': String(blob.size),
          },
        });
        await cache.put(file.url, cacheResp);
      }

      const meta = this.getMetadata();
      meta[preset.id] = {
        lastUsedAt: Date.now(),
        sizeBytes: cumulativeLoaded,
      };
      this.saveMetadata(meta);

      onProgress({
        percent: 100,
        loadedMB: cumulativeLoaded / (1024 * 1024),
        totalMB: cumulativeLoaded / (1024 * 1024),
        evicted,
      });
    } catch (err) {
      // Clean up partial files on abort/error
      await Promise.all([
        cache.delete(preset.voiceJsonUrl),
        cache.delete(preset.encoderUrl),
        cache.delete(preset.decoderUrl),
      ]).catch(() => {});
      throw err;
    } finally {
      this.activeController = null;
    }
  }

  cancelActiveDownload() {
    if (this.activeController) {
      this.activeController.abort();
      this.activeController = null;
    }
  }

  async clearAll() {
    await caches.delete(this.cacheName);
    localStorage.removeItem(META_STORAGE_KEY);
  }
}

// ============================================================================
// 4. Pure JavaScript WAV & MP3 Audio Encoders
// ============================================================================

/**
 * Pure JavaScript PCM WAV encoder writing a standard 44-byte RIFF/WAVE header.
 * @param {Float32Array} samples Mono audio samples in [-1, 1]
 * @param {number} sampleRate Sample rate in Hz (24000 for nara-mini-v1)
 * @returns {Blob}
 */
function encodeWAV(samples, sampleRate = 24000) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeASCII = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor
  writeASCII(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeASCII(8, 'WAVE');

  // fmt sub-chunk
  writeASCII(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // ByteRate
  view.setUint16(32, numChannels * bytesPerSample, true); // BlockAlign
  view.setUint16(34, bitsPerSample, true);

  // data sub-chunk
  writeASCII(36, 'data');
  view.setUint32(40, dataSize, true);

  const isLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (isLittleEndian) {
    const pcm16 = new Int16Array(buffer, 44, samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const clamped = s > 1 ? 1 : s < -1 ? -1 : s;
      pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
  } else {
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Encodes Float32Array PCM samples to a 128kbps MP3 Blob using lamejs.
 * Yields periodically so the main thread stays smooth on mobile devices.
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @returns {Promise<Blob>}
 */
async function encodeMP3(samples, sampleRate = 24000) {
  if (typeof window.lamejs === 'undefined' || !window.lamejs.Mp3Encoder) {
    throw new Error('MP3 encoder (lamejs) failed to load.');
  }

  const encoder = new window.lamejs.Mp3Encoder(1, sampleRate, 128);
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  const mp3Chunks = [];
  const blockSize = 1152;
  for (let i = 0; i < int16.length; i += blockSize) {
    const chunk = int16.subarray(i, i + blockSize);
    const encoded = encoder.encodeBuffer(chunk);
    if (encoded.length > 0) {
      mp3Chunks.push(new Uint8Array(encoded));
    }
    if ((i / blockSize) % 60 === 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  const endBuf = encoder.flush();
  if (endBuf.length > 0) {
    mp3Chunks.push(new Uint8Array(endBuf));
  }

  return new Blob(mp3Chunks, { type: 'audio/mpeg' });
}

function triggerFileDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ============================================================================
// 4B. SRT Subtitle Formatter & Synthesis History Manager (localStorage + IDB)
// ============================================================================
const MAX_TEXT_CHARS = 10000;
const HISTORY_STORAGE_KEY = 'mm-tts-history-v1';
const HISTORY_IDB_NAME = 'mm-tts-history-db';
const HISTORY_IDB_STORE = 'audioBlobs';
const MAX_HISTORY_ITEMS = 20;
const MAX_HISTORY_AUDIO_BLOBS = 10;

function formatSrtTimestamp(totalSeconds) {
  const totalMs = Math.max(0, Math.round((totalSeconds || 0) * 1000));
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const seconds = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return (
    `${String(hours).padStart(2, '0')}:` +
    `${String(minutes).padStart(2, '0')}:` +
    `${String(seconds).padStart(2, '0')},` +
    `${String(ms).padStart(3, '0')}`
  );
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

function splitTextForSrtFallback(text, maxChars = 56) {
  const rawSentences = text.trim().split(/(?<=[။…\n])\s*|(?<=[.!?])(?:\s+|$)/);
  const sentences = [];

  for (const seg of rawSentences) {
    const trimmed = seg.trim();
    if (!trimmed) continue;
    if (sentences.length > 0 && sentences[sentences.length - 1].length <= 4) {
      sentences[sentences.length - 1] = `${sentences[sentences.length - 1]} ${trimmed}`;
    } else {
      sentences.push(trimmed);
    }
  }

  const units = [];
  for (const sentence of sentences) {
    const clauses = sentence.split(/(?<=၊)\s*/);
    for (const clause of clauses) {
      const subLines = splitSubtitleLines(clause, maxChars);
      for (let i = 0; i < subLines.length; i++) {
        units.push({
          text: subLines[i],
          gapBeforeSec: units.length === 0 ? 0 : i === 0 ? 0.25 : 0.01,
        });
      }
    }
  }
  return units;
}

function estimateCuesFromText(text, speed = 1.0) {
  const pieces = splitTextForSrtFallback(text, 56);
  const charsPerSec = 13.5 * Math.max(0.7, Math.min(1.5, speed));
  const cues = [];
  let cursor = 0;

  for (const piece of pieces) {
    cursor += piece.gapBeforeSec;
    const duration = Math.max(0.9, piece.text.length / charsPerSec);
    cues.push({
      startSec: cursor,
      endSec: cursor + duration,
      text: piece.text,
    });
    cursor += duration;
  }
  return cues;
}

function normalizeSrtCues(cues) {
  const normalized = [];

  for (const cue of cues) {
    const rawText = (cue && cue.text ? cue.text : '').trim();
    if (!rawText) continue;

    const parts = splitTextForSrtFallback(rawText, 56);
    const startSec = Math.max(0, Number(cue.startSec) || 0);
    const endSec = Math.max(startSec + 0.15, Number(cue.endSec) || startSec + 1.0);

    if (parts.length <= 1) {
      normalized.push({
        startSec,
        endSec,
        text: parts.length === 1 ? parts[0].text : rawText,
      });
      continue;
    }

    const totalCueDur = endSec - startSec;
    const rawGapSum = parts.slice(1).reduce((sum, p) => sum + p.gapBeforeSec, 0);
    const gapScale = rawGapSum > totalCueDur * 0.25 && rawGapSum > 0 ? (totalCueDur * 0.25) / rawGapSum : 1;
    const usableDur = Math.max(0.15 * parts.length, totalCueDur - rawGapSum * gapScale);
    const weights = parts.map((p) => Math.max(1, p.text.replace(/\s+/g, '').length));
    const totalWeight = weights.reduce((a, b) => a + b, 0);

    let cursor = startSec;
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) {
        cursor += parts[i].gapBeforeSec * gapScale;
      }
      const isLast = i === parts.length - 1;
      const segDur = isLast ? Math.max(0.15, endSec - cursor) : usableDur * (weights[i] / totalWeight);
      const segEnd = isLast ? Math.max(cursor + 0.15, endSec) : Math.min(endSec, cursor + segDur);
      normalized.push({
        startSec: cursor,
        endSec: segEnd,
        text: parts[i].text,
      });
      cursor = segEnd;
    }
  }

  return normalized;
}

function buildSrtContent(cues) {
  const normalized = normalizeSrtCues(cues);
  let prevEndMs = 0;
  return normalized
    .map((cue, index) => {
      const rawStartMs = Math.max(0, Math.round((cue.startSec || 0) * 1000));
      const startMs = index === 0 ? rawStartMs : Math.max(rawStartMs, prevEndMs + 10);
      const rawEndMs = Math.max(startMs + 150, Math.round((cue.endSec || 0) * 1000));
      prevEndMs = rawEndMs;
      return `${index + 1}\n${formatSrtTimestamp(startMs / 1000)} --> ${formatSrtTimestamp(rawEndMs / 1000)}\n${cue.text.trim()}`;
    })
    .join('\n\n')
    .concat('\n');
}

function parseWavToFloat32(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const sampleRate = view.getUint32(24, true) || 24000;
  const dataBytes = arrayBuffer.byteLength - 44;
  const numSamples = Math.max(0, Math.floor(dataBytes / 2));
  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const int16 = view.getInt16(44 + i * 2, true);
    samples[i] = int16 < 0 ? int16 / 0x8000 : int16 / 0x7fff;
  }
  return { samples, sampleRate };
}

class HistoryManager {
  constructor() {
    this.dbPromise = null;
  }

  openDB() {
    if (this.dbPromise) return this.dbPromise;
    this.dbPromise = new Promise((resolve) => {
      if (!('indexedDB' in window)) {
        resolve(null);
        return;
      }
      const req = indexedDB.open(HISTORY_IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(HISTORY_IDB_STORE)) {
          db.createObjectStore(HISTORY_IDB_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
    return this.dbPromise;
  }

  getItems() {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  saveItems(items) {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
    } catch {
      // Trim oldest items if localStorage quota is tight
      const trimmed = items.slice(0, Math.max(5, Math.floor(items.length / 2)));
      try {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmed));
      } catch {}
    }
  }

  async putAudioBlob(id, blob) {
    const db = await this.openDB();
    if (!db) return false;
    return new Promise((resolve) => {
      const tx = db.transaction(HISTORY_IDB_STORE, 'readwrite');
      tx.objectStore(HISTORY_IDB_STORE).put(blob, id);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    });
  }

  async getAudioBlob(id) {
    const db = await this.openDB();
    if (!db) return null;
    return new Promise((resolve) => {
      const tx = db.transaction(HISTORY_IDB_STORE, 'readonly');
      const req = tx.objectStore(HISTORY_IDB_STORE).get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  }

  async deleteAudioBlob(id) {
    const db = await this.openDB();
    if (!db) return;
    await new Promise((resolve) => {
      const tx = db.transaction(HISTORY_IDB_STORE, 'readwrite');
      tx.objectStore(HISTORY_IDB_STORE).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  }

  async addEntry({ text, voiceId, speed, expressiveness, normalizeNumbers, durationSec, cues, wavBlob }) {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const hasAudio = wavBlob ? await this.putAudioBlob(id, wavBlob) : false;

    const entry = {
      id,
      text,
      voiceId,
      speed,
      expressiveness,
      normalizeNumbers,
      durationSec,
      cues,
      hasAudio,
      createdAt: Date.now(),
    };

    const items = [entry, ...this.getItems()];
    const kept = items.slice(0, MAX_HISTORY_ITEMS);
    const dropped = items.slice(MAX_HISTORY_ITEMS);

    for (const old of dropped) {
      await this.deleteAudioBlob(old.id);
    }

    // Keep only the newest MAX_HISTORY_AUDIO_BLOBS WAV blobs in IndexedDB
    let audioCount = 0;
    for (const item of kept) {
      if (item.hasAudio) {
        audioCount++;
        if (audioCount > MAX_HISTORY_AUDIO_BLOBS) {
          item.hasAudio = false;
          await this.deleteAudioBlob(item.id);
        }
      }
    }

    this.saveItems(kept);
    return kept;
  }

  async removeEntry(id) {
    const items = this.getItems().filter((item) => item.id !== id);
    this.saveItems(items);
    await this.deleteAudioBlob(id);
    return items;
  }

  async clearAll() {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
    const db = await this.openDB();
    if (!db) return;
    await new Promise((resolve) => {
      const tx = db.transaction(HISTORY_IDB_STORE, 'readwrite');
      tx.objectStore(HISTORY_IDB_STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  }
}

// ============================================================================
// 5. Application State & DOM Controller
// ============================================================================
const cacheManager = new ModelCacheManager();
const historyManager = new HistoryManager();
let ttsWorker = null;
let selectedVoiceId = localStorage.getItem('mm-tts-voice') || 'shoon';
let currentLang = localStorage.getItem('mm-tts-lang') || 'en';
let currentTheme = localStorage.getItem('mm-tts-theme') || 'light';

let isDownloading = false;
let isSynthesizing = false;

let lastSamples = null;
let lastSampleRate = 24000;
let lastWavBlob = null;
let lastAudioUrl = null;
let lastCues = [];
let lastSynthesizedText = '';

const audioEl = new Audio();
let isSeeking = false;
let deferredInstallPrompt = null;

const el = {
  connectivityStatus: document.getElementById('connectivity-status'),
  connectivityLabel: document.getElementById('connectivity-label'),
  mobileConnectivityLabel: document.getElementById('mobile-connectivity-label'),
  headerCacheCount: document.getElementById('header-cache-count'),
  mobileCacheCount: document.getElementById('mobile-cache-count'),
  btnClearCache: document.getElementById('btn-clear-cache'),
  btnMobileClearCache: document.getElementById('btn-mobile-clear-cache'),
  btnLangToggle: document.getElementById('btn-lang-toggle'),
  langLabel: document.getElementById('lang-label'),
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  iconMoon: document.getElementById('icon-moon'),
  iconSun: document.getElementById('icon-sun'),
  metaThemeColor: document.getElementById('meta-theme-color'),
  btnInstallPwa: document.getElementById('btn-install-pwa'),
  iosModal: document.getElementById('ios-install-modal'),
  btnCloseIosModal: document.getElementById('btn-close-ios-modal'),
  storagePersistStatus: document.getElementById('storage-persist-status'),

  voiceList: document.getElementById('voice-list'),
  downloadBox: document.getElementById('download-progress-box'),
  downloadTitle: document.getElementById('download-title'),
  downloadPercent: document.getElementById('download-percent'),
  downloadProgressBar: document.getElementById('download-progressbar'),
  downloadProgressFill: document.getElementById('download-progress-fill'),
  downloadBytes: document.getElementById('download-bytes'),
  btnCancelDownload: document.getElementById('btn-cancel-download'),

  activeVoiceSummary: document.getElementById('active-voice-summary'),
  btnCopyText: document.getElementById('btn-copy-text'),
  btnClearText: document.getElementById('btn-clear-text'),
  ttsInput: document.getElementById('tts-input'),
  charCounter: document.getElementById('char-counter'),
  chunkCounter: document.getElementById('chunk-counter'),

  btnDefaultControls: document.getElementById('btn-default-controls'),
  speedSlider: document.getElementById('speed-slider'),
  speedValue: document.getElementById('speed-value'),
  btnLockSpeed: document.getElementById('btn-lock-speed'),
  lockSpeedLabel: document.getElementById('lock-speed-label'),
  expressSlider: document.getElementById('express-slider'),
  expressValue: document.getElementById('express-value'),
  btnLockExpress: document.getElementById('btn-lock-express'),
  lockExpressLabel: document.getElementById('lock-express-label'),
  chkNormalizeNumbers: document.getElementById('chk-normalize-numbers'),

  btnGenerate: document.getElementById('btn-generate'),
  generateParticles: document.getElementById('generate-particles'),
  generateBtnLabel: document.getElementById('generate-btn-label'),
  synthStatus: document.getElementById('synth-status'),
  synthStatusText: document.getElementById('synth-status-text'),
  synthStatusPct: document.getElementById('synth-status-pct'),

  appAlert: document.getElementById('app-alert'),
  appAlertText: document.getElementById('app-alert-text'),
  btnDismissAlert: document.getElementById('btn-dismiss-alert'),

  btnPlayerPlay: document.getElementById('btn-player-play'),
  playerIconPlay: document.getElementById('player-icon-play'),
  playerIconPause: document.getElementById('player-icon-pause'),
  btnPlayerReplay: document.getElementById('btn-player-replay'),
  playerTrackTitle: document.getElementById('player-track-title'),
  playerTime: document.getElementById('player-time'),
  playerSeek: document.getElementById('player-seek'),

  btnDownloadWav: document.getElementById('btn-download-wav'),
  btnDownloadMp3: document.getElementById('btn-download-mp3'),
  btnDownloadSrt: document.getElementById('btn-download-srt'),
  mp3BtnLabel: document.getElementById('mp3-btn-label'),

  historyList: document.getElementById('history-list'),
  historyEmpty: document.getElementById('history-empty'),
  btnClearHistory: document.getElementById('btn-clear-history'),
};

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function showAlert(message) {
  el.appAlertText.textContent = message;
  el.appAlert.classList.remove('hidden');
}

function hideAlert() {
  el.appAlert.classList.add('hidden');
}

const CONTROLS_STORAGE_KEY = 'mm-tts-controls-v1';
const DEFAULT_SPEED = 1.0;
const DEFAULT_EXPRESSIVENESS = 0.65;

let controlsState = {
  speed: DEFAULT_SPEED,
  expressiveness: DEFAULT_EXPRESSIVENESS,
  speedLocked: false,
  expressLocked: false,
};

function loadControlsState() {
  try {
    const raw = localStorage.getItem(CONTROLS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      controlsState = {
        speed: typeof parsed.speed === 'number' ? parsed.speed : DEFAULT_SPEED,
        expressiveness:
          typeof parsed.expressiveness === 'number' ? parsed.expressiveness : DEFAULT_EXPRESSIVENESS,
        speedLocked: Boolean(parsed.speedLocked),
        expressLocked: Boolean(parsed.expressLocked),
      };
    }
  } catch {}
}

function saveControlsState() {
  try {
    localStorage.setItem(CONTROLS_STORAGE_KEY, JSON.stringify(controlsState));
  } catch {}
}

function syncControlsUI() {
  el.speedSlider.value = String(controlsState.speed);
  el.speedValue.textContent = `${controlsState.speed.toFixed(2)}x`;
  el.speedSlider.disabled = controlsState.speedLocked;

  el.expressSlider.value = String(controlsState.expressiveness);
  el.expressValue.textContent = `${Math.round(controlsState.expressiveness * 100)}%`;
  el.expressSlider.disabled = controlsState.expressLocked;

  const dict = I18N[currentLang];

  el.btnLockSpeed.classList.toggle('is-locked', controlsState.speedLocked);
  el.btnLockSpeed.setAttribute('aria-pressed', String(controlsState.speedLocked));
  el.btnLockSpeed.querySelector('.lock-icon-open').classList.toggle('hidden', controlsState.speedLocked);
  el.btnLockSpeed.querySelector('.lock-icon-closed').classList.toggle('hidden', !controlsState.speedLocked);
  el.lockSpeedLabel.textContent = controlsState.speedLocked ? dict.locked : dict.lock;

  el.btnLockExpress.classList.toggle('is-locked', controlsState.expressLocked);
  el.btnLockExpress.setAttribute('aria-pressed', String(controlsState.expressLocked));
  el.btnLockExpress.querySelector('.lock-icon-open').classList.toggle('hidden', controlsState.expressLocked);
  el.btnLockExpress.querySelector('.lock-icon-closed').classList.toggle('hidden', !controlsState.expressLocked);
  el.lockExpressLabel.textContent = controlsState.expressLocked ? dict.locked : dict.lock;
}

async function copyTextWithFeedback(text, buttonEl, restoreText) {
  if (!text) return;
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text);
    } else {
      const temp = document.createElement('textarea');
      temp.value = text;
      temp.setAttribute('readonly', '');
      temp.style.position = 'fixed';
      temp.style.opacity = '0';
      document.body.appendChild(temp);
      temp.select();
      document.execCommand('copy');
      temp.remove();
    }
    if (buttonEl) {
      buttonEl.textContent = I18N[currentLang].copied;
      setTimeout(() => {
        buttonEl.textContent = restoreText();
      }, 1500);
    }
  } catch {
    showAlert('Clipboard copy permission denied.');
  }
}

function applyTheme(theme) {
  currentTheme = theme === 'dark' ? 'dark' : 'light';
  localStorage.setItem('mm-tts-theme', currentTheme);
  document.documentElement.setAttribute('data-theme', currentTheme);
  document.documentElement.classList.toggle('dark', currentTheme === 'dark');
  el.iconMoon.classList.toggle('hidden', currentTheme === 'dark');
  el.iconSun.classList.toggle('hidden', currentTheme !== 'dark');
  el.metaThemeColor.setAttribute('content', currentTheme === 'dark' ? '#0b0f14' : '#eab308');
}

function applyLanguage(lang) {
  currentLang = lang === 'my' ? 'my' : 'en';
  localStorage.setItem('mm-tts-lang', currentLang);
  document.documentElement.lang = currentLang;

  const dict = I18N[currentLang];
  document.querySelectorAll('[data-i18n]').forEach((node) => {
    const key = node.getAttribute('data-i18n');
    if (dict[key]) node.textContent = dict[key];
  });

  el.langLabel.textContent = currentLang === 'en' ? 'EN / မြန်မာ' : 'မြန်မာ / EN';
  if (!isSynthesizing && !isDownloading) {
    el.generateBtnLabel.textContent = dict.generateBtn;
  }

  syncControlsUI();
  updateConnectivityUI();
  renderVoiceCards();
  updateActiveVoiceSummary();
  renderHistory();
}

function updateConnectivityUI() {
  const online = navigator.onLine;
  const dict = I18N[currentLang];
  const label = online ? dict.online : dict.offline;
  el.connectivityLabel.textContent = label;
  el.mobileConnectivityLabel.textContent = label;
  el.connectivityStatus.classList.toggle('is-offline', !online);
}

function updateCacheCountUI() {
  const cachedCount = cacheManager.getCachedList().length;
  const text =
    currentLang === 'my'
      ? `သိမ်းထားသောအသံ: ${cachedCount}/${MAX_CACHED_VOICES}`
      : `Cached Voices: ${cachedCount}/${MAX_CACHED_VOICES}`;
  el.headerCacheCount.textContent = text;
  el.mobileCacheCount.textContent = text;
}

function updateActiveVoiceSummary() {
  const preset = VOICE_PRESETS.find((v) => v.id === selectedVoiceId) || VOICE_PRESETS[0];
  const name = currentLang === 'my' ? preset.nameMy : preset.nameEn;
  const cached = cacheManager.isVoiceCached(preset.id);
  const statusStr = cached ? I18N[currentLang].offlineReady : I18N[currentLang].downloadRequired;
  el.activeVoiceSummary.textContent = `${name} · ${statusStr} · 24 kHz Mono`;
}

function renderVoiceCards() {
  const dict = I18N[currentLang];
  el.voiceList.innerHTML = '';

  VOICE_PRESETS.forEach((preset) => {
    const isSelected = preset.id === selectedVoiceId;
    const isCached = cacheManager.isVoiceCached(preset.id);
    const sizeMB = (preset.approxBytes / (1024 * 1024)).toFixed(0);

    const card = document.createElement('button');
    card.type = 'button';
    card.className = `voice-card${isSelected ? ' is-selected' : ''}`;
    card.setAttribute('role', 'radio');
    card.setAttribute('aria-checked', String(isSelected));
    card.dataset.voiceId = preset.id;

    const statusIcon = isCached
      ? '<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>'
      : '<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';

    card.innerHTML = `
      <div class="voice-card-top">
        <span class="voice-name">${currentLang === 'my' ? preset.nameMy : preset.nameEn}</span>
        <span class="voice-status ${isCached ? 'is-cached' : 'is-remote'}">
          ${statusIcon}
          <span>${isCached ? dict.offlineReady : dict.downloadRequired}</span>
        </span>
      </div>
      <p class="voice-desc">${currentLang === 'my' ? preset.descMy : preset.descEn}</p>
      <div class="voice-meta-line">
        <span>${preset.gender}</span>
        <span class="tabular">~${sizeMB} MB</span>
      </div>
    `;

    card.addEventListener('click', () => handleSelectVoice(preset.id));
    el.voiceList.appendChild(card);
  });

  updateCacheCountUI();
  updateActiveVoiceSummary();
}

async function handleSelectVoice(voiceId) {
  if (isDownloading || isSynthesizing) return;
  selectedVoiceId = voiceId;
  localStorage.setItem('mm-tts-voice', voiceId);
  if (cacheManager.isVoiceCached(voiceId)) {
    cacheManager.touchVoice(voiceId);
    warmupSelectedVoice();
  }
  renderVoiceCards();
  hideAlert();

  // If the user selects an uncached voice while online, start downloading right away
  if (!cacheManager.isVoiceCached(voiceId) && navigator.onLine) {
    const preset = VOICE_PRESETS.find((v) => v.id === voiceId);
    if (preset) {
      await downloadVoiceWithUI(preset);
    }
  }
}

async function downloadVoiceWithUI(preset) {
  isDownloading = true;
  el.btnGenerate.disabled = true;
  el.generateBtnLabel.textContent = I18N[currentLang].downloadingBtn;
  el.downloadBox.classList.remove('hidden');
  el.downloadTitle.textContent = `${preset.nameEn} · Downloading...`;
  el.downloadPercent.textContent = '0%';
  el.downloadProgressFill.style.width = '0%';
  el.downloadProgressBar.setAttribute('aria-valuenow', '0');

  try {
    await cacheManager.downloadVoice(preset, ({ percent, loadedMB, totalMB, evicted }) => {
      el.downloadPercent.textContent = `${percent}%`;
      el.downloadProgressFill.style.width = `${percent}%`;
      el.downloadProgressBar.setAttribute('aria-valuenow', String(percent));
      el.downloadBytes.textContent = `${loadedMB.toFixed(1)} / ${totalMB.toFixed(1)} MB`;
      if (evicted && evicted.length) {
        el.downloadTitle.textContent = `${preset.nameEn} (LRU removed: ${evicted.join(', ')})`;
      }
    });
    renderVoiceCards();
    warmupSelectedVoice();
  } catch (err) {
    if (err.name === 'AbortError') {
      showAlert('Voice download cancelled.');
    } else {
      showAlert(`Voice download failed: ${err.message}`);
    }
    renderVoiceCards();
  } finally {
    isDownloading = false;
    el.downloadBox.classList.add('hidden');
    el.btnGenerate.disabled = false;
    el.generateBtnLabel.textContent = I18N[currentLang].generateBtn;
  }
}

function updateTextCounters() {
  const txt = el.ttsInput.value;
  el.charCounter.textContent = `${txt.length} / ${MAX_TEXT_CHARS} chars`;
  const trimmedLen = txt.trim().length;
  const roughSegments = trimmedLen ? Math.max(1, splitTextForSrtFallback(txt, 56).length) : 0;
  el.chunkCounter.textContent =
    roughSegments === 1 ? '1 line' : `${roughSegments} lines`;
  el.btnDownloadSrt.disabled = trimmedLen === 0 && lastCues.length === 0;
}

// ============================================================================
// 6. Web Worker Synthesis Execution & Background Warmup
// ============================================================================
function getWorker() {
  if (!ttsWorker) {
    ttsWorker = new Worker('/tts-worker.js?v=4', { type: 'module' });
  }
  return ttsWorker;
}

function warmupSelectedVoice() {
  const preset = VOICE_PRESETS.find((v) => v.id === selectedVoiceId);
  if (!preset || !cacheManager.isVoiceCached(preset.id)) return;
  try {
    getWorker().postMessage({
      id: `warmup-${preset.id}`,
      type: 'warmup',
      payload: { voiceConfig: preset },
    });
  } catch {}
}

function runWorkerSynthesis(payload) {
  return new Promise((resolve, reject) => {
    const worker = getWorker();
    const jobId = `${Date.now()}-${Math.random()}`;

    const onMessage = (event) => {
      const msg = event.data || {};
      if (msg.type === 'status') {
        el.synthStatusText.textContent = msg.message;
        return;
      }
      if (msg.type === 'progress') {
        const currentSentence = Math.min(msg.completedChunks + 1, msg.totalChunks);
        el.synthStatusText.textContent = `Synthesizing sentence ${currentSentence}/${msg.totalChunks}...`;
        el.synthStatusPct.textContent = `${msg.percent}%`;
        return;
      }
      if (msg.id !== jobId) return;

      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);

      if (msg.type === 'result') {
        resolve({
          audioBuffer: msg.audioBuffer,
          sampleRate: msg.sampleRate,
          cues: msg.cues || [],
        });
      } else if (msg.type === 'error') {
        reject(new Error(msg.error || 'Synthesis failed'));
      }
    };

    const onError = (err) => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      reject(new Error(err.message || 'Web Worker error'));
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage({ id: jobId, type: 'synthesize', payload });
  });
}

let particleTimer = null;
let waveTimer = null;
const YELLOW_PARTICLE_COLORS = ['#fef9c3', '#fef08a', '#fde047', '#facc15', '#eab308', '#f59e0b'];
const MAX_ACTIVE_PARTICLES = 90;

function triggerButtonWave(durationMs = 950) {
  if (!el.btnGenerate) return;
  el.btnGenerate.classList.add('is-wave-active');
  if (waveTimer) clearTimeout(waveTimer);
  waveTimer = setTimeout(() => {
    el.btnGenerate.classList.remove('is-wave-active');
    waveTimer = null;
  }, durationMs);
}

function spawnYellowParticle(isBurst = false) {
  if (!el.generateParticles) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (el.generateParticles.childElementCount >= MAX_ACTIVE_PARTICLES) return;

  const particle = document.createElement('span');
  particle.className = 'yellow-particle';
  const size = isBurst ? 4 + Math.random() * 5.5 : 3.5 + Math.random() * 5;
  const startLeft = 2 + Math.random() * 96;
  const dx = (Math.random() - 0.5) * (isBurst ? 76 : 56);
  const dy = 56 + Math.random() * 72;
  const dur = 750 + Math.random() * 650;
  const color = YELLOW_PARTICLE_COLORS[Math.floor(Math.random() * YELLOW_PARTICLE_COLORS.length)];

  particle.style.width = `${size.toFixed(1)}px`;
  particle.style.height = `${size.toFixed(1)}px`;
  particle.style.left = `${startLeft.toFixed(1)}%`;
  particle.style.backgroundColor = color;
  particle.style.setProperty('--dx', `${dx.toFixed(1)}px`);
  particle.style.setProperty('--dy', `${dy.toFixed(1)}px`);
  particle.style.setProperty('--dur', `${dur.toFixed(0)}ms`);

  particle.addEventListener('animationend', () => particle.remove(), { once: true });
  el.generateParticles.appendChild(particle);
}

function burstYellowParticles(count = 16) {
  for (let i = 0; i < count; i++) {
    spawnYellowParticle(true);
  }
}

function startYellowParticles() {
  stopYellowParticles();
  el.btnGenerate.classList.add('is-synthesizing');
  burstYellowParticles(18);
  particleTimer = setInterval(() => {
    const batch = 3 + Math.floor(Math.random() * 2);
    for (let i = 0; i < batch; i++) {
      spawnYellowParticle(false);
    }
  }, 55);
}

function stopYellowParticles() {
  if (particleTimer) {
    clearInterval(particleTimer);
    particleTimer = null;
  }
  el.btnGenerate.classList.remove('is-synthesizing');
}

async function handleGenerateAndPlay() {
  if (isDownloading || isSynthesizing) return;
  hideAlert();
  triggerButtonWave(950);
  burstYellowParticles(14);

  const text = el.ttsInput.value.trim();
  if (!text) {
    showAlert(
      currentLang === 'my'
        ? 'ကျေးဇူးပြု၍ စာသား ရိုက်ထည့်ပါ။'
        : 'Please enter Burmese or English text to synthesize.'
    );
    return;
  }

  const preset = VOICE_PRESETS.find((v) => v.id === selectedVoiceId) || VOICE_PRESETS[0];

  // Ensure voice is cached first (stream download with progress if needed)
  if (!cacheManager.isVoiceCached(preset.id)) {
    if (!navigator.onLine) {
      showAlert(
        currentLang === 'my'
          ? 'ဤအသံမော်ဒယ်ကို အော့ဖ်လိုင်း သိမ်းဆည်းထားခြင်း မရှိသေးပါ။ အွန်လိုင်းချိတ်ဆက်ပြီး ဒေါင်းလုဒ်ဆွဲပါ။'
          : 'This voice model is not cached yet. Connect online to download it (~181 MB).'
      );
      return;
    }
    await downloadVoiceWithUI(preset);
    if (!cacheManager.isVoiceCached(preset.id)) return;
  }

  cacheManager.touchVoice(preset.id);

  isSynthesizing = true;
  el.btnGenerate.disabled = true;
  el.generateBtnLabel.textContent = I18N[currentLang].generatingBtn;
  el.synthStatus.classList.remove('hidden');
  el.synthStatusText.textContent = 'Preparing text & WASM session...';
  el.synthStatusPct.textContent = '0%';
  startYellowParticles();

  try {
    const speed = parseFloat(el.speedSlider.value);
    const expressiveness = parseFloat(el.expressSlider.value);
    const normalizeNumbers = el.chkNormalizeNumbers.checked;

    const { audioBuffer, sampleRate, cues } = await runWorkerSynthesis({
      text,
      voiceConfig: preset,
      speed,
      expressiveness,
      normalizeNumbers,
    });

    lastSamples = audioBuffer;
    lastSampleRate = sampleRate;
    lastWavBlob = encodeWAV(audioBuffer, sampleRate);
    lastCues = cues.length ? cues : estimateCuesFromText(text, speed);
    lastSynthesizedText = text;

    if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
    lastAudioUrl = URL.createObjectURL(lastWavBlob);

    audioEl.src = lastAudioUrl;
    audioEl.load();

    el.btnPlayerPlay.disabled = false;
    el.btnPlayerReplay.disabled = false;
    el.playerSeek.disabled = false;
    el.btnDownloadWav.disabled = false;
    el.btnDownloadMp3.disabled = false;
    el.btnDownloadSrt.disabled = false;

    const voiceDisplay = currentLang === 'my' ? preset.nameMy : preset.nameEn;
    el.playerTrackTitle.textContent = `${voiceDisplay} · ${speed.toFixed(2)}x`;

    const durationSec = audioBuffer.length / sampleRate;
    await historyManager.addEntry({
      text,
      voiceId: preset.id,
      speed,
      expressiveness,
      normalizeNumbers,
      durationSec,
      cues: lastCues,
      wavBlob: lastWavBlob,
    });
    renderHistory();

    await audioEl.play().catch(() => {});
  } catch (err) {
    showAlert(`Synthesis error: ${err.message}`);
  } finally {
    stopYellowParticles();
    isSynthesizing = false;
    el.synthStatus.classList.add('hidden');
    el.btnGenerate.disabled = false;
    el.generateBtnLabel.textContent = I18N[currentLang].generateBtn;
  }
}

// ============================================================================
// 7. Audio Player Event Wiring
// ============================================================================
function syncPlayerIcons() {
  const playing = !audioEl.paused && !audioEl.ended;
  el.playerIconPlay.classList.toggle('hidden', playing);
  el.playerIconPause.classList.toggle('hidden', !playing);
}

audioEl.addEventListener('loadedmetadata', () => {
  el.playerTime.textContent = `${formatTime(0)} / ${formatTime(audioEl.duration)}`;
});

audioEl.addEventListener('timeupdate', () => {
  if (!isSeeking && Number.isFinite(audioEl.duration) && audioEl.duration > 0) {
    el.playerSeek.value = String(Math.round((audioEl.currentTime / audioEl.duration) * 1000));
    el.playerTime.textContent = `${formatTime(audioEl.currentTime)} / ${formatTime(audioEl.duration)}`;
  }
});

audioEl.addEventListener('play', syncPlayerIcons);
audioEl.addEventListener('pause', syncPlayerIcons);
audioEl.addEventListener('ended', () => {
  syncPlayerIcons();
  el.playerSeek.value = '0';
});

el.btnPlayerPlay.addEventListener('click', () => {
  if (!lastAudioUrl) return;
  if (audioEl.paused) {
    audioEl.play().catch(() => {});
  } else {
    audioEl.pause();
  }
});

el.btnPlayerReplay.addEventListener('click', () => {
  if (!lastAudioUrl) return;
  audioEl.currentTime = 0;
  audioEl.play().catch(() => {});
});

el.playerSeek.addEventListener('input', () => {
  isSeeking = true;
});

el.playerSeek.addEventListener('change', () => {
  if (Number.isFinite(audioEl.duration) && audioEl.duration > 0) {
    audioEl.currentTime = (Number(el.playerSeek.value) / 1000) * audioEl.duration;
  }
  isSeeking = false;
});

// ============================================================================
// 8. Export Handlers (WAV, MP3 & SRT) + History Renderer
// ============================================================================
el.btnDownloadWav.addEventListener('click', () => {
  if (!lastWavBlob) return;
  const filename = `mm-tts-${selectedVoiceId}-${Date.now()}.wav`;
  triggerFileDownload(lastWavBlob, filename);
});

el.btnDownloadMp3.addEventListener('click', async () => {
  if (!lastSamples) return;
  const dict = I18N[currentLang];
  el.btnDownloadMp3.disabled = true;
  el.mp3BtnLabel.textContent = 'Encoding MP3...';

  try {
    const mp3Blob = await encodeMP3(lastSamples, lastSampleRate);
    const filename = `mm-tts-${selectedVoiceId}-${Date.now()}.mp3`;
    triggerFileDownload(mp3Blob, filename);
  } catch (err) {
    showAlert(`MP3 export failed: ${err.message}`);
  } finally {
    el.btnDownloadMp3.disabled = false;
    el.mp3BtnLabel.textContent = dict.downloadMp3;
  }
});

el.btnDownloadSrt.addEventListener('click', () => {
  const text = el.ttsInput.value.trim();
  const speed = parseFloat(el.speedSlider.value);
  const cues =
    lastCues.length && text === lastSynthesizedText
      ? lastCues
      : text
        ? estimateCuesFromText(text, speed)
        : lastCues;

  if (!cues.length) return;
  const srtText = buildSrtContent(cues);
  const srtBlob = new Blob(['\ufeff', srtText], { type: 'application/x-subrip;charset=utf-8' });
  triggerFileDownload(srtBlob, `mm-tts-${selectedVoiceId}-${Date.now()}.srt`);
});

function formatRelativeDate(ts) {
  try {
    return new Date(ts).toLocaleString(currentLang === 'my' ? 'my-MM' : 'en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

function renderHistory() {
  const items = historyManager.getItems();
  const dict = I18N[currentLang];
  el.historyList.innerHTML = '';

  const hasItems = items.length > 0;
  el.historyEmpty.classList.toggle('hidden', hasItems);
  el.btnClearHistory.classList.toggle('hidden', !hasItems);

  items.forEach((item) => {
    const preset = VOICE_PRESETS.find((v) => v.id === item.voiceId) || VOICE_PRESETS[0];
    const voiceName = currentLang === 'my' ? preset.nameMy : preset.nameEn;

    const row = document.createElement('div');
    row.className = 'history-item';
    row.setAttribute('role', 'listitem');

    const body = document.createElement('div');
    body.className = 'history-item-body';

    const textP = document.createElement('p');
    textP.className = 'history-item-text';
    textP.textContent = item.text;

    const meta = document.createElement('div');
    meta.className = 'history-item-meta';
    meta.innerHTML = `
      <span>${voiceName}</span>
      <span aria-hidden="true">·</span>
      <span class="tabular">${formatTime(item.durationSec)}</span>
      <span aria-hidden="true">·</span>
      <span class="tabular">${Number(item.speed || 1).toFixed(2)}x</span>
      <span aria-hidden="true">·</span>
      <span class="tabular">${item.text.length} chars</span>
      <span aria-hidden="true">·</span>
      <span>${formatRelativeDate(item.createdAt)}</span>
    `;

    body.appendChild(textP);
    body.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'history-item-actions';

    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.className = 'btn-history-action is-primary';
    playBtn.textContent = item.hasAudio ? dict.loadAndPlay : dict.loadText;
    playBtn.addEventListener('click', () => loadHistoryEntry(item));

    const mp3Btn = document.createElement('button');
    mp3Btn.type = 'button';
    mp3Btn.className = 'btn-history-action';
    mp3Btn.textContent = dict.mp3Short;
    mp3Btn.disabled = !item.hasAudio;
    mp3Btn.addEventListener('click', async () => {
      if (!item.hasAudio) return;
      mp3Btn.disabled = true;
      mp3Btn.textContent = '...';
      try {
        const wavBlob = await historyManager.getAudioBlob(item.id);
        if (!wavBlob) {
          showAlert('Cached audio for this entry is no longer available.');
          return;
        }
        const buf = await wavBlob.arrayBuffer();
        const { samples, sampleRate } = parseWavToFloat32(buf);
        const mp3Blob = await encodeMP3(samples, sampleRate);
        triggerFileDownload(mp3Blob, `mm-tts-${item.voiceId}-${item.createdAt}.mp3`);
      } catch (err) {
        showAlert(`MP3 export failed: ${err.message}`);
      } finally {
        mp3Btn.disabled = !item.hasAudio;
        mp3Btn.textContent = I18N[currentLang].mp3Short;
      }
    });

    const srtBtn = document.createElement('button');
    srtBtn.type = 'button';
    srtBtn.className = 'btn-history-action';
    srtBtn.textContent = 'SRT';
    srtBtn.addEventListener('click', () => {
      const cues =
        Array.isArray(item.cues) && item.cues.length
          ? item.cues
          : estimateCuesFromText(item.text, item.speed || 1);
      const srtText = buildSrtContent(cues);
      const srtBlob = new Blob(['\ufeff', srtText], { type: 'application/x-subrip;charset=utf-8' });
      triggerFileDownload(srtBlob, `mm-tts-${item.voiceId}-${item.createdAt}.srt`);
    });

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'btn-history-action';
    copyBtn.textContent = dict.copyText;
    copyBtn.addEventListener('click', () => {
      copyTextWithFeedback(item.text, copyBtn, () => I18N[currentLang].copyText);
    });

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'btn-history-action is-danger';
    delBtn.textContent = dict.deleteItem;
    delBtn.addEventListener('click', async () => {
      await historyManager.removeEntry(item.id);
      renderHistory();
    });

    actions.appendChild(playBtn);
    actions.appendChild(mp3Btn);
    actions.appendChild(srtBtn);
    actions.appendChild(copyBtn);
    actions.appendChild(delBtn);

    row.appendChild(body);
    row.appendChild(actions);
    el.historyList.appendChild(row);
  });
}

async function loadHistoryEntry(item) {
  el.ttsInput.value = item.text;
  selectedVoiceId = item.voiceId;
  localStorage.setItem('mm-tts-voice', item.voiceId);
  if (!controlsState.speedLocked) {
    controlsState.speed = item.speed || DEFAULT_SPEED;
  }
  if (!controlsState.expressLocked) {
    controlsState.expressiveness = item.expressiveness ?? DEFAULT_EXPRESSIVENESS;
  }
  saveControlsState();
  syncControlsUI();
  el.chkNormalizeNumbers.checked = item.normalizeNumbers !== false;

  lastCues = Array.isArray(item.cues) ? item.cues : estimateCuesFromText(item.text, item.speed || 1);
  lastSynthesizedText = item.text;

  renderVoiceCards();
  updateTextCounters();

  if (item.hasAudio) {
    const wavBlob = await historyManager.getAudioBlob(item.id);
    if (wavBlob) {
      const buf = await wavBlob.arrayBuffer();
      const parsed = parseWavToFloat32(buf);
      lastSamples = parsed.samples;
      lastSampleRate = parsed.sampleRate;
      lastWavBlob = wavBlob;

      if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
      lastAudioUrl = URL.createObjectURL(wavBlob);

      audioEl.src = lastAudioUrl;
      audioEl.load();

      el.btnPlayerPlay.disabled = false;
      el.btnPlayerReplay.disabled = false;
      el.playerSeek.disabled = false;
      el.btnDownloadWav.disabled = false;
      el.btnDownloadMp3.disabled = false;
      el.btnDownloadSrt.disabled = false;

      const preset = VOICE_PRESETS.find((v) => v.id === item.voiceId) || VOICE_PRESETS[0];
      const voiceDisplay = currentLang === 'my' ? preset.nameMy : preset.nameEn;
      el.playerTrackTitle.textContent = `${voiceDisplay} · ${Number(item.speed || 1).toFixed(2)}x`;

      await audioEl.play().catch(() => {});
      return;
    }
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// 9. PWA Service Worker & In-App Install Flow
// ============================================================================
function initPWA() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker
        .register('/sw.js', { updateViaCache: 'none' })
        .then((reg) => reg.update().catch(() => {}))
        .catch(() => {});
    });
  }

  const isStandalone =
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;

  if (isStandalone) {
    el.btnInstallPwa.classList.add('hidden');
    return;
  }

  const isIOS = /iphone|ipad|ipod/.test(navigator.userAgent.toLowerCase());
  if (isIOS) {
    el.btnInstallPwa.classList.remove('hidden');
    el.btnInstallPwa.addEventListener('click', () => {
      el.iosModal.classList.remove('hidden');
    });
    el.btnCloseIosModal.addEventListener('click', () => {
      el.iosModal.classList.add('hidden');
    });
    return;
  }

  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    el.btnInstallPwa.classList.remove('hidden');
  });

  el.btnInstallPwa.addEventListener('click', async () => {
    if (!deferredInstallPrompt) return;
    await deferredInstallPrompt.prompt();
    const { outcome } = await deferredInstallPrompt.userChoice;
    if (outcome === 'accepted') {
      el.btnInstallPwa.classList.add('hidden');
    }
    deferredInstallPrompt = null;
  });

  window.addEventListener('appinstalled', () => {
    el.btnInstallPwa.classList.add('hidden');
    deferredInstallPrompt = null;
  });
}

// ============================================================================
// 10. Event Bindings & Bootstrap
// ============================================================================
async function clearCacheHandler() {
  if (isDownloading || isSynthesizing) return;
  await cacheManager.clearAll();
  if (ttsWorker) {
    ttsWorker.postMessage({ id: 'unload', type: 'unload' });
  }
  renderVoiceCards();
}

el.btnClearCache.addEventListener('click', clearCacheHandler);
el.btnMobileClearCache.addEventListener('click', clearCacheHandler);
el.btnCancelDownload.addEventListener('click', () => cacheManager.cancelActiveDownload());
el.btnDismissAlert.addEventListener('click', hideAlert);

el.btnClearHistory.addEventListener('click', async () => {
  await historyManager.clearAll();
  renderHistory();
});

el.btnLangToggle.addEventListener('click', () => {
  applyLanguage(currentLang === 'en' ? 'my' : 'en');
});

el.btnThemeToggle.addEventListener('click', () => {
  applyTheme(currentTheme === 'light' ? 'dark' : 'light');
});

el.btnCopyText.addEventListener('click', () => {
  copyTextWithFeedback(el.ttsInput.value, el.btnCopyText, () => I18N[currentLang].copyText);
});

el.btnClearText.addEventListener('click', () => {
  el.ttsInput.value = '';
  updateTextCounters();
  el.ttsInput.focus();
});

document.querySelectorAll('.btn-sample').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.sample;
    if (SAMPLE_TEXTS[key]) {
      el.ttsInput.value = SAMPLE_TEXTS[key];
      updateTextCounters();
    }
  });
});

el.ttsInput.addEventListener('input', updateTextCounters);

el.speedSlider.addEventListener('input', () => {
  if (controlsState.speedLocked) return;
  controlsState.speed = parseFloat(el.speedSlider.value);
  saveControlsState();
  el.speedValue.textContent = `${controlsState.speed.toFixed(2)}x`;
});

el.expressSlider.addEventListener('input', () => {
  if (controlsState.expressLocked) return;
  controlsState.expressiveness = parseFloat(el.expressSlider.value);
  saveControlsState();
  el.expressValue.textContent = `${Math.round(controlsState.expressiveness * 100)}%`;
});

el.btnLockSpeed.addEventListener('click', () => {
  controlsState.speedLocked = !controlsState.speedLocked;
  saveControlsState();
  syncControlsUI();
});

el.btnLockExpress.addEventListener('click', () => {
  controlsState.expressLocked = !controlsState.expressLocked;
  saveControlsState();
  syncControlsUI();
});

el.btnDefaultControls.addEventListener('click', () => {
  if (controlsState.speedLocked && controlsState.expressLocked) {
    showAlert(
      currentLang === 'my'
        ? 'Speed နှင့် Expressiveness ဘားများကို လော့ခ်ချထားပါသည်။ Default ပြန်ထားရန် လော့ခ်ဖြုတ်ပါ။'
        : 'Controls are locked. Unlock them to reset to default values.'
    );
    return;
  }
  if (!controlsState.speedLocked) {
    controlsState.speed = DEFAULT_SPEED;
  }
  if (!controlsState.expressLocked) {
    controlsState.expressiveness = DEFAULT_EXPRESSIVENESS;
  }
  saveControlsState();
  syncControlsUI();
});

el.btnGenerate.addEventListener('pointerdown', () => {
  if (!isDownloading && !isSynthesizing) {
    triggerButtonWave(750);
  }
});
el.btnGenerate.addEventListener('click', handleGenerateAndPlay);

window.addEventListener('online', updateConnectivityUI);
window.addEventListener('offline', updateConnectivityUI);

// Initialize application on startup
(async function bootstrap() {
  loadControlsState();
  applyTheme(currentTheme);
  await cacheManager.verifyCachedVoices();
  applyLanguage(currentLang);
  updateTextCounters();
  initPWA();
  warmupSelectedVoice();

  const persisted = await cacheManager.requestPersistence();
  el.storagePersistStatus.textContent = persisted
    ? I18N[currentLang].storagePersisted
    : I18N[currentLang].storageTemporary;
})();
