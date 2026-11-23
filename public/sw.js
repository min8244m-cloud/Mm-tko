/**
 * MM TTS FREE — Service Worker (sw.js)
 *
 * Caches the PWA App Shell (HTML, CSS, JS, Icons, Manifest, lamejs, and ONNX WASM
 * runtime files) for full offline execution.
 *
 * Note on AI Model Weights:
 * Large .onnx voice models (~181 MB per voice) are intentionally NOT intercepted
 * here. They are managed separately by ModelCacheManager using the Browser Cache
 * API ('mm-tts-models-v1') with streaming progress and a 2-voice LRU eviction policy.
 */

const APP_SHELL_CACHE = 'mm-tts-shell-v4';
const FONT_CACHE = 'mm-tts-fonts-v1';
const MODEL_CACHE = 'mm-tts-models-v1';

// Core assets required for 100% offline launch and synthesis
const CORE_SHELL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon.svg',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
  '/pwa-maskable-512x512.png',
  '/apple-touch-icon.png',
  '/tts-worker.js',
  '/vendor/lame.min.js',
  '/vendor/ort/ort.wasm.min.mjs',
  '/vendor/ort/ort-wasm-simd-threaded.mjs',
  '/vendor/ort/ort-wasm-simd-threaded.wasm',
];

// Optional unbundled/dev source paths cached opportunistically when present
const OPTIONAL_SHELL_ASSETS = [
  '/style.css',
  '/app.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(APP_SHELL_CACHE);

      // Cache critical assets; use allSettled so a single missing optional path
      // never blocks Service Worker activation across dev/prod builds.
      await Promise.allSettled(
        CORE_SHELL_ASSETS.map(async (url) => {
          const response = await fetch(url, { cache: 'reload' });
          if (response.ok) {
            await cache.put(url, response);
          }
        })
      );

      await Promise.allSettled(
        OPTIONAL_SHELL_ASSETS.map(async (url) => {
          const response = await fetch(url, { cache: 'reload' });
          if (response.ok) {
            await cache.put(url, response);
          }
        })
      );

      await self.skipWaiting();
    })()
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keepCaches = new Set([APP_SHELL_CACHE, FONT_CACHE, MODEL_CACHE]);
      const keys = await caches.keys();
      await Promise.all(
        keys.map((key) => (!keepCaches.has(key) ? caches.delete(key) : Promise.resolve()))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.method !== 'GET') {
    return;
  }

  const url = new URL(request.url);

  // Never intercept Hugging Face model downloads — handled by ModelCacheManager
  if (
    url.hostname.includes('huggingface.co') ||
    url.hostname.includes('hf.co') ||
    url.pathname.endsWith('.onnx')
  ) {
    return;
  }

  // Google Fonts (Noto Sans Myanmar, Plus Jakarta Sans, JetBrains Mono): CacheFirst
  if (
    url.hostname === 'fonts.googleapis.com' ||
    url.hostname === 'fonts.gstatic.com'
  ) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(FONT_CACHE);
        const cached = await cache.match(request);
        if (cached) return cached;
        try {
          const networkResponse = await fetch(request);
          if (networkResponse.ok || networkResponse.type === 'opaque') {
            cache.put(request, networkResponse.clone());
          }
          return networkResponse;
        } catch {
          return new Response('', { status: 504, statusText: 'Offline font unavailable' });
        }
      })()
    );
    return;
  }

  // Only handle same-origin app shell & bundled assets
  if (url.origin !== self.location.origin) {
    return;
  }

  // Navigation requests: Network-first with offline index.html fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        const cache = await caches.open(APP_SHELL_CACHE);
        try {
          const fresh = await fetch(request, { cache: 'no-cache' });
          if (fresh.ok) {
            cache.put('/index.html', fresh.clone());
          }
          return fresh;
        } catch {
          return (
            (await cache.match('/index.html')) ||
            (await cache.match('/')) ||
            new Response('Offline — MM TTS FREE shell unavailable', {
              status: 503,
              headers: { 'Content-Type': 'text/plain; charset=utf-8' },
            })
          );
        }
      })()
    );
    return;
  }

  // Immutable vendor binaries & icons: Cache-first
  if (
    url.pathname.startsWith('/vendor/') ||
    url.pathname.endsWith('.wasm') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.svg')
  ) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(APP_SHELL_CACHE);
        const cached = await cache.match(request, { ignoreSearch: true });
        if (cached) return cached;

        try {
          const networkResponse = await fetch(request);
          if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
          }
          return networkResponse;
        } catch {
          return new Response('Offline resource unavailable', { status: 503 });
        }
      })()
    );
    return;
  }

  // App code (JS, CSS, Worker, Manifest): Network-first so updates take effect immediately,
  // with full offline Cache fallback.
  event.respondWith(
    (async () => {
      const cache = await caches.open(APP_SHELL_CACHE);
      try {
        const networkResponse = await fetch(request, { cache: 'no-cache' });
        if (networkResponse.ok) {
          cache.put(url.pathname, networkResponse.clone());
        }
        return networkResponse;
      } catch {
        const cached =
          (await cache.match(url.pathname)) ||
          (await cache.match(request, { ignoreSearch: true }));
        return cached || new Response('Offline resource unavailable', { status: 503 });
      }
    })()
  );
});
