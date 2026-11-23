<script setup lang="ts">
import { ref, watch, computed, onBeforeUnmount, useTemplateRef } from 'vue'

const props = defineProps<{ file: File }>()
const emit = defineEmits<{ loaded: [duration: number] }>()

const audio = useTemplateRef<HTMLAudioElement>('audio')
const src = ref('')
const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const error = ref(false)
let raf = 0

// New file -> new blob URL; the old one is freed by onCleanup
// (onCleanup also runs when the component unmounts)
watch(
  () => props.file,
  (file, _old, onCleanup) => {
    const url = URL.createObjectURL(file)
    src.value = url
    isPlaying.value = false
    currentTime.value = 0
    duration.value = 0
    error.value = false
    onCleanup(() => URL.revokeObjectURL(url))
  },
  { immediate: true },
)

// `timeupdate` only fires ~4x per second, too slow for a 0.07 s bangu
// stroke, so follow the playhead every frame while playing
function tick() {
  if (audio.value) currentTime.value = audio.value.currentTime
  raf = requestAnimationFrame(tick)
}

function onLoaded() {
  const d = audio.value?.duration ?? 0
  duration.value = Number.isFinite(d) ? d : 0
  emit('loaded', duration.value)
}

function onPlay() {
  isPlaying.value = true
  raf = requestAnimationFrame(tick)
}
function onPause() {
  isPlaying.value = false
  cancelAnimationFrame(raf)
  if (audio.value) currentTime.value = audio.value.currentTime
}

// Round down while playing; at the end show the same (rounded-up) value as the total
const shownTime = computed(() =>
  currentTime.value >= duration.value ? Math.ceil(duration.value) : Math.floor(currentTime.value),
)

function onEnded() {
  currentTime.value = duration.value
}


async function toggle() {
  const el = audio.value
  if (!el) return
  if (!el.paused) return el.pause()
  if (el.ended) el.currentTime = 0
  try {
    await el.play()
  } catch {
    error.value = true // e.g. the browser cannot decode this file
  }
}

function seek(e: Event) {
  const t = Number((e.target as HTMLInputElement).value)
  if (audio.value) audio.value.currentTime = t
  currentTime.value = t
}

// Format seconds as mm:ss
const fmt = (s: number) => {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

onBeforeUnmount(() => cancelAnimationFrame(raf))
</script>

<template>
  <div class="w-full flex items-center gap-3">
    <audio
      ref="audio"
      :src="src"
      preload="metadata"
      @loadedmetadata="onLoaded"
      @play="onPlay"
      @pause="onPause"
      @ended="onEnded"
      @error="error = true"
    ></audio>

    <p v-if="error" class="text-sm text-red-600">Preview not available for this file.</p>

    <template v-else>
      <button
        type="button"
        class="size-9 flex-none grid place-items-center rounded-full bg-emerald-700 text-white cursor-pointer disabled:bg-gray-300"
        :disabled="!duration"
        :aria-label="isPlaying ? 'Pause' : 'Play'"
        @click="toggle"
      >
        <svg v-if="!isPlaying" viewBox="0 0 24 24" fill="currentColor" class="size-4" aria-hidden="true">
          <path d="M8 5v14l11-7z" />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor" class="size-4" aria-hidden="true">
          <path d="M7 5h4v14H7zM13 5h4v14h-4z" />
        </svg>
      </button>

      <input
        type="range"
        class="flex-1 min-w-0 accent-emerald-700"
        min="0"
        :max="duration"
        step="0.001"
        :value="currentTime"
        aria-label="Audio position"
        @input="seek"
      />

      <span class="flex-none text-xs tabular-nums text-gray-600">
        {{ fmt(shownTime) }} / {{ fmt(Math.ceil(duration)) }}
      </span>
    </template>
  </div>
</template>