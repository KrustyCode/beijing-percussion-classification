<script setup lang="ts">
import { ref, computed, onBeforeUnmount } from 'vue'

const props = defineProps({
  minDuration: { type: Number, default: 1 }, // seconds
  maxDuration: { type: Number, default: 30 }, // seconds
  maxSizeMB: { type: Number, default: 10 },
})

const emit = defineEmits<{ classify: [file: File] }>()

const inputRef = ref<HTMLInputElement | null>(null)
const file = ref<File | null>(null)
const error = ref('')
const isDragging = ref(false)
const isChecking = ref(false)

const canSubmit = computed(() => !!file.value && !error.value && !isChecking.value)

function openPicker() {
  inputRef.value?.click()
}

function onInputChange(e: Event) {
  const input = e.target as HTMLInputElement
  handleFile(input.files?.[0])
  input.value = '' // allow re-selecting the same file
}

function onDrop(e: DragEvent) {
  isDragging.value = false
  handleFile(e.dataTransfer?.files?.[0])
}

function getDuration(f: File) {
  return new Promise<number>((resolve, reject) => {
    const url = URL.createObjectURL(f)
    const audio = new Audio()
    audio.preload = 'metadata'
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(url)
      resolve(audio.duration)
    }
    audio.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('unreadable'))
    }
    audio.src = url
  })
}

async function handleFile(f: File | undefined) {
  if (!f) return
  error.value = ''
  file.value = null

  const isWav =
    /\.wav$/i.test(f.name) ||
    ['audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'].includes(f.type)
  if (!isWav) {
    error.value = 'Unsupported format. WAV files only.'
    return
  }
  if (f.size > props.maxSizeMB * 1024 * 1024) {
    error.value = `The file is larger than ${props.maxSizeMB} MB.`
    return
  }

  isChecking.value = true
  try {
    const duration = await getDuration(f)
    if (duration < props.minDuration || duration > props.maxDuration) {
      error.value = `The duration must be between ${props.minDuration}–${props.maxDuration} seconds.`
      return
    }
    file.value = f
  } catch {
    error.value = 'The audio file could not be read.'
  } finally {
    isChecking.value = false
  }
}

function removeFile() {
  file.value = null
  error.value = ''
}

function formatSize(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function submit() {
  if (canSubmit.value && file.value) emit('classify', file.value)
}

onBeforeUnmount(() => {
  file.value = null
})
</script>

<template>
  <div class="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
    <h2 class="mb-4 text-lg font-bold text-gray-900">Upload an audio file</h2>

    <!-- Drop zone -->
    <div
      role="button"
      tabindex="0"
      aria-label="Upload a WAV audio file"
      class="flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-8 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
      :class="
        isDragging
          ? 'border-teal-600 bg-teal-50'
          : error
            ? 'border-red-400 bg-red-50/40'
            : 'border-gray-300 bg-gray-50 hover:border-gray-400'
      "
      @click="openPicker"
      @keydown.enter.prevent="openPicker"
      @keydown.space.prevent="openPicker"
      @dragenter.prevent="isDragging = true"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
    >
      <!-- Upload icon -->
      <svg
        class="h-10 w-10 text-teal-800"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M12 16V4" />
        <path d="m7 9 5-5 5 5" />
        <path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
      </svg>

      <!-- Selected file -->
      <template v-if="file">
        <div class="flex max-w-full items-center gap-2 rounded-lg bg-white px-3 py-2 ring-1 ring-gray-200">
          <span class="truncate text-sm font-medium text-gray-900">{{ file.name }}</span>
          <span class="shrink-0 text-xs text-gray-500">{{ formatSize(file.size) }}</span>
          <button
            type="button"
            class="shrink-0 rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-900"
            aria-label="Remove file"
            @click.stop="removeFile"
          >
            ✕
          </button>
        </div>
      </template>

      <!-- Empty state -->
      <template v-else>
        <p class="font-semibold text-gray-900">
          {{ isChecking ? 'Checking file…' : 'Drag a file here or' }}
          <button
            v-if="!isChecking"
            type="button"
            class="font-semibold text-teal-800 underline underline-offset-2 hover:text-teal-900"
            @click.stop="openPicker"
          >
            choose a file
          </button>
        </p>
        <p class="text-sm text-gray-500">
          WAV only · max {{ maxSizeMB }} MB · {{ minDuration }}–{{ maxDuration }} seconds
        </p>
      </template>

      <p v-if="error" class="text-sm font-medium text-red-600" role="alert">{{ error }}</p>

      <input
        ref="inputRef"
        type="file"
        accept=".wav,audio/wav,audio/x-wav"
        class="hidden"
        @change="onInputChange"
      />
    </div>

    <!-- Submit -->
    <button
      type="button"
      class="mt-6 w-full rounded-xl px-4 py-3.5 text-base font-bold transition-colors"
      :class="
        canSubmit
          ? 'bg-teal-800 text-white hover:bg-teal-900'
          : 'cursor-not-allowed bg-gray-200 text-gray-500'
      "
      :disabled="!canSubmit"
      @click="submit"
    >
      Classify
    </button>

    <!-- Privacy note -->
    <div class="mt-6 flex items-start gap-3 text-sm text-gray-600">
      <svg
        class="mt-0.5 h-5 w-5 shrink-0 text-gray-700"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect x="5" y="11" width="14" height="9" rx="2" />
        <path d="M8 11V8a4 4 0 0 1 8 0v3" />
      </svg>
      <p>Files are only processed during the request and deleted right after.</p>
    </div>
  </div>
</template>