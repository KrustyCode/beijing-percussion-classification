<script setup lang="ts">
import { ref, useTemplateRef } from 'vue'
import { LIMITS, fmt } from '@/constants/errors'

const file = defineModel<File | null>('file', { default: null })

const input = useTemplateRef<HTMLInputElement>('input')
const isDragging = ref(false)

function openPicker(): void {
    input.value?.click()
}

function onInputChange(e: Event): void {
    const target = e.target as HTMLInputElement
    file.value = target.files?.[0] ?? null
    target.value = ''
}

function onDrop(e: DragEvent): void {
    isDragging.value = false
    file.value = e.dataTransfer?.files?.[0] ?? null
}

defineExpose({ openPicker })
</script>

<template>
    <div
        role="button"
        tabindex="0"
        class="flex flex-col justify-center items-center border-2 w-full min-h-56 border-dashed rounded-lg space-y-1 p-4 cursor-pointer transition-colors"
        :class="isDragging ? 'border-emerald-700 bg-emerald-50' : 'border-gray-400 bg-mist-100'"
        @click="openPicker"
        @keydown.enter.prevent="openPicker"
        @keydown.space.prevent="openPicker"
        @dragenter.prevent="isDragging = true"
        @dragover.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="onDrop"
    >
        <div>
            <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.6"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="size-10 text-emerald-700"
            aria-hidden="true"
            >
            <path d="M12 16V4" />
            <path d="M7 9l5-5 5 5" />
            <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
        </div>
        <div class="flex flex-wrap justify-center items-center space-x-1">
            <span class="text-center">Drag a file here or </span>
            <button type="button" class="underline text-emerald-700 cursor-pointer" @click.stop="openPicker">choose a file</button>
        </div>
        <span class="text-center text-xs text-gray-500">{{ LIMITS.formats.join(' or ') }} · max {{ fmt(LIMITS.maxSizeMb) }} MB · {{ fmt(LIMITS.minDurationS) }}–{{ fmt(LIMITS.maxDurationS) }} seconds</span>
        <span class="text-center text-xs text-gray-500">One recording of a single instrument stroke</span>

        <input
            ref="input"
            type="file"
            accept=".wav,.mp3,audio/wav,audio/mpeg"
            class="hidden"
            @change="onInputChange"
        />
    </div>
</template>
