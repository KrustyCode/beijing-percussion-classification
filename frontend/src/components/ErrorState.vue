<script setup lang="ts">
import { type ResultState } from '@/constants/api';
type ErrorResult = Extract<ResultState, { status: 'error' }>

const props = defineProps<{ result: ErrorResult }>()
</script>
<template>
    <div class="w-full h-fit bg-red-50 border-red-600 border-1 rounded-xl">
        <div class="w-full py-8 px-4 text-red-800 gap-3 flex flex-col">
            <div class="flex gap-2 items-center">
                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    aria-hidden="true"
                    class="size-6 shrink-0 text-red-800"
                    >
                    <circle cx="12" cy="12" r="9" />
                    <path d="M12 7v6M12 16.5v.5" />
                </svg>
                <p class="font-bold">{{ result.error.title }}</p>
            </div>
            <div >
                <p class="text-sm">{{ result.error.message }}</p>
                <p class="text-sm">{{ result.error.action }}</p>
            </div>
            <div>
                <p class="text-xs text-red-900">The file was not classified or stored.</p>
            </div>
        </div>
    </div>
</template>