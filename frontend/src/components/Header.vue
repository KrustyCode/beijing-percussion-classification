<script setup lang="ts">
import ModelStatus from './ModelStatus.vue'
</script>

<template>
    <header class="sticky top-0 z-10 w-full bg-emerald-900 text-gray-100 shadow-md px-4 md:px-16">
        <div class="mx-auto max-w-6xl flex flex-wrap items-center justify-between gap-x-4 gap-y-3 py-3">
            <div class="flex items-center gap-3 min-w-0">
                <div class="grid place-items-center size-10 shrink-0 rounded-lg bg-white/10 ring-1 ring-white/15">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="1.8"
                        stroke-linecap="round"
                        class="size-6 text-emerald-300"
                        aria-hidden="true"
                    >
                        <path d="M4 12h1M8 8v8M12 5v14M16 9v6M20 11v2" />
                    </svg>
                </div>
                <div class="min-w-0">
                    <h1 class="text-lg font-bold leading-tight tracking-tight">Percussion Classifier</h1>
                    <p class="text-xs text-emerald-200/80 leading-tight truncate">
                        Beijing opera percussion · bangu, daluo, naobo, xiaoluo
                    </p>
                </div>
            </div>
            <ModelStatus />
        </div>
    </header>
</template>
