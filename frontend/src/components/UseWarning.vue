<script setup lang="ts">
    interface Props {
        title?: string
        description?: string
    }
    const {
        title = "Caution",
        description = "The model always picks one of the four instruments (bangu, daluo, naobo or xiaoluo), even if the audio contains other sounds. The result can be wrong, so double-check when in doubt."
    } = defineProps<Props>()
</script>
<template>
    <div class="w-full bg-amber-50 border-1 border-amber-300 rounded-lg p-4 flex flex-col gap-2">
        <div class="flex items-center gap-2">
            <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
            class="size-4 shrink-0 text-yellow-800"
            >
                <path d="M12 3l10 18H2z" />
                <path d="M12 10v5M12 18v.5" />
            </svg>
            <p class="text-yellow-800 text-sm font-medium">{{ title }}</p>
        </div>
        <p class="text-xs text-yellow-700">{{ description }}</p>
    </div>
</template>