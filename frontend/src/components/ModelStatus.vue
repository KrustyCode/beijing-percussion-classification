<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { health } from '@/constants/api'

type Status = 'checking' | 'online' | 'offline'

const POLL_MS = 30_000 // re-check so the light follows the server going up or down

const status = ref<Status>('checking')
const modelName = ref('')
let timer: ReturnType<typeof setInterval> | undefined

async function check() {
    try {
        const h = await health()
        modelName.value = h.models
        status.value = 'online'
    } catch {
        status.value = 'offline'
    }
}

const VIEW: Record<Status, { label: string; dot: string }> = {
    checking: { label: 'Checking model…', dot: 'bg-gray-300 animate-pulse' },
    online: { label: 'Model available', dot: 'bg-emerald-400' },
    offline: { label: 'Model unavailable', dot: 'bg-red-400' },
}
const view = computed(() => VIEW[status.value])
const tooltip = computed(() =>
    status.value === 'online' ? `Active model: ${modelName.value}` : view.value.label,
)

onMounted(() => {
    check()
    timer = setInterval(check, POLL_MS)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
    <div
        class="flex items-center gap-2 rounded-full bg-white/10 ring-1 ring-white/15 px-3 py-1 text-xs font-medium"
        role="status"
        aria-live="polite"
        :title="tooltip"
    >
        <span class="relative flex size-2.5 shrink-0">
            <span
                v-if="status === 'online'"
                class="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-60"
            ></span>
            <span class="relative inline-flex size-2.5 rounded-full" :class="view.dot"></span>
        </span>
        <span class="whitespace-nowrap">{{ view.label }}</span>
    </div>
</template>
