<script setup lang="ts">
    import {computed} from 'vue'
    const props = defineProps<{ file: File }>()
    const extension = computed(() =>
        props.file.name.split('.').pop()?.toUpperCase() ?? ''
    )

    const sizeMb = computed(() =>
    (props.file.size / (1024 * 1024)).toLocaleString('en-US', {
        maximumFractionDigits: 1,
    })
    )
    const emit = defineEmits<{
        remove: []
        change: []
    }>()

    
</script>
<template>
  <div class="w-full flex items-center gap-3 px-4 py-2 bg-mist-100 border-2 rounded-lg border-gray-400">
    <div class="flex-none">
        <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        class="size-7 text-emerald-700 flex-none"
        aria-hidden="true"
        >
        <path d="M4 12h1M8 8v8M12 5v14M16 9v6M20 11v2" />
        </svg>
    </div>
    
    <div class="flex-1 min-w-0 flex flex-col gap-0.5">
      <span class="truncate">{{ file.name }}</span>
      <span class="text-xs">{{ sizeMb }} MB · {{extension}}</span>
    </div>

    <button class="flex-none underline p-2 text-emerald-700 cursor-pointer" @click="emit('change')">Change</button>
    <button class="flex-none p-2 cursor-pointer" aria-label="Remove file" @click="emit('remove')">
        <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        class="size-5"
        aria-hidden="true"
        >
        <path d="M6 6l12 12M18 6L6 18" />
        </svg>
    </button>
  </div>
</template>