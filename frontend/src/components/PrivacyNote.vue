<script setup lang="ts">
    interface Props{
        noteLabel?: String
    }
    const{
        noteLabel = 'This is Privacy Note!'
    } = defineProps<Props>()

</script>

<template>
    <div class="flex items-center gap-3 text-sm text-gray-600">
      <svg
        class="mt-0.5 h-5 w-5 shrink-0 text-gray-500"
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
      <p class="text-xs">{{ noteLabel }}</p>
    </div>
</template>