<script setup lang="ts">
    interface Props{
        mainLoadingLabel?: String,
        secondLoadingLabel?: String
    }
    const{
        mainLoadingLabel = "Classifying the file with the model",
        secondLoadingLabel = "The file is being checked, processed and classified. This usually takes only a few seconds."
    } = defineProps<Props>()
</script>

<template>
    <div class="w-full flex flex-col justify-center items-center text-center border-2 border-dashed border-gray-200 bg-gray-100 py-20 rounded-lg px-2">
        <div>
            <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            class="size-11 text-emerald-700 animate-spin"
            aria-hidden="true"
            >
            <path d="M21 12a9 9 0 1 1-9-9" />
            </svg>
        </div>
        <p>{{ mainLoadingLabel }}</p>
        <p class="text-sm text-gray-500">{{ secondLoadingLabel }}</p>
    </div>
</template>