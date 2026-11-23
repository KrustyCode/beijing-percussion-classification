<script setup lang="ts">
    import Legends from './Legends.vue';
    import { CLASSES } from '@/constants/labels'
</script>
<template>
    <div class="flex flex-col w-full gap-3">
        <div class="w-full flex flex-col justify-center items-center text-center border-2 border-dashed border-gray-200 bg-gray-100 py-20 rounded-lg">
            <div>
                <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.6"
                stroke-linecap="round"
                class="size-7 text-gray-500 flex-none"
                aria-hidden="true"
                >
                <path d="M4 12h1M8 8v8M12 5v14M16 9v6M20 11v2" />
                </svg>
            </div>
            <p>No result yet</p>
            <p class="text-sm text-gray-500">Upload one audio file to start the classification.</p>
        </div>
        <span class="text-sm text-gray-500">Instruments the model can recognize</span>
        <Legends :classes="CLASSES"></Legends>
    </div>
</template>