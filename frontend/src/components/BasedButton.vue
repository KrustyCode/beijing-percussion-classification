<script setup lang="ts">
    import {ref} from 'vue'
    const status = ref(false)
    defineProps({
        buttonLabel: String,
    })


</script>

<template>
    <button :disabled="status" class="disabled:bg-gray-300 disabled:text-gray-600
        font-bold
     enabled:bg-emerald-800 enabled:hover:bg-emerald-700 enabled:hover:cursor-pointer enabled:text-gray-100 rounded-lg w-full h-12">{{ buttonLabel }}</button>
</template>