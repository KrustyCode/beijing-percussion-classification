<script setup lang="ts">
import { ref, watch } from 'vue';
import UploadPanel from './UploadPanel.vue';
import ResultPanel from './ResultPanel.vue';
import { predict, ApiError, type ResultState } from '@/constants/api';
import { ERRORS } from '@/constants/errors';

const file = ref<File | null>(null)
const result = ref<ResultState>({ status: 'empty' })

watch(file, () => {
    result.value = { status: 'empty' }
})

async function upload() {
    const sent = file.value
    if (!sent) return

    result.value = { status: 'loading' }
    try {
        const prediction = await predict(sent)
        if (file.value !== sent) return
        result.value = { status: 'success', prediction }
    } catch (e) {
        if (file.value !== sent) return
        result.value = { status: 'error', error: e instanceof ApiError ? e.info : ERRORS.UNKNOWN_ERROR }
    }
}

</script>

<template>
    <div class="flex-1 flex justify-center items-center p-4 md:mx-8 md:p-8">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-6xl">
            <div class="flex justify-center">
                <UploadPanel v-model:file="file" @classify="upload"></UploadPanel>
            </div>
            <div class="flex justify-center">
                <ResultPanel :result="result"></ResultPanel>
            </div>
        </div>
    </div>
</template>
