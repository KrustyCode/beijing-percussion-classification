<script setup lang="ts">
    import EmptyState from './EmptyState.vue';
    import LoadingState from './LoadingState.vue';
    import PredictionResult from './PredictionResult.vue';
    import ErrorState from './ErrorState.vue';
    import type { ResultState } from '@/constants/api';
    import { CLASS_BY_ID, isClassId } from '@/constants/labels';

    defineProps<{ result: ResultState }>()

    function className(label: string): string {
        return isClassId(label) ? CLASS_BY_ID[label].name : label
    }
</script>
<template>
    <div class="w-full max-w-2xl flex flex-col items-center justify-center p-6 min-h-fit gap-6 bg-gray-100 rounded-xl border-2 border-gray-400">
        <h2 class="w-full font-bold">Classification result</h2>
        <EmptyState v-if="result.status === 'empty'"></EmptyState>
        <LoadingState v-else-if="result.status === 'loading'"></LoadingState>

        <PredictionResult v-else-if="result.status === 'success'" :result ="result"></PredictionResult>

        <ErrorState v-else :result="result"></ErrorState>
    </div>
</template>
