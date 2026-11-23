<script setup lang="ts">
import {computed} from 'vue';
import { CLASS_BY_ID, type ClassInfo, isClassId} from '@/constants/labels';
import { type ResultState } from '@/constants/api';
import UseWarning from './UseWarning.vue';

type SuccessResult = Extract<ResultState, { status: 'success' }>

const props = defineProps<{ result: SuccessResult }>()

const classInfo = computed<ClassInfo | null>(() => {
  const id =  props.result?.prediction.label.toLowerCase()
  return isClassId(id) ? CLASS_BY_ID[id] : null       // unknown label → null
})



</script>
<template>
    <div class="w-full flex flex-col gap-3">
        <div class="w-full bg-emerald-50 px-4 py-6 flex flex-col gap-2 rounded-lg border-1 border-emerald-500">
            <h2 class="text-xs font-medium text-emerald-800">DETECTED INSTRUMENT</h2>
            <p class="text-4xl font-bold">{{ classInfo?.name}}</p>
            <div>
                <span class="text-emerald-800 text-xl">{{ classInfo?.nameZh }}</span><span> - </span><span>{{ classInfo?.description }}</span>
            </div>
            <div class="w-full h-[1px] bg-gray-300"></div>
            <p class="text-xs text-gray-500">
                {{ result.status=='success'?result.prediction.analysed_seconds: '' }} seconds analyzed · {{ result.status=="success" ? result.prediction.processing_ms: '' }} ms
            </p>
        </div>
        <UseWarning></UseWarning>
    </div>
</template>