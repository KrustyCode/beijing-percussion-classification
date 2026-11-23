<script setup lang="ts">
import type { ClassInfo } from '@/constants/labels'

defineProps<{ classes: readonly ClassInfo[] }>()
</script>

<template>
  <div class="grid grid-cols-2 gap-3 w-full">
    <div
      v-for="item in classes"
      :key="item.id"
      class="border-2 border-gray-200 rounded-lg p-3 flex flex-col gap-1"
    >
      <div class="flex flex-wrap items-baseline gap-2">
        <span class="text-xl">{{ item.name }}</span>
        <span class="text-xl text-emerald-800">{{ item.nameZh }}</span>
      </div>
      <span class="text-xs text-gray-500">{{ item.description }}</span>
    </div>
  </div>
</template>