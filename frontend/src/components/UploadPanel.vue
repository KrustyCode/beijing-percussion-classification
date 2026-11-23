<script setup lang="ts">
    import FileDropZone from './FileDropZone.vue';
    import BasedButton from './BasedButton.vue';
    import PrivacyNote from './PrivacyNote.vue';
    import SelectedFile from './SelectedFile.vue';
    import AudioPreview from './AudioPreview.vue';

    import { useTemplateRef } from 'vue';

    const file = defineModel<File | null> ('file', {default:null})
    const dropZone = useTemplateRef('dropZone')

    const emit = defineEmits<{
        classify: []
    }>()

</script>

<template>
    <div class="flex flex-col max-w-2xl w-full p-6 h-fit space-y-6 bg-gray-100 rounded-xl border-2 border-gray-400">
        <h2 class="w-full font-bold">Upload audio</h2>
        <FileDropZone v-show="file === null" ref="dropZone" v-model:file="file"></FileDropZone>
        <SelectedFile
            v-if="file !== null"
            :file="file"
            @remove="file = null"
            @change="dropZone?.openPicker()"
        ></SelectedFile>
        <AudioPreview v-if="file !== null" :file="file"></AudioPreview>
        <BasedButton :disabled="file === null" buttonLabel="Classify" @click="file !== null && emit('classify')"></BasedButton>
        <PrivacyNote noteLabel='Files are only processed during the request and deleted right after.'></PrivacyNote>
    </div>
</template>