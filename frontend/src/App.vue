<script setup lang="ts">
import UploadPage from './components/UploadPage.vue'
import Header from './components/Header.vue'
</script>

<template>
  <div class="min-h-screen flex flex-col bg-gray-200">
    <Header />
    <main class="flex-1 flex flex-col">
      <UploadPage />
    </main>
  </div>
</template>
