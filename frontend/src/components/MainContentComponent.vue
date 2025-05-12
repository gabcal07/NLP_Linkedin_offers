<template>
  <v-card class="bg-blue-grey-darken-4">
        <v-card-title v-if="selectedModel == ''" class="font-italic font-weight-light">
          Please select a model
        </v-card-title>
        <v-card-title v-else>
          {{ selectedModel }}
          <v-card-text>
          <div>

            <h4>
            Input
          </h4>
            <v-textarea label="Input" class="mt-2 ml-n1" clearable></v-textarea>
        </div>
        <div class="mt-4">
          <h4>
            Output
          </h4>
        </div>
        </v-card-text>
        </v-card-title>
  </v-card>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  selectedRegression: {
    type: String,
    default: "",
  },
  selectedGeneration: {
    type: String,
    default: "",
  },
});

const selectedModel = computed(() => {
  if (props.selectedRegression != "") {
    return props.selectedRegression;
  } else if (props.selectedGeneration != "") {
    return props.selectedGeneration;
  }
  return "";
});

</script>
