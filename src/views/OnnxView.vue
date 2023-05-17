<!-- Front-end Part -->
<script setup lang="ts">
import * as ort from 'onnxruntime-web'
import { ref, computed, onMounted, nextTick } from 'vue'
import { transformImage, preprocess, predict } from '@/utils/runtime'

import BarChart from '@/components/BarChart.vue'
import CardItem from '@/components/CardItem.vue'
import FilePicker from '@/components/FilePicker.vue'
import ListSelect from '@/components/ListSelect.vue'

// wasm 路径
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/ort-wasm.wasm',
  'ort-wasm-threaded.wasm':
    'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/ort-wasm-threaded.wasm',
  'ort-wasm-simd.wasm':
    'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/ort-wasm-simd.wasm',
  'ort-wasm-simd-threaded.wasm':
    'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/ort-wasm-simd-threaded.wasm'
}

const executionProvider = ref<'wasm' | 'webgl'>('wasm')
const modelUrl = ref<string>('')
const model = ref<ort.InferenceSession | null>(null)
const classIndices = ref<{
  [key: number]: string
} | null>(null)
const inputImageUrl = ref<string>('')

const isLoadingModel = ref(false)
const isLoadingPredict = ref(false)
const canPredict = computed(() => {
  return model.value && classIndices.value
})

const seriesData = ref<number[]>([])
const seriesLabels = ref<string[]>([])
const timeCost = ref<number>(0)

const useCamera = ref(false)

// webcam
const availableDevices = ref<InputDeviceInfo[]>([])
const canvas = ref<HTMLCanvasElement | null>(null)
const ctx = ref<CanvasRenderingContext2D | null>(null)
const globalStream = ref<MediaStream | null>(null)
const constraints = ref({
  audio: false,
  video: { width: 256, height: 256, deviceId: '' }
})
const deviceOptions = computed(() => {
  return availableDevices.value.map((device) => {
    return {
      value: device.deviceId,
      label: device.label
    }
  })
})
const deviceSelected = ref<{
  value: string
  label: string
}>({
  value: '',
  label: ''
})
const isCameraOn = ref(false)

const processFrame = async (ctx: CanvasRenderingContext2D) => {
  if (useCamera.value) {
    const imageData = ctx.getImageData(0, 0, 256, 256)
    const inputTensor = transformImage(imageData, 256, 256)
    const res = await predict(model.value, inputTensor, classIndices.value)
    if (res) {
      timeCost.value = res.timeCost
      seriesData.value = res.data
      seriesLabels.value = res.labels
    }
  } else {
    seriesData.value = []
    seriesLabels.value = []
  }
}

const startCamera = async () => {
  useCamera.value = true
  nextTick(async () => {
    canvas.value = document.getElementById('canvas') as HTMLCanvasElement
    ctx.value = canvas.value.getContext('2d', {
      willReadFrequently: true
    }) as CanvasRenderingContext2D
    if (deviceSelected.value.value !== '') {
      constraints.value.video.deviceId = deviceSelected.value.value
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints.value)
        // @ts-ignore
        window.stream = stream // make variable available to browser console
        const videoTracks = stream.getVideoTracks()
        console.log('Got stream with constraints:', constraints)
        console.log('Using video device: ' + videoTracks[0].label)
        globalStream.value = stream // make variable available to browser console
        const videoElement = document.createElement('video')
        videoElement.srcObject = stream
        videoElement.play()
        isCameraOn.value = true
        const renderFrame = async () => {
          if (canvas.value !== null && ctx.value !== null) {
            ctx.value.drawImage(videoElement, 0, 0, canvas.value.width, canvas.value.height)
            processFrame(ctx.value)
          }
          if (useCamera.value) {
            requestAnimationFrame(renderFrame)
          }
        }
        requestAnimationFrame(renderFrame)
      } catch (error: any) {
        if (error.name === 'ConstraintNotSatisfiedError') {
          console.error('ConstraintNotSatisfiedError error: ' + error.name, error)
        } else if (error.name === 'PermissionDeniedError') {
          console.error(
            'Permissions have not been granted to use your camera and ' +
              'microphone, you need to allow the page access to your devices in ' +
              'order for the demo to work.'
          )
        }
        console.error('getUserMedia error: ' + error.name, error)
      }
    }
  })
}

const stopCamera = (stream: MediaStream | null) => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop())
  }
  isCameraOn.value = false
  useCamera.value = false
}

const openMediaDevices = async (constraints: MediaStreamConstraints) => {
  return await navigator.mediaDevices.getUserMedia(constraints)
}

onMounted(async () => {
  // Load default model (too slow)
  // const defaultModelUrl = '/resnet18_imagenet.onnx'; // 默认模型的 URL，根据实际情况修改
  // isLoadingModel.value = true;
  // try {
  //   model.value = await ort.InferenceSession.create(defaultModelUrl, {
  //     executionProviders: [executionProvider.value]
  //   });
  // } catch (error) {
  //   console.error('Failed to load default model 模型加载失败:', error);
  // }
  // isLoadingModel.value = false;
  // get camera
  console.log('Get Cameras')
  try {
    const stream = await openMediaDevices({
      audio: false,
      video: true
    })
    // 关闭摄像头
    stopCamera(stream)
    const devices = await navigator.mediaDevices.enumerateDevices()
    devices.forEach(function (device) {
      if (device.kind === 'videoinput' && device.label !== '' && device.deviceId !== '') {
        availableDevices.value.push(device)
      }
    })
    if (availableDevices.value.length > 0) {
      deviceSelected.value = {
        value: availableDevices.value[0].deviceId,
        label: availableDevices.value[0].label
      }
    }
  } catch (error) {
    console.error('Error opening video camera.', error)
  }
  
})

// execution provider change handler
const executionProviderChangeHandler = async (provider: 'wasm' | 'webgl') => {
  executionProvider.value = provider
  if (modelUrl.value) {
    isLoadingModel.value = true
    model.value = await ort.InferenceSession.create(modelUrl.value, {
      executionProviders: [executionProvider.value]
    })
    isLoadingModel.value = false
  }
}

// model file change handler
// select model file
// model file change handler
// select model file
const modelFileChangeHandler = async (event: Event) => {
  isLoadingModel.value = true
  const model_file = (event.target as HTMLInputElement).files?.[0]
  const model_file_array_buffer = await model_file?.arrayBuffer()
  if (model_file_array_buffer) {
    const model_file_blob = new Blob([model_file_array_buffer])
    const model_url = URL.createObjectURL(model_file_blob)
    modelUrl.value = model_url
    model.value = await ort.InferenceSession.create(model_url, {
      executionProviders: [executionProvider.value]
    })
  }
  isLoadingModel.value = false
  
  // Re-run prediction if we have an image
  if (inputImageUrl.value !== '') {
    isLoadingPredict.value = true
    if (model.value && classIndices.value) {
      const inputTensor = await preprocess(inputImageUrl.value)
      const res = await predict(model.value, inputTensor, classIndices.value)
      if (res) {
        timeCost.value = res.timeCost
        seriesData.value = res.data
        seriesLabels.value = res.labels
      }
    }
    isLoadingPredict.value = false
  }
}


// const modelFileChangeHandler = async (event: Event) => {
//   isLoadingModel.value = true
//   const model_file = (event.target as HTMLInputElement).files?.[0]
//   const model_file_array_buffer = await model_file?.arrayBuffer()
//   if (model_file_array_buffer) {
//     const model_file_blob = new Blob([model_file_array_buffer])
//     const model_url = URL.createObjectURL(model_file_blob)
//     modelUrl.value = model_url
//     model.value = await ort.InferenceSession.create(model_url, {
//       executionProviders: [executionProvider.value]
//     })
//   }
//   isLoadingModel.value = false
// }

// classes file change handler
const classesFileChangeHandler = async (event: Event) => {
  const classes_file = (event.target as HTMLInputElement).files?.[0]
  const classes_file_array_buffer = await classes_file?.arrayBuffer()
  if (classes_file_array_buffer) {
    const classes_file_blob = new Blob([classes_file_array_buffer])
    const classes_url = URL.createObjectURL(classes_file_blob)
    const classes_text = await fetch(classes_url).then((response) => response.text())
    classIndices.value = JSON.parse(classes_text)
  }
}

// image file change handler
const imageFileChangeHandler = async (event: Event) => {
  useCamera.value = false
  const image_file = (event.target as HTMLInputElement).files?.[0] ?? null
  if (image_file) {
    seriesData.value = []
    seriesLabels.value = []
    const inputImageBlob = new Blob([image_file])
    inputImageUrl.value = URL.createObjectURL(inputImageBlob)
    isLoadingPredict.value = true
    if (model.value && classIndices.value) {
      const inputTensor = await preprocess(inputImageUrl.value)
      const res = await predict(model.value, inputTensor, classIndices.value)
      if (res) {
        timeCost.value = res.timeCost
        seriesData.value = res.data
        seriesLabels.value = res.labels
      }
    }
    isLoadingPredict.value = false
  }
}

// predict
// select device, load model, predict
const imagePredictHandler = async () => {
  const imageUploader = document.getElementById('imageUploader')
  if (imageUploader) {
    imageUploader.click()
  }
}
</script>
<template>
  <main class="w-full flex flex-col gap-4 p-4">
    <section class="grid grid-cols-5 gap-4 <lg:grid-cols-3 <md:grid-cols-2 <sm:grid-cols-1">
      <CardItem title="0. Select Inference Device">
        <div>
          <input
            id="cpu"
            class="form-radio peer/cpu mb-0.5 mr-2 border-1 border-slate-300 border-solid text-sky-400 focus:ring-sky-300"
            type="radio"
            name="status"
            checked
            @change="executionProviderChangeHandler('wasm')"
          /><label class="font-medium peer-checked/cpu:text-sky-500" for="cpu"
            >CPU - WebAssembly</label
          >
          <br />
          <input
            id="gpu"
            class="form-radio peer/gpu mb-0.5 mr-2 border-1 border-slate-300 border-solid text-sky-400 focus:ring-sky-300"
            type="radio"
            name="status"
            @change="executionProviderChangeHandler('webgl')"
          /><label class="font-medium peer-checked/gpu:text-sky-500" for="gpu">GPU - WebGL</label>
          <br />
          <input
            id="webgpu"
            class="form-radio peer/webgpu mb-0.5 mr-2 border-1 border-slate-300 border-solid text-sky-400 focus:ring-sky-300"
            type="radio"
            name="status"
            @change="executionProviderChangeHandler('webgl')"
          /><label class="font-medium peer-checked/webgpu:text-sky-500" for="webgpu"
            >GPU - WebGPU</label
          >
        </div>
      </CardItem>
      <CardItem title="1. Upload Onnx File" :loading="isLoadingModel" loading-text="Loading...">
        <FilePicker accept=".onnx" name="model" @change="modelFileChangeHandler" />
      </CardItem>
      <CardItem title="2. Upload class File (json)">
        <FilePicker accept=".json" name="classes" @change="classesFileChangeHandler" />
      </CardItem>
      <CardItem title="Predict By Image" :loading="isLoadingPredict" loading-text="Inferencing...">
        <input
          type="file"
          accept="image/*"
          name="image"
          id="imageUploader"
          class="hidden"
          @change="imageFileChangeHandler"
        />
        <button
          id="imagePredict"
          @click="imagePredictHandler"
          :title="!canPredict ? '请先上传模型和类别文件' : ''"
          :disabled="!canPredict"
          class="disabled:cursor-disallow w-full border-0 rounded-full px-4 py-2 text-sm font-semibold enabled:cursor-pointer disabled:cursor-not-allowed disabled:bg-slate-200 enabled:bg-teal-50 disabled:text-slate-500 enabled:text-teal enabled:hover:bg-teal-100"
        >
          Upload Image
        </button>
      </CardItem>
      <!-- <CardItem title="推理预测/摄像头实时"> -->
      <CardItem title="Predict by Camera">
        <ListSelect :options="deviceOptions" v-model="deviceSelected" />
        <button
          v-if="isCameraOn === false"
          class="disabled:cursor-disallow mt-2 w-full border-0 rounded-full px-4 py-2 text-sm font-semibold enabled:cursor-pointer disabled:cursor-not-allowed disabled:bg-slate-200 enabled:bg-teal-50 disabled:text-slate-500 enabled:text-teal enabled:hover:bg-teal-100"
          :disabled="deviceSelected.value === '' || !canPredict"
          :title="!canPredict ? '请先上传模型和类别文件' : ''"
          @click="startCamera"
        >
          Open Camera
        </button>
        <button
          v-else
          class="disabled:cursor-disallow mt-2 w-full border-0 rounded-full px-4 py-2 text-sm font-semibold enabled:cursor-pointer disabled:cursor-not-allowed disabled:bg-slate-200 enabled:bg-red-50 disabled:text-slate-500 enabled:text-red enabled:hover:bg-red-100"
          @click="stopCamera(globalStream)"
        >
          Close Camera
        </button>
      </CardItem>
    </section>
    <section v-if="seriesData.length > 0 && !useCamera">
      <h2 class="mb-2 text-base font-semibold prose prose-slate">
        Predictions(Time：{{ timeCost }}ms)
      </h2>
      <div class="not-prose flex gap-4 <md:flex-wrap">
        <div class="flex flex-col overflow-hidden rounded bg-white shadow">
          <img :src="inputImageUrl" alt="" class="<md:w-full md:w-256px" />
        </div>
        <div class="flex-1 rounded bg-white p-4 shadow <md:flex-auto">
          <BarChart :data="seriesData" :labels="seriesLabels" />
        </div>
      </div>
    </section>
    <section v-if="useCamera">
      <h2 class="mb-2 text-base font-semibold prose prose-slate">
        Real-time Prediction(Time：{{ timeCost }}ms)
      </h2>
      <div class="not-prose flex gap-4 <md:flex-wrap">
        <div class="max-w-1/4 flex flex-col overflow-hidden rounded bg-white shadow <md:max-w-full">
          <canvas width="256" height="256" id="canvas"></canvas>
        </div>
        <div class="flex-1 rounded bg-white p-4 shadow <md:flex-auto">
          <BarChart :data="seriesData" :labels="seriesLabels" />
        </div>
      </div>
    </section>
    <!-- Example Resources Download Part -->
    <section class="w-full">
      <h2 class="mb-2 text-base font-semibold text-slate-700">Examples</h2>
      <div class="grid grid-cols-3 gap-4 <md:grid-cols-2 <sm:grid-cols-1">
        <article class="w-full rounded bg-white p-4 text-sm text-slate-700 shadow">
          <p class="leading-6">
            ONNX ResNet-18：<a
              href="https://drive.google.com/uc?export=download&id=1RSbr6a_cYthQvYZLLeddUJmNraohvtHJ"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >resnet18_imagenet.onnx</a
            >
          </p>
          <p class="leading-6">
            ONNX ResNet-50：<a
              href="https://drive.google.com/uc?export=download&id=1aX99hNTSff32Fq1r-O3HLdbIwc__Zueh"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >resnet50_imagenet.onnx</a
            >
          </p>
          <p class="leading-6">
            Imagenet classes index：<a
              href="https://drive.google.com/uc?export=download&id=1FAVLFX9xykJ061BESY1qbdYVk2X2Rhku"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >imagenet_class_index.json</a
            >
          </p>
        </article>
        <article class="w-full rounded bg-white p-4 text-sm text-slate-700 shadow">
          <p class="leading-6">
            Koala：<a
              href="https://drive.google.com/uc?export=download&id=1wnbKkwgbljXXLkEKbhiFEA1Ej8Z8oQGQ"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >koala_1.jpeg</a
            >
          </p>
          <p class="leading-6">
            Giant Panda：<a
              href="https://drive.google.com/uc?export=download&id=1-BiZ1Cin8Pvwc-Suc72KW3VYpBRbB7NL"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >giant_panda_1.jpg</a
            >
          </p>
          <p class="leading-6">
            Red Fox：<a
              href="https://drive.google.com/file/d/1da2HHqLw8mJzddinKecdJq46UbiKuPlO/view?usp=sharing"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >red_fox_1.jpg</a
            >
          </p>
          <p class="leading-6">
            Mountain Bike：<a
              href="https://drive.google.com/uc?export=download&id=1D62UgfAOIitoVjFOQMIKJDwgg9V-otfn"
              target="_blank"
              class="text-blue-500 no-underline visited:text-blue-700"
              >bike_1.jpg</a
            >
          </p>
        </article>
      </div>
    <!-- more information -->
    </section>
    <section class="w-full">
      <h2 class="mb-2 text-base font-semibold text-slate-700">Usage</h2>
      <article class="w-full rounded bg-white p-4 text-sm text-slate-700 shadow">
        <p class="leading-6">
          A Image Classification System based on ONNX Runtime.

        </p>
        <p class="leading-6">
          <br />
        </p>
        <p class="leading-6">
          1. Upload a trained model in onnx format and the corresponding class index file in json format.
           Some examples are given in the link above.
          <br>
          2. Upload the image you want to predict. Or use camera to predict in real time.
          <br>
          You can download the ONNX model from the link above, and also you can download
           the example images and use them to test the model.
          <br>
          If you change the model, you should also change the class index file.
           The format can be refer to the example json file.

        </p>
        <p>
          <br>
        </p>
        <p class="leading-6">
          Convert pytorch model to onnx <a
            href="https://colab.research.google.com/drive/1tuko14EQQorQDBecR5_KFbLeTmFFRNw4?usp=sharing"
            target="_blank"
            class="text-blue-500 no-underline visited:text-blue-700"
            >Example</a
          >
        </p>
        <p class="leading-6">
          Link of <a
            href="https://github.com/camtrik/image-classification-web-app"
            target="_blank"
            class="text-blue-500 no-underline visited:text-blue-700"
            >Github</a
          >
        </p>
      </article>
    </section>
  </main>
</template>
<style lang="scss" scoped></style>
