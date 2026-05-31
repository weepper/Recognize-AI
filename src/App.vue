<template>
	<div id="recognize_ai-admin-settings-inner">
		<!-- ============================================================ -->
		<!-- Section 1: Backend Status -->
		<!-- ============================================================ -->
		<NcSettingsSection title="Backend Status"
			description="Health check and AI model loading status">
			<div class="status-section">
				<!-- Health Check -->
				<div class="health-check">
					<div class="health-check__row">
						<NcButton type="secondary"
							:disabled="healthLoading"
							@click="checkHealth">
							<template #icon>
								<span v-if="healthLoading" class="icon-loading-small" />
							</template>
							{{ healthLoading ? 'Checking…' : 'Run health check' }}
						</NcButton>

						<span v-if="healthStatus === 'ok'" class="health-check__result health-check__result--ok">
							✓ Backend is healthy
							<span v-if="healthResponseTime" class="health-check__time">
								({{ healthResponseTime }}ms)
							</span>
						</span>
						<span v-else-if="healthStatus === 'error'" class="health-check__result health-check__result--error">
							✗ Backend unreachable
						</span>
					</div>

					<NcNoteCard v-if="healthError" type="error">
						{{ healthError }}
					</NcNoteCard>
				</div>

				<!-- Model Status Cards -->
				<div class="model-cards">
					<h4 v-if="models.length" class="model-cards__title">AI Models</h4>
					<div v-if="modelsLoading && !models.length" class="model-cards__loading">
						<span class="icon-loading-small" /> Loading model status…
					</div>
					<ModelStatusCard v-for="model in models"
						:key="model.name"
						:name="model.name"
						:loaded="model.loaded"
						:error="model.error"
						:quantized="model.quantized" />
					<NcNoteCard v-if="modelsError" type="warning">
						Could not fetch model status: {{ modelsError }}
					</NcNoteCard>
				</div>
			</div>
		</NcSettingsSection>

		<!-- ============================================================ -->
		<!-- Section 2: GPU & Runtime -->
		<!-- ============================================================ -->
		<NcSettingsSection title="GPU &amp; Runtime"
			description="Execution provider and hardware acceleration details">
			<div v-if="gpuLoading" class="gpu-section__loading">
				<span class="icon-loading-small" /> Loading GPU info…
			</div>

			<template v-else-if="gpuInfo">
				<div class="info-grid">
					<div class="info-grid__item">
						<span class="info-grid__label">Execution Provider</span>
						<span class="info-grid__value">
							<span class="provider-badge" :class="providerBadgeClass">
								{{ gpuInfo.provider || 'Unknown' }}
							</span>
						</span>
					</div>
					<div v-if="gpuInfo.device_name" class="info-grid__item">
						<span class="info-grid__label">Device</span>
						<span class="info-grid__value">{{ gpuInfo.device_name }}</span>
					</div>
					<div v-if="gpuInfo.device_id !== undefined" class="info-grid__item">
						<span class="info-grid__label">Device ID</span>
						<span class="info-grid__value">{{ gpuInfo.device_id }}</span>
					</div>
					<div v-if="gpuInfo.cuda_version" class="info-grid__item">
						<span class="info-grid__label">CUDA Version</span>
						<span class="info-grid__value">{{ gpuInfo.cuda_version }}</span>
					</div>
					<div v-if="gpuInfo.memory_total" class="info-grid__item">
						<span class="info-grid__label">GPU Memory</span>
						<span class="info-grid__value">{{ gpuInfo.memory_total }}</span>
					</div>
					<div v-if="gpuInfo.onnxruntime_version" class="info-grid__item">
						<span class="info-grid__label">ONNX Runtime</span>
						<span class="info-grid__value">{{ gpuInfo.onnxruntime_version }}</span>
					</div>
				</div>
			</template>

			<NcNoteCard v-if="gpuError" type="warning">
				Could not fetch GPU info: {{ gpuError }}
			</NcNoteCard>
		</NcSettingsSection>

		<!-- ============================================================ -->
		<!-- Section 3: File Scanner -->
		<!-- ============================================================ -->
		<NcSettingsSection title="File Scanner"
			description="Configure and monitor background file analysis">
			<ScannerPanel />
		</NcSettingsSection>

		<!-- ============================================================ -->
		<!-- Section 4: Configuration -->
		<!-- ============================================================ -->
		<NcSettingsSection title="Configuration"
			description="Current backend configuration (read-only)">
			<div v-if="configLoading" class="config-section__loading">
				<span class="icon-loading-small" /> Loading configuration…
			</div>

			<template v-else-if="configData">
				<div class="info-grid">
					<div v-for="(value, key) in configData"
						:key="key"
						class="info-grid__item">
						<span class="info-grid__label">{{ formatConfigKey(key) }}</span>
						<span class="info-grid__value">
							<template v-if="typeof value === 'boolean'">
								<span :class="value ? 'config-bool--true' : 'config-bool--false'">
									{{ value ? 'Yes' : 'No' }}
								</span>
							</template>
							<template v-else-if="Array.isArray(value)">
								{{ value.join(', ') }}
							</template>
							<template v-else>
								{{ value }}
							</template>
						</span>
					</div>
				</div>
			</template>

			<NcNoteCard v-if="configError" type="warning">
				Could not fetch configuration: {{ configError }}
			</NcNoteCard>
		</NcSettingsSection>
	</div>
</template>

<script>
import axios from '@nextcloud/axios'
import { generateUrl } from '@nextcloud/router'
import NcButton from '@nextcloud/vue/dist/Components/NcButton.js'
import NcNoteCard from '@nextcloud/vue/dist/Components/NcNoteCard.js'
import NcSettingsSection from '@nextcloud/vue/dist/Components/NcSettingsSection.js'

import ModelStatusCard from './components/ModelStatusCard.vue'
import ScannerPanel from './components/ScannerPanel.vue'

export default {
	name: 'App',
	components: {
		NcButton,
		NcNoteCard,
		NcSettingsSection,
		ModelStatusCard,
		ScannerPanel,
	},
	data() {
		return {
			// Health check
			healthLoading: false,
			healthStatus: null,
			healthResponseTime: null,
			healthError: '',

			// Models
			models: [],
			modelsLoading: false,
			modelsError: '',

			// GPU
			gpuInfo: null,
			gpuLoading: false,
			gpuError: '',

			// Config
			configData: null,
			configLoading: false,
			configError: '',
		}
	},
	computed: {
		providerBadgeClass() {
			if (!this.gpuInfo || !this.gpuInfo.provider) {
				return ''
			}
			const p = this.gpuInfo.provider.toLowerCase()
			if (p.includes('cuda') || p.includes('tensorrt')) {
				return 'provider-badge--gpu'
			}
			return 'provider-badge--cpu'
		},
	},
	mounted() {
		this.fetchAll()
	},
	methods: {
		async fetchAll() {
			// Fire all requests in parallel
			await Promise.allSettled([
				this.checkHealth(),
				this.fetchModels(),
				this.fetchGpuInfo(),
				this.fetchConfig(),
			])
		},

		async checkHealth() {
			this.healthLoading = true
			this.healthError = ''
			this.healthStatus = null
			this.healthResponseTime = null

			const startTime = Date.now()
			try {
				const url = generateUrl('/apps/recognize_ai/health')
				const { data } = await axios.get(url)
				this.healthResponseTime = Date.now() - startTime
				this.healthStatus = data.status === 'ok' ? 'ok' : 'error'
			} catch (err) {
				this.healthStatus = 'error'
				this.healthError = err.response?.data?.detail
					|| err.message
					|| 'Could not reach the backend'
			} finally {
				this.healthLoading = false
			}
		},

		async fetchModels() {
			this.modelsLoading = true
			this.modelsError = ''

			try {
				const url = generateUrl('/apps/recognize_ai/models/status')
				const { data } = await axios.get(url)
				// Transform {name: {loaded, error}} to array
				this.models = Object.entries(data).map(([name, info]) => ({
					name,
					loaded: info.loaded || false,
					error: info.error || null,
					// Detect quantized from model name convention or backend info
					quantized: info.quantized || false,
				}))
			} catch (err) {
				this.modelsError = err.response?.data?.detail
					|| err.message
					|| 'Connection error'
			} finally {
				this.modelsLoading = false
			}
		},

		async fetchGpuInfo() {
			this.gpuLoading = true
			this.gpuError = ''

			try {
				const url = generateUrl('/apps/recognize_ai/api/v1/gpu-info')
				const { data } = await axios.get(url)
				this.gpuInfo = data
			} catch (err) {
				this.gpuError = err.response?.data?.detail
					|| err.message
					|| 'Connection error'
			} finally {
				this.gpuLoading = false
			}
		},

		async fetchConfig() {
			this.configLoading = true
			this.configError = ''

			try {
				const url = generateUrl('/apps/recognize_ai/api/v1/config')
				const { data } = await axios.get(url)
				this.configData = data
			} catch (err) {
				this.configError = err.response?.data?.detail
					|| err.message
					|| 'Connection error'
			} finally {
				this.configLoading = false
			}
		},

		formatConfigKey(key) {
			// Convert snake_case to Title Case
			return key
				.replace(/_/g, ' ')
				.replace(/\b\w/g, c => c.toUpperCase())
		},
	},
}
</script>

<style scoped>
#recognize_ai-admin-settings-inner {
	max-width: 900px;
}

/* Health Check */
.health-check {
	margin-bottom: 24px;
}

.health-check__row {
	display: flex;
	align-items: center;
	gap: 16px;
	flex-wrap: wrap;
}

.health-check__result {
	font-size: 14px;
	font-weight: 600;
}

.health-check__result--ok {
	color: var(--color-success, #46ba61);
}

.health-check__result--error {
	color: var(--color-error, #e9322d);
}

.health-check__time {
	font-weight: 400;
	color: var(--color-text-maxcontrast, #767676);
}

/* Model Cards */
.model-cards {
	margin-top: 8px;
}

.model-cards__title {
	font-size: 15px;
	font-weight: 600;
	margin: 0 0 12px 0;
	color: var(--color-main-text);
}

.model-cards__loading {
	display: flex;
	align-items: center;
	gap: 8px;
	color: var(--color-text-maxcontrast);
	font-size: 13px;
}

/* Info Grid (GPU & Config) */
.info-grid {
	display: grid;
	grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
	gap: 12px;
}

.info-grid__item {
	display: flex;
	flex-direction: column;
	padding: 12px 16px;
	background-color: var(--color-background-dark, #f5f5f5);
	border-radius: var(--border-radius-large, 10px);
}

.info-grid__label {
	font-size: 12px;
	font-weight: 600;
	color: var(--color-text-maxcontrast, #767676);
	text-transform: uppercase;
	letter-spacing: 0.5px;
	margin-bottom: 4px;
}

.info-grid__value {
	font-size: 14px;
	color: var(--color-main-text);
	word-break: break-all;
}

/* Provider badge */
.provider-badge {
	display: inline-block;
	padding: 2px 10px;
	border-radius: var(--border-radius-pill, 20px);
	font-size: 13px;
	font-weight: 700;
}

.provider-badge--gpu {
	background-color: #e8f5e9;
	color: #2e7d32;
}

.provider-badge--cpu {
	background-color: var(--color-background-dark, #ededed);
	color: var(--color-text-maxcontrast, #767676);
}

/* Config booleans */
.config-bool--true {
	color: var(--color-success, #46ba61);
	font-weight: 600;
}

.config-bool--false {
	color: var(--color-text-maxcontrast, #767676);
}

/* Section loading */
.gpu-section__loading,
.config-section__loading {
	display: flex;
	align-items: center;
	gap: 8px;
	color: var(--color-text-maxcontrast);
	font-size: 13px;
}
</style>
