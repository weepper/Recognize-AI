<template>
	<div class="scanner-panel">
		<!-- Enable/Disable Toggle -->
		<div class="scanner-panel__toggle">
			<NcCheckboxRadioSwitch :checked.sync="enabled"
				type="switch"
				@update:checked="onToggle">
				Enable background file scanner
			</NcCheckboxRadioSwitch>
		</div>

		<!-- Scanner Configuration -->
		<div class="scanner-panel__config">
			<div class="scanner-panel__field">
				<label for="scanner-batch-size" class="scanner-panel__label">
					Batch size
				</label>
				<NcTextField id="scanner-batch-size"
					:value.sync="batchSize"
					type="number"
					label="Batch size"
					:show-trailing-button="false"
					placeholder="10" />
				<p class="scanner-panel__hint">
					Number of files to process per batch
				</p>
			</div>
			<div class="scanner-panel__field">
				<label for="scanner-interval" class="scanner-panel__label">
					Polling interval (seconds)
				</label>
				<NcTextField id="scanner-interval"
					:value.sync="interval"
					type="number"
					label="Interval (seconds)"
					:show-trailing-button="false"
					placeholder="30" />
				<p class="scanner-panel__hint">
					Time between scan cycles in seconds
				</p>
			</div>
		</div>

		<!-- Apply Settings Button -->
		<div class="scanner-panel__actions">
			<NcButton type="primary"
				:disabled="saving"
				@click="saveSettings">
				<template #icon>
					<span v-if="saving" class="icon-loading-small" />
				</template>
				{{ saving ? 'Saving…' : 'Apply settings' }}
			</NcButton>
			<span v-if="saveSuccess" class="scanner-panel__save-ok">✓ Saved</span>
			<span v-if="saveError" class="scanner-panel__save-err">{{ saveError }}</span>
		</div>

		<!-- Scanner Status -->
		<div class="scanner-panel__status">
			<h4 class="scanner-panel__status-title">Scanner Status</h4>

			<div v-if="statusLoading && !statusData" class="scanner-panel__loading">
				<span class="icon-loading-small" /> Loading status…
			</div>

			<template v-else-if="statusData">
				<div class="scanner-panel__stats">
					<div class="scanner-panel__stat">
						<span class="scanner-panel__stat-value">{{ statusData.files_processed || 0 }}</span>
						<span class="scanner-panel__stat-label">Files processed</span>
					</div>
					<div class="scanner-panel__stat">
						<span class="scanner-panel__stat-value">{{ statusData.queue_remaining || 0 }}</span>
						<span class="scanner-panel__stat-label">Queue remaining</span>
					</div>
					<div class="scanner-panel__stat">
						<span class="scanner-panel__stat-value">{{ statusData.running ? 'Active' : 'Idle' }}</span>
						<span class="scanner-panel__stat-label">Status</span>
					</div>
				</div>

				<div v-if="progressPercent !== null" class="scanner-panel__progress">
					<NcProgressBar :value="progressPercent" size="medium" />
					<span class="scanner-panel__progress-text">{{ progressPercent }}%</span>
				</div>

				<p v-if="statusData.last_run" class="scanner-panel__last-run">
					Last run: {{ statusData.last_run }}
				</p>
			</template>

			<NcNoteCard v-if="statusError" type="warning">
				Could not fetch scanner status: {{ statusError }}
			</NcNoteCard>
		</div>
	</div>
</template>

<script>
import axios from '@nextcloud/axios'
import { generateUrl } from '@nextcloud/router'
import NcButton from '@nextcloud/vue/dist/Components/NcButton.js'
import NcCheckboxRadioSwitch from '@nextcloud/vue/dist/Components/NcCheckboxRadioSwitch.js'
import NcNoteCard from '@nextcloud/vue/dist/Components/NcNoteCard.js'
import NcProgressBar from '@nextcloud/vue/dist/Components/NcProgressBar.js'
import NcTextField from '@nextcloud/vue/dist/Components/NcTextField.js'

export default {
	name: 'ScannerPanel',
	components: {
		NcButton,
		NcCheckboxRadioSwitch,
		NcNoteCard,
		NcProgressBar,
		NcTextField,
	},
	data() {
		return {
			enabled: false,
			batchSize: '10',
			interval: '30',
			saving: false,
			saveSuccess: false,
			saveError: '',
			statusData: null,
			statusLoading: false,
			statusError: '',
			pollTimer: null,
		}
	},
	computed: {
		progressPercent() {
			if (!this.statusData) {
				return null
			}
			const processed = this.statusData.files_processed || 0
			const remaining = this.statusData.queue_remaining || 0
			const total = processed + remaining
			if (total === 0) {
				return 0
			}
			return Math.round((processed / total) * 100)
		},
	},
	mounted() {
		this.fetchStatus()
	},
	beforeDestroy() {
		this.stopPolling()
	},
	methods: {
		async onToggle(value) {
			this.enabled = value
			await this.saveSettings()
		},

		async saveSettings() {
			this.saving = true
			this.saveSuccess = false
			this.saveError = ''

			try {
				const url = generateUrl('/apps/recognize_ai/api/v1/scanner/toggle')
				await axios.post(url, {
					enabled: this.enabled,
					batch_size: parseInt(this.batchSize, 10) || 10,
					interval: parseInt(this.interval, 10) || 30,
				})
				this.saveSuccess = true
				setTimeout(() => { this.saveSuccess = false }, 3000)

				// Start or stop polling based on enabled state
				if (this.enabled) {
					this.startPolling()
				} else {
					this.stopPolling()
				}
			} catch (err) {
				this.saveError = err.response?.data?.detail || err.message || 'Failed to save settings'
			} finally {
				this.saving = false
			}
		},

		async fetchStatus() {
			this.statusLoading = true
			this.statusError = ''
			try {
				const url = generateUrl('/apps/recognize_ai/api/v1/scanner/status')
				const { data } = await axios.get(url)
				this.statusData = data

				// Sync enabled state from backend if available
				if (data.enabled !== undefined) {
					this.enabled = data.enabled
				}
				if (data.batch_size !== undefined) {
					this.batchSize = String(data.batch_size)
				}
				if (data.interval !== undefined) {
					this.interval = String(data.interval)
				}

				// Start polling if scanner is running
				if (this.enabled && !this.pollTimer) {
					this.startPolling()
				}
			} catch (err) {
				this.statusError = err.response?.data?.detail || err.message || 'Connection error'
			} finally {
				this.statusLoading = false
			}
		},

		startPolling() {
			this.stopPolling()
			this.pollTimer = setInterval(() => {
				this.fetchStatus()
			}, 5000)
		},

		stopPolling() {
			if (this.pollTimer) {
				clearInterval(this.pollTimer)
				this.pollTimer = null
			}
		},
	},
}
</script>

<style scoped>
.scanner-panel__toggle {
	margin-bottom: 20px;
}

.scanner-panel__config {
	display: flex;
	gap: 20px;
	margin-bottom: 20px;
	flex-wrap: wrap;
}

.scanner-panel__field {
	flex: 1;
	min-width: 180px;
	max-width: 280px;
}

.scanner-panel__label {
	display: block;
	font-weight: 600;
	font-size: 13px;
	margin-bottom: 4px;
	color: var(--color-main-text);
}

.scanner-panel__hint {
	font-size: 12px;
	color: var(--color-text-maxcontrast, #767676);
	margin-top: 4px;
}

.scanner-panel__actions {
	display: flex;
	align-items: center;
	gap: 12px;
	margin-bottom: 24px;
}

.scanner-panel__save-ok {
	color: var(--color-success, #46ba61);
	font-weight: 600;
	font-size: 13px;
}

.scanner-panel__save-err {
	color: var(--color-error, #e9322d);
	font-size: 13px;
}

.scanner-panel__status {
	border-top: 1px solid var(--color-border);
	padding-top: 16px;
}

.scanner-panel__status-title {
	font-size: 15px;
	font-weight: 600;
	margin: 0 0 12px 0;
	color: var(--color-main-text);
}

.scanner-panel__loading {
	display: flex;
	align-items: center;
	gap: 8px;
	color: var(--color-text-maxcontrast);
	font-size: 13px;
}

.scanner-panel__stats {
	display: flex;
	gap: 24px;
	margin-bottom: 16px;
	flex-wrap: wrap;
}

.scanner-panel__stat {
	display: flex;
	flex-direction: column;
	align-items: center;
	min-width: 100px;
	padding: 12px 16px;
	background-color: var(--color-background-dark, #f5f5f5);
	border-radius: var(--border-radius-large, 10px);
}

.scanner-panel__stat-value {
	font-size: 22px;
	font-weight: 700;
	color: var(--color-main-text);
}

.scanner-panel__stat-label {
	font-size: 12px;
	color: var(--color-text-maxcontrast, #767676);
	margin-top: 4px;
}

.scanner-panel__progress {
	display: flex;
	align-items: center;
	gap: 12px;
	margin-bottom: 12px;
}

.scanner-panel__progress-text {
	font-size: 13px;
	font-weight: 600;
	color: var(--color-main-text);
	white-space: nowrap;
}

.scanner-panel__last-run {
	font-size: 12px;
	color: var(--color-text-maxcontrast, #767676);
}
</style>
