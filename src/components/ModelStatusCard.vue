<template>
	<div class="model-status-card" :class="{ 'model-status-card--loaded': loaded, 'model-status-card--error': !loaded }">
		<div class="model-status-card__header">
			<span class="model-status-card__indicator" :class="indicatorClass" />
			<h4 class="model-status-card__name">{{ displayName }}</h4>
			<span class="model-status-card__badge" :class="badgeClass">
				{{ quantized ? 'INT8' : 'FP32' }}
			</span>
		</div>
		<div class="model-status-card__body">
			<span v-if="loaded" class="model-status-card__status model-status-card__status--ok">
				✓ Loaded
			</span>
			<span v-else class="model-status-card__status model-status-card__status--error">
				✗ Not loaded
			</span>
		</div>
		<div v-if="error" class="model-status-card__error">
			<NcNoteCard type="error">
				{{ error }}
			</NcNoteCard>
		</div>
	</div>
</template>

<script>
import NcNoteCard from '@nextcloud/vue/dist/Components/NcNoteCard.js'

export default {
	name: 'ModelStatusCard',
	components: {
		NcNoteCard,
	},
	props: {
		name: {
			type: String,
			required: true,
		},
		loaded: {
			type: Boolean,
			default: false,
		},
		error: {
			type: String,
			default: null,
		},
		quantized: {
			type: Boolean,
			default: false,
		},
	},
	computed: {
		displayName() {
			const nameMap = {
				yolov8n: 'YOLOv8n — Object Detection',
				arcface: 'ArcFace — Facial Recognition',
				clip_visual: 'CLIP ViT-B/32 — Semantic Search',
			}
			return nameMap[this.name] || this.name
		},
		indicatorClass() {
			return this.loaded
				? 'model-status-card__indicator--ok'
				: 'model-status-card__indicator--error'
		},
		badgeClass() {
			return this.quantized
				? 'model-status-card__badge--int8'
				: 'model-status-card__badge--fp32'
		},
	},
}
</script>

<style scoped>
.model-status-card {
	border: 1px solid var(--color-border);
	border-radius: var(--border-radius-large, 10px);
	padding: 16px;
	margin-bottom: 12px;
	background-color: var(--color-main-background);
	transition: border-color 0.2s ease;
}

.model-status-card--loaded {
	border-left: 4px solid var(--color-success, #46ba61);
}

.model-status-card--error {
	border-left: 4px solid var(--color-error, #e9322d);
}

.model-status-card__header {
	display: flex;
	align-items: center;
	gap: 10px;
	margin-bottom: 8px;
}

.model-status-card__indicator {
	width: 12px;
	height: 12px;
	border-radius: 50%;
	flex-shrink: 0;
}

.model-status-card__indicator--ok {
	background-color: var(--color-success, #46ba61);
	box-shadow: 0 0 6px var(--color-success, #46ba61);
}

.model-status-card__indicator--error {
	background-color: var(--color-error, #e9322d);
	box-shadow: 0 0 6px var(--color-error, #e9322d);
}

.model-status-card__name {
	margin: 0;
	font-size: 15px;
	font-weight: 600;
	color: var(--color-main-text);
	flex-grow: 1;
}

.model-status-card__badge {
	font-size: 11px;
	font-weight: 700;
	padding: 2px 8px;
	border-radius: var(--border-radius-pill, 20px);
	text-transform: uppercase;
	letter-spacing: 0.5px;
	flex-shrink: 0;
}

.model-status-card__badge--int8 {
	background-color: var(--color-primary-element-light, #d4e8fd);
	color: var(--color-primary-element, #0082c9);
}

.model-status-card__badge--fp32 {
	background-color: var(--color-background-dark, #ededed);
	color: var(--color-text-maxcontrast, #767676);
}

.model-status-card__body {
	margin-bottom: 4px;
}

.model-status-card__status {
	font-size: 13px;
	font-weight: 500;
}

.model-status-card__status--ok {
	color: var(--color-success, #46ba61);
}

.model-status-card__status--error {
	color: var(--color-error, #e9322d);
}

.model-status-card__error {
	margin-top: 8px;
}
</style>
