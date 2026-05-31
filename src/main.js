import Vue from 'vue'
import App from './App.vue'

// Disable Vue production tip in console
Vue.config.productionTip = false

document.addEventListener('DOMContentLoaded', () => {
	const mountEl = document.getElementById('recognize_ai-admin-settings')
	if (!mountEl) {
		return
	}

	new Vue({
		el: mountEl,
		render: h => h(App),
	})
})
