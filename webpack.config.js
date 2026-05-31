const path = require('path')
const { VueLoaderPlugin } = require('vue-loader')

module.exports = {
	entry: path.resolve(__dirname, 'src', 'main.js'),
	output: {
		path: path.resolve(__dirname, 'js'),
		publicPath: '/js/',
		filename: 'recognize_ai-admin.js',
		clean: true,
	},
	module: {
		rules: [
			{
				test: /\.vue$/,
				loader: 'vue-loader',
			},
			{
				test: /\.js$/,
				loader: 'babel-loader',
				exclude: /node_modules/,
			},
			{
				test: /\.css$/,
				use: ['style-loader', 'css-loader'],
			},
		],
	},
	plugins: [
		new VueLoaderPlugin(),
	],
	resolve: {
		alias: {
			vue$: 'vue/dist/vue.esm.js',
		},
		extensions: ['*', '.js', '.vue', '.json'],
	},
}
