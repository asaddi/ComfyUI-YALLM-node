import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
	name: "YALLM.node",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeType.comfyClass === "LLMProvider") {
			nodeType.prototype.myRefreshModels = (node, name) => {
				const provWidget = node.widgets.find(
					(widget) => widget.name === "provider",
				);
				const modelWidget = node.widgets.find(
					(widget) => widget.name === "model",
				);
				const fetchWidget = node.widgets.find(
					(widget) => widget.name === "fetch models",
				);

				const provName = name ?? provWidget.value;

				const myDisableWidgets = (state) => {
					provWidget.disabled = state;
					modelWidget.disabled = state;
					// Note: This is the "fetch models" button we add down below
					fetchWidget.disabled = state;
				};
				myDisableWidgets(true);

				api.fetchApi(`/llm_models?name=${provName}`).then((resp) => {
					resp.json().then((data) => {
						const widget = modelWidget;
						widget.options.values = data;

						if (
							!widget.options.values.includes(widget.value) &&
							widget.options.values.length > 0
						) {
							widget.value = widget.options.values[0];
						}

						myDisableWidgets(false);
					});
				});
			};
		}
	},
	async nodeCreated(node) {
		if (node?.comfyClass === "LLMProvider") {
			const provWidget = node.widgets.find(
				(widget) => widget.name === "provider",
			);
			const modelWidget = node.widgets.find(
				(widget) => widget.name === "model",
			);

			// Whenever provider changes value, fetch models
			const original_callback = provWidget.callback;
			provWidget.callback = function (...args) {
				const name = args?.[0];
				node.myRefreshModels(node, name);
				return original_callback?.apply(this, args);
			};

			// Disable model combo widget until we fetch models
			modelWidget.disabled = true;

			// Add button to fetch models manually
			const btn = node.addWidget("button", "fetch models", "models", () => {
				node.myRefreshModels(node);
			});
			btn.serializeValue = () => void 0;
		}
	},
	async setup() {
		api.addEventListener("executed", (event) => {
			const node = app.graph.getNodeById(event.detail.node);
			if (node?.comfyClass === "LLMTextLatch") {
				const textWidget = node.widgets.find(
					(widget) => widget.name === "text",
				);
				textWidget.value = event.detail.output.text.join("");
				app.graph.setDirtyCanvas(true, false);
			}
		});
	},
});
