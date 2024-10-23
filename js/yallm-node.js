import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
	name: "YALLM.node",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass=="LLMProvider") {
            nodeType.prototype.myRefreshModels = function (node, name) {
                if (!name) name = node.widgets[0].value;
                api.fetchApi(`/llm_models?name=${name}`).then((resp) => {
                    resp.json().then((data) => {
                        let widget = node.widgets[1];
                        widget.options.values = data;

                        if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
                            widget.value = widget.options.values[0];
                        }
                    })
                })
            }
        }
    },
    async nodeCreated(node) {
        if (node?.comfyClass==="LLMProvider") {
            let prov_widget = node.widgets[0];
            // Whenever provider changes value, fetch models
            const original_callback = prov_widget.callback;
            prov_widget.callback = function () {
                let name = arguments?.[0];
                node.myRefreshModels(node, name);
                return original_callback?.apply(this, arguments);
            }
        }
    },
    loadedGraphNode(node) {
        if (node?.comfyClass==="LLMProvider") {
            // At this point, the provider widget is correctly set.
            node.myRefreshModels(node);
        }
    }
})
