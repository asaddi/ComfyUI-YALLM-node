import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
	name: "YALLM.node",
    async nodeCreated(node) {
        if (node?.comfyClass==="LLMProvider") {
            // Are these indexes fixed?!
            let prov_widget = node.widgets[0];

            // Whenever provider changes value, fetch models
            const original_callback = prov_widget.callback;
            prov_widget.callback = function () {
                let value = arguments?.[0];
                api.fetchApi(`/llm_models?name=${value}`).then((resp) => {
                    resp.json().then((data) => {
                        let widget = node.widgets[1];
                        widget.options.values = data;

                        if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
                            widget.value = widget.options.values[0];
                        }
                    })
                })

                return original_callback?.apply(this, arguments);
            }
        }
    },
    loadedGraphNode(node) {
        if (node?.comfyClass==="LLMProvider") {
            let prov_widget = node.widgets[0];

            // Do a fetch right now as well
            api.fetchApi(`/llm_models?name=${prov_widget.value}`).then((resp) => {
                resp.json().then((data) => {
                    let widget = node.widgets[1];
                    widget.options.values = data

                    // Verify that widget is valid
                    // TODO Cleaner way of telling Python side?
                    if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
                        widget.value = widget.options.values[0];
                    }
                })
            })
        }
    },
})
