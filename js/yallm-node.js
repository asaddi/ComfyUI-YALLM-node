import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
	name: "YALLM.node",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass==="LLMProvider") {
            nodeType.prototype.myRefreshModels = function (node, name) {
                if (!name) name = node.widgets[0].value;

                const myDisableWidgets = (state) => {
                    node.widgets[0].disabled = state;
                    node.widgets[1].disabled = state;
                    // Note: This is the "fetch models" button we add down below
                    node.widgets[2].disabled = state;
                };
                myDisableWidgets(true);

                api.fetchApi(`/llm_models?name=${name}`).then((resp) => {
                    resp.json().then((data) => {
                        let widget = node.widgets[1];
                        widget.options.values = data;

                        if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
                            widget.value = widget.options.values[0];
                        }

                        myDisableWidgets(false);
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

            // Disable model combo widget until we fetch models
            node.widgets[1].disabled = true;

            // Add button to fetch models manually
            const btn = node.addWidget(
                "button",
                "fetch models",
                "models",
                () => { node.myRefreshModels(node) }
            )
            btn.serializeValue = () => void 0;
        }

        else if (node?.comfyClass==="LLMTextLatch") {
            node.widgets[0].inputEl.readOnly = true;
        }
    },
    async setup() {
        api.addEventListener("executed", function (event) {
            var node = app.graph.getNodeById(event.detail.node);
            if (node?.comfyClass==="LLMTextLatch") {
                node.widgets[0].value = event.detail.output.text.join("");
                app.graph.setDirtyCanvas(true, false);
            }
        });
    }
})
