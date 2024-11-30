class PromptUtils:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_node_info(self, node_id: int | str) -> dict:
        node_id = str(node_id)
        return self.prompt[node_id]

    def is_input_connected(self, node_id: int | str, name: str) -> bool:
        node_info = self.get_node_info(node_id)
        input = node_info.get("inputs", {}).get(name)
        # In reality, input will be a 2-elem list of source node ID + slot
        # if it is connected.
        return input is not None

    @property
    def output_map(self):
        if hasattr(self, "_output_map"):
            return self._output_map
        # A map of (src node ID, slot) -> (dest node ID, "input name")
        self._output_map: dict[tuple[str, int], list[tuple[str, str]]] = {}
        for dest_id, dest_info in self.prompt.items():
            for input_name, input in dest_info.get("inputs", {}).items():
                if isinstance(input, list) and len(input) == 2:
                    # This is apparently a link
                    src_id, src_slot = input
                    out_key = (src_id, src_slot)
                    outputs = self._output_map.get(out_key)
                    if outputs is None:
                        self._output_map[out_key] = outputs = []
                    outputs.append((dest_id, input_name))
        return self._output_map

    def is_output_connected(self, node_id: int | str, slot: int) -> bool:
        out_key = (str(node_id), slot)
        return out_key in self.output_map

    # The assumption is that the given output slot can only be connected
    # to nodes of the same type as the one being checked.
    def get_downstream_nodes(self, node_id: int | str, slot: int) -> list[str]:
        downstream: list[str] = []

        to_check = [str(node_id)]
        while to_check:
            check_id = to_check.pop(0)

            # Yes, consider the starting node downstream from itself as well
            downstream.append(check_id)

            outputs = self.output_map.get((check_id, slot))
            if outputs is not None:
                # NB There can be multiple outputs
                out_ids: list[str] = [link_info[0] for link_info in outputs]
                to_check.extend(out_ids)

        return downstream


class WorkflowUtils:
    def __init__(self, extra_pnginfo):
        # Note: API mode does not have extra_pnginfo (it is None)
        self.extra_pnginfo = extra_pnginfo

    def get_node_info(self, node_id: str | int) -> dict | None:
        # Be somewhat paranoid since API mode offers much less access
        if (
            self.extra_pnginfo is not None
            and (workflow := self.extra_pnginfo.get("workflow")) is not None
            and (nodes := workflow.get("nodes")) is not None
        ):
            node_id = int(node_id)
            # TODO For the time being, we aren't caching anything.
            # Concievably, if there are a large number of nodes, a lot of time
            # could be wasted searching.
            my_node = [node_info for node_info in nodes if node_info["id"] == node_id]
            if my_node:
                return my_node[0]
        return None

    def set_property(self, node_id: str | int, property_name: str, value):
        if (my_node := self.get_node_info(node_id)) is not None:
            my_node["properties"][property_name] = value

    def set_widget(self, node_id: str | int, slot: int, value):
        if (my_node := self.get_node_info(node_id)) is not None:
            values = my_node.get("widgets_values", [])
            # TODO When would there be a mismatch?
            if slot < len(values):
                values[slot] = value
