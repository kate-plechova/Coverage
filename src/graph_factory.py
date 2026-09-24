"""Graph factory for the unified HLA matching backend."""

from .lol_unified import LolUnifiedGraphBuilder


class GraphFactory:
    """Build graphs using the unified LOL backend only."""

    @staticmethod
    def build(conf):
        donors_file = conf.get("donors_input_file")
        patients_file = conf.get("recipients_input_file")

        if not donors_file:
            raise ValueError("Missing 'donors_input_file' in configuration")
        if not patients_file:
            raise ValueError("Missing 'recipients_input_file' in configuration")

        return LolUnifiedGraphBuilder(conf).build_graph(
            donors_file,
            patients_file,
        )
