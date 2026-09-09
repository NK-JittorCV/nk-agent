from __future__ import annotations

import json

from geoagent_jittor.cli import _render_output, _unwrap_json_fence


def test_unwrap_json_fence():
    assert _unwrap_json_fence('```json\n{"city": "Madrid"}\n```') == '{"city": "Madrid"}'
    assert _unwrap_json_fence('{"city": "Madrid"}') == '{"city": "Madrid"}'


def test_render_output_formats_fenced_json(capsys):
    _render_output('```json\n{"city": "Madrid"}\n```')

    assert json.loads(capsys.readouterr().out) == {"city": "Madrid"}
