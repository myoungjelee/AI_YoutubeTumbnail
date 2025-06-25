import ast
from pathlib import Path
import pytest

# Load sanitize_filename from the source file without importing heavy deps
ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'scripts' / 'Youtube_Crawler.py').read_text()
module = ast.parse(source)
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'sanitize_filename':
        sanitize_src = ast.get_source_segment(source, node)
        break
namespace = {}
exec('import re\n' + sanitize_src, namespace)
sanitize_filename = namespace['sanitize_filename']


def test_removes_emojis_and_forbidden_chars():
    assert sanitize_filename("Video: Title? 😀") == "Video Title"


def test_forbidden_filesystem_characters_removed():
    assert sanitize_filename("bad<>\"|*/:?filename") == "badfilename"


def test_collapse_spaces_and_strip():
    assert sanitize_filename("  Hello   World  ") == "Hello World"


def test_allowed_characters_preserved():
    assert sanitize_filename("file_name-123.jpg") == "file_name-123.jpg"


def test_remove_non_english_special_characters():
    assert sanitize_filename("sp\xc3\xa9ci\xc3\xa5l \xc3\xa7h\xc3\xa4r\xc3\xa5ct\xc4\x99rs") == "spcil hrctrs"
