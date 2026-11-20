import xml.etree.ElementTree as ET

from mflux.utils.metadata_builder import MetadataBuilder


def test_build_xmp_packet_escapes_user_controlled_fields():
    # User-controlled fields that contain XML-hostile characters. On a raw
    # f-string build (no escaping) the bare ampersand/angle brackets make the
    # XMP packet non-well-formed, so ElementTree.fromstring would raise a
    # ParseError. The regression this guards: a LoRA filename, model id and
    # version flowing verbatim into XMP element text.
    #
    # Note: user LoRA values are the file *basename*, so a path separator is
    # NOT used here; the hostile characters are what we care about.
    lora_name = "Studio & Ghibli <v2>.safetensors"
    metadata = {
        "prompt": 'a dragon & a "knight" <in> the mist',
        "lora_paths": [lora_name],
        "lora_scales": [0.8],
        "model": "acme/flux <dev> & co",
        "mflux_version": "1.2 <beta> & rc",
    }

    xmp = MetadataBuilder.build_xmp_packet(metadata)

    # 1) The produced packet must be well-formed XML. Strip the xpacket
    #    processing-instruction wrapper that ET.fromstring does not accept,
    #    keeping only the <x:xmpmeta> document element.
    start = xmp.index("<x:xmpmeta")
    end = xmp.index("</x:xmpmeta>") + len("</x:xmpmeta>")
    document = xmp[start:end]

    # Declare the namespace prefixes used inside the document so a plain
    # ElementTree parse succeeds (otherwise it rejects unbound prefixes).
    wrapped = (
        '<root xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
        'xmlns:mflux="http://ns.mflux.ai/1.0/">' + document + "</root>"
    )

    # This parse raises xml.etree.ElementTree.ParseError on the pre-fix code
    # (bare '&' is an invalid XML entity reference).
    tree = ET.fromstring(wrapped)

    # 2) The hostile characters must survive the escape/parse round-trip back
    #    to their original values (escaped only in the raw bytes, decoded by
    #    the XML parser).
    texts = {el.tag.split("}")[-1]: (el.text or "") for el in tree.iter()}

    # dc:description/rdf:li carries the prompt; <mflux:model>, <mflux:loras>,
    # and <xmp:CreatorTool> carry the other user fields.
    li_texts = [el.text or "" for el in tree.iter() if el.tag.endswith("}li")]
    assert 'a dragon & a "knight" <in> the mist' in li_texts

    assert texts["model"] == "acme/flux <dev> & co"
    assert texts["loras"] == "Studio & Ghibli <v2>.safetensors:0.8"
    assert texts["CreatorTool"] == "MFLUX 1.2 <beta> & rc"


def test_build_xmp_packet_does_not_corrupt_safe_fields():
    # Counter-example / neighbor: a value with no special characters must not
    # be altered, and the fix must not double-escape an existing entity it did
    # not introduce. Guards against a regression where the escape helper
    # mangles ordinary prompts or runs twice.
    metadata = {
        "prompt": "a cat",
        "lora_paths": ["portrait.safetensors"],
        "lora_scales": [1.0],
        "model": "dev",
        "mflux_version": "0.1.0",
    }

    xmp = MetadataBuilder.build_xmp_packet(metadata)

    start = xmp.index("<x:xmpmeta")
    end = xmp.index("</x:xmpmeta>") + len("</x:xmpmeta>")
    wrapped = (
        '<root xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
        'xmlns:mflux="http://ns.mflux.ai/1.0/">' + xmp[start:end] + "</root>"
    )
    tree = ET.fromstring(wrapped)
    texts = {el.tag.split("}")[-1]: (el.text or "") for el in tree.iter()}

    li_texts = [el.text or "" for el in tree.iter() if el.tag.endswith("}li")]
    assert "a cat" in li_texts
    assert texts["model"] == "dev"
    assert texts["loras"] == "portrait.safetensors:1.0"

    # No stray entity text leaked into the raw packet for a safe prompt: the
    # ampersand entity must be absent because none of the safe values use '&'.
    assert "&amp;amp;" not in xmp  # never double-escaped
    assert "&amp;" not in xmp  # no '&' present at all for these safe values

    # The helper itself leaves a clean string untouched and never
    # double-escapes an already-escaped entity (ampersand replaced first).
    assert MetadataBuilder._xml_escape("a cat") == "a cat"
    assert MetadataBuilder._xml_escape("a & b") == "a &amp; b"
    assert MetadataBuilder._xml_escape("a &amp; b") == "a &amp;amp; b"
