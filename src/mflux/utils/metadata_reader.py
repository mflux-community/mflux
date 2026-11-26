import json
import logging
from pathlib import Path

import piexif
import PIL.Image

log = logging.getLogger(__name__)


class MetadataReader:
    @staticmethod
    def read_exif_metadata(image_path: str | Path) -> dict | None:
        try:
            with PIL.Image.open(image_path) as img:
                exif_bytes = img.info.get("exif")

                if not exif_bytes:
                    return None

                exif_dict = piexif.load(exif_bytes)
                user_comment = exif_dict["Exif"].get(0x9286, b"")

                if user_comment:
                    # Try to parse as JSON (strip the ASCII prefix if present)
                    if user_comment.startswith(b"ASCII\x00\x00\x00"):
                        metadata_str = user_comment[8:].decode("utf-8")
                    else:
                        metadata_str = user_comment.decode("utf-8")
                    return json.loads(metadata_str)

                return None

        except (OSError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.debug(f"Error reading EXIF metadata: {e}")
            return None

    @staticmethod
    def read_xmp_metadata(image_path: str | Path) -> dict | None:
        try:
            with PIL.Image.open(image_path) as img:
                xmp_data = img.info.get("XML:com.adobe.xmp")

                if not xmp_data:
                    return None

                # Parse XMP XML to extract key fields
                xmp_dict = {}

                # Simple XML parsing for common fields
                fields = {
                    "description": '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">',
                    "creator": "<dc:creator><rdf:Seq><rdf:li>",
                    "rights": '<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">',
                    "creator_tool": "<xmp:CreatorTool>",
                    "category": "<photoshop:Category>",
                    "credit": "<photoshop:Credit>",
                    "seed": "<mflux:seed>",
                    "steps": "<mflux:steps>",
                    "guidance": "<mflux:guidance>",
                    "model": "<mflux:model>",
                    "loras": "<mflux:loras>",
                    "generation_time": "<mflux:generationTime>",
                }

                for key, start_tag in fields.items():
                    if start_tag in xmp_data:
                        start_idx = xmp_data.index(start_tag) + len(start_tag)
                        # Find the closing tag
                        if key in ["description", "rights"]:
                            end_tag = "</rdf:li>"
                        elif key == "creator":
                            end_tag = "</rdf:li>"
                        else:
                            # Turn the opening tag "<ns:name>" into its closing form
                            # "</ns:name>". Splitting on ":" keeps the leading "<" in
                            # the namespace part, so building the closing tag from the
                            # pieces produced "</<ns:name>" and index() never found it,
                            # which made the whole XMP read fail at <xmp:CreatorTool>.
                            end_tag = "</" + start_tag[1:]

                        end_idx = xmp_data.index(end_tag, start_idx)
                        value = xmp_data[start_idx:end_idx]
                        xmp_dict[key] = value

                return xmp_dict if xmp_dict else None

        except (OSError, KeyError, ValueError) as e:
            log.debug(f"Error reading XMP metadata: {e}")
            return None

    @staticmethod
    def read_all_metadata(image_path: str | Path) -> dict:
        return {
            "exif": MetadataReader.read_exif_metadata(image_path),
            "xmp": MetadataReader.read_xmp_metadata(image_path),
        }
