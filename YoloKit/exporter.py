import cv2
import logging
import xml.etree.ElementTree as etree  # nosec B405

from abc import abstractmethod
from numpy.typing import NDArray
from xml.dom import minidom
from YoloKit.Utils import get_utc_time
from YoloKit.data import BBox, Line

class Exporter:
    """
    Abstract base class for OCR result exporters.

    Defines the interface and common functionality for exporting OCR results
    to various formats. Subclasses implement specific export formats.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the exporter with output directory.

        Args:
            output_dir: Directory path where exported files will be saved
        """
        self.output_dir = output_dir
        logging.info("Init Exporter")

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "export_lines") and callable(subclass.export_lines) or NotImplemented
    

    @abstractmethod
    def export_lines(
        self,
        image: NDArray | None,
        image_name: str,
        lines: list[NDArray]
    ):
        """Exports text lines and line informations"""
        raise NotImplementedError


    def get_bbox(self, contour: NDArray) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(contour)

        if x == 0 or y == 0 or w == 0 or h == 0:
            print(f"warning: zero bbox")
        return BBox(x, y, w, h)
    
    def get_text_bbox(self, lines: list[NDArray]):
        all_bboxes = [self.get_bbox(x) for x in lines]
        min_x = min(a.x for a in all_bboxes)
        min_y = min(a.y for a in all_bboxes)

        max_w = max(a.w for a in all_bboxes)
        max_h = all_bboxes[-1].y + all_bboxes[-1].h

        bbox = BBox(min_x, min_y, max_w, max_h)

        return bbox


    @staticmethod
    def get_text_points(contour: NDArray) -> str:
        return " ".join([f"{x},{y}" for x, y in contour])

    @staticmethod
    def get_bbox_points(bbox: BBox):
        """
        Convert BBox to coordinate points string for XML export.

        Args:
            bbox: BBox object defining rectangular region

        Returns:
            String of corner coordinates in XML format
        """
        points = (
            f"{bbox.x},{bbox.y} {bbox.x + bbox.w},{bbox.y} "
            f"{bbox.x + bbox.w},{bbox.y + bbox.h} {bbox.x},{bbox.y + bbox.h}"
        )
        return points
    


class PageXMLExporter(Exporter):
    """
    Exporter for PageXML format compatible with Transkribus and other OCR tools.

    PageXML is a standardized format for representing document layout and
    OCR results with detailed coordinate information.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialize PageXML exporter.

        Args:
            output_dir: Directory path for exported XML files
        """
        super().__init__(output_dir)
        logging.info("Init XML Exporter")

    def get_text_line_block(self, coordinate, index: int, unicode_text: str):
        """
        Create XML element for a single text line.

        Args:
            coordinate: Line coordinate information
            index: Line index for ordering
            unicode_text: Recognized text content

        Returns:
            XML element representing the text line
        """
        text_line = etree.Element("Textline", id="", custom=f"readingOrder {{index:{index};}}")
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords

        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line

    def build_xml_document(
        self,
        image: NDArray,
        image_name: str,
        text_bbox: str,
        lines: list[str]
    ):
        """
        Build complete PageXML document structure.

        Args:
            image: Source image array for dimensions
            image_name: Name of the image file
            text_bbox: Bounding box coordinates for text region
            lines: List of line coordinate strings
            text_lines: List of OCR text results

        Returns:
            Formatted XML document string
        """
        root = etree.Element("PcGts")
        root.attrib["xmlns"] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib["xsi:schemaLocation"] = (
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
        )

        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = get_utc_time()

        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = text_bbox

        for l_idx, line in enumerate(lines):
            text_region.append(
                self.get_text_line_block(coordinate=line, index=l_idx, unicode_text="")
            )

        parsed_xml = minidom.parseString(etree.tostring(root))
        parsed_xml = parsed_xml.toprettyxml()

        return parsed_xml

    def export_lines(
        self,
        image: NDArray | None,
        image_name: str,
        lines: list[NDArray],
        use_bbox: bool = False
    ):

        if use_bbox:
            plain_lines = [self.get_bbox(x) for x in lines]
        else:
            plain_lines = [self.get_text_points(x) for x in lines]

        text_bbox = self.get_text_bbox(lines)
        plain_box = self.get_bbox_points(text_bbox)

        xml_doc = self.build_xml_document(
            image,
            image_name,
            text_bbox=plain_box,
            lines=plain_lines
        )

        out_file = f"{self.output_dir}/{image_name}.xml"

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(xml_doc)
