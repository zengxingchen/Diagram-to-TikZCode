#!/usr/bin/env python3
"""
Geometry-similarity batch worker
Simplified from geometry_sim_worker.py for use as an RL reward.
Input  : base64-encoded bytes of two PDFs.
Output : a single similarity score in [0, 1].
"""

import sys
import json
import base64
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from math import sqrt, pi, atan2
from scipy.optimize import linear_sum_assignment
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF (fitz) not available. Please install with: pip install PyMuPDF")
    fitz = None


@dataclass
class GeometricElement:
    """A geometric element extracted from a PDF."""
    element_type: str  # 'line', 'rectangle', 'circle', 'ellipse', 'curve', 'arrow', 'polygon', 'closed_curve'
    coordinates: List[float]  # coordinates
    properties: Dict[str, Any]  # extra properties (colour, line width, etc.)
    bbox: Tuple[float, float, float, float]  # bounding box
    centroid: Tuple[float, float]  # centroid
    area: float  # area (or length for lines)
    orientation: float  # orientation (for oriented elements)


@dataclass
class GeometrySimilarityResult:
    """Result of a geometry-similarity comparison."""
    overall_similarity: float
    type_similarities: Dict[str, float]  # similarity per shape type
    element_counts: Dict[str, Tuple[int, int]]  # element counts per type (candidate, reference)
    total_elements: Tuple[int, int]  # total element counts (candidate, reference)


class SimplifiedGeometrySimilarity:
    """Simplified geometry-similarity computer (training use)."""
    
    def __init__(self):
        """Initialise the computer."""
        pass
    
    def calculate_similarity_from_pdf_bytes(self, candidate_pdf_bytes: bytes, reference_pdf_bytes: bytes) -> float:
        """
        Compute geometry similarity from PDF byte data.
        
        Args:
            candidate_pdf_bytes : candidate PDF bytes
            reference_pdf_bytes : reference PDF bytes
            
        Returns:
            float: similarity score in [0, 1]
        """
        try:
            # Extract geometric elements
            candidate_elements = self._extract_geometric_elements(candidate_pdf_bytes)
            reference_elements = self._extract_geometric_elements(reference_pdf_bytes)
            
            # Compute similarity
            result = self._calculate_similarity_with_matching(candidate_elements, reference_elements)
            
            return result.overall_similarity
            
        except Exception as e:
            print(f"Error calculating geometry similarity: {e}")
            return 0.0
    
    def _extract_geometric_elements(self, pdf_bytes: bytes) -> List[GeometricElement]:
        """Extract geometric elements from a PDF page."""
        elements = []
        
        if fitz is None:
            print("Error: PyMuPDF (fitz) not available for PDF processing")
            return elements
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            if len(doc) == 0:
                return elements
            
            page = doc[0]
            
            # Get drawing commands
            drawings = page.get_drawings()
            
            for drawing in drawings:
                if not isinstance(drawing, dict):
                    continue
                    
                items = drawing.get("items", [])
                if not items:
                    continue
                
                # Parse drawing path
                self._parse_drawing_path(drawing, items, elements)

            # Post-process: detect circles
            self._post_process_detect_circles(elements)
            
            # Drop duplicate elements
            elements = self._remove_duplicate_elements(elements)

            doc.close()
            
        except Exception as e:
            print(f"Error extracting geometric elements: {e}")
        
        return elements
    
    def _parse_drawing_path(self, drawing: Dict, items: List, elements: List[GeometricElement]):
        """Parse a single drawing path into geometric elements."""
        path_segments = []
        current_point = [0.0, 0.0]
        path_start_point = [0.0, 0.0]
        
        for item in items:
            if isinstance(item, tuple) and len(item) >= 2:
                item_type = item[0]
                item_data = item[1:]
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                item_data = item
            else:
                continue
            
            if item_type == "m":  # moveto
                if isinstance(item_data, tuple) and len(item_data) >= 1:
                    point = item_data[0]
                    new_point = [float(point.x), float(point.y)]
                    current_point = new_point.copy()
                    path_start_point = new_point.copy()
                    if path_segments:
                        self._process_path_segments(path_segments, drawing, elements)
                        path_segments = []
                    
            elif item_type == "l":  # lineto
                if isinstance(item_data, tuple) and len(item_data) >= 2:
                    start_point = item_data[0]
                    end_point = item_data[1]
                    start_coords = [float(start_point.x), float(start_point.y)]
                    end_coords = [float(end_point.x), float(end_point.y)]
                    
                    path_segments.append({
                        'type': 'line',
                        'start': start_coords,
                        'end': end_coords
                    })
                    current_point = end_coords.copy()
                    
            elif item_type == "c":  # curveto (cubic Bezier)
                if isinstance(item_data, tuple) and len(item_data) >= 4:
                    curve_points = []
                    for point in item_data:
                        curve_points.append([float(point.x), float(point.y)])
                    
                    path_segments.append({
                        'type': 'curve',
                        'points': curve_points
                    })
                    current_point = curve_points[-1].copy()
                    
            elif item_type == "re":  # rectangle
                if isinstance(item_data, tuple) and len(item_data) >= 1 and hasattr(item_data[0], 'x0'):
                    rect_obj = item_data[0]
                    rect = [rect_obj.x0, rect_obj.y0, rect_obj.x1 - rect_obj.x0, rect_obj.y1 - rect_obj.y0]
                    element = self._create_rectangle_element(rect, drawing)
                    elements.append(element)

            elif item_type == "qu":  # quad (may be a rectangle)
                if isinstance(item_data, tuple) and len(item_data) >= 1:
                    quad_obj = item_data[0] if len(item_data) > 0 else item_data
                    
                    # Prefer the drawing's `rect` info when available (more accurate)
                    rect_info = drawing.get('rect')
                    if rect_info and hasattr(rect_info, 'x0'):
                        rect = [rect_info.x0, rect_info.y0, rect_info.x1 - rect_info.x0, rect_info.y1 - rect_info.y0]
                        element = self._create_rectangle_element(rect, drawing)
                        elements.append(element)
                    else:
                        # Fallback: extract points from the Quad object
                        try:
                            # PyMuPDF's Quad iterates to yield 4 corner points
                            if hasattr(quad_obj, '__iter__'):
                                points = [[point.x, point.y] for point in quad_obj]
                                if len(points) == 4:
                                    # Check whether the quad is actually a rectangle
                                    rectangle = self._detect_strict_rectangle_from_points(points, drawing)
                                    if rectangle:
                                        elements.append(rectangle)
                                    else:
                                        # If not a strict rectangle, treat as a polygon
                                        polygon = self._create_polygon_element(points, drawing)
                                        elements.append(polygon)
                        except Exception:
                            # If extraction fails, skip this Quad
                            continue
                        
            elif item_type == "s" or item_type == "h":  # closepath
                if path_segments and current_point != path_start_point:
                    path_segments.append({
                        'type': 'line',
                        'start': current_point,
                        'end': path_start_point
                    })
                if path_segments:
                    self._process_closed_path(path_segments, drawing, elements)
                    path_segments = []
        
        # Process any remaining path segment
        if path_segments:
            if self._is_potentially_closed_path(path_segments):
                self._process_closed_path(path_segments, drawing, elements)
            else:
                self._process_path_segments(path_segments, drawing, elements)
    
    def _is_potentially_closed_path(self, path_segments: List[Dict]) -> bool:
        """Heuristic check whether a path segment is likely closed."""
        if len(path_segments) < 3:
            return False
        
        first_segment = path_segments[0]
        last_segment = path_segments[-1]
        
        if first_segment['type'] == 'line':
            start_point = first_segment['start']
        elif first_segment['type'] == 'curve':
            start_point = first_segment['points'][0]
        else:
            return False
        
        if last_segment['type'] == 'line':
            end_point = last_segment['end']
        elif last_segment['type'] == 'curve':
            end_point = last_segment['points'][-1]
        else:
            return False
        
        distance = ((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)**0.5
        return distance < 5.0
    
    def _process_closed_path(self, path_segments: List[Dict], drawing: Dict, elements: List[GeometricElement]):
        """Handle a closed path."""
        if len(path_segments) < 3:
            self._process_path_segments(path_segments, drawing, elements)
            return
        
        # detect a circle
        strict_circle = self._detect_strict_circle_from_path(path_segments, drawing)
        if strict_circle:
            elements.append(strict_circle)
            return
            
        # detect a polygon
        line_polygon = self._detect_strict_line_polygon_from_path(path_segments, drawing)
        if line_polygon:
            elements.append(line_polygon)
            return
        
        # any other closed shape
        closed_curve = self._create_closed_curve_group(path_segments, drawing)
        elements.append(closed_curve)
    
    def _process_path_segments(self, path_segments: List[Dict], drawing: Dict, elements: List[GeometricElement]):
        """Handle a generic path segment."""
        for segment in path_segments:
            if segment['type'] == 'line':
                element = self._create_line_element(segment['start'], segment['end'], drawing)
                elements.append(element)
            elif segment['type'] == 'curve':
                element = self._create_curve_element(segment['points'], drawing)
                elements.append(element)
    
    def _detect_strict_circle_from_path(self, path_segments: List[Dict], drawing: Dict) -> Optional[GeometricElement]:
        """Strict circle detection."""
        curve_segments = [seg for seg in path_segments if seg['type'] == 'curve']
        if len(curve_segments) < 3:
            return None
        
        all_points = []
        for segment in curve_segments:
            all_points.extend(segment['points'])
        
        if len(all_points) < 12:
            return None
        
        return self._fit_strict_circle_to_points(all_points, drawing)
    
    def _detect_strict_line_polygon_from_path(self, path_segments: List[Dict], drawing: Dict) -> Optional[GeometricElement]:
        """Strict polygon detection."""
        line_segments = [seg for seg in path_segments if seg['type'] == 'line']
        if len(line_segments) != len(path_segments) or len(line_segments) < 3:
            return None
        
        points = []
        for segment in line_segments:
            points.append(segment['start'])
        
        if len(points) >= 3:
            # Check if rectangle
            rectangle = self._detect_strict_rectangle_from_points(points, drawing)
            if rectangle:
                return rectangle
            
            # Generic polygon
            return self._create_polygon_element(points, drawing)
        
        return None
    
    def _detect_strict_rectangle_from_points(self, points: List[Tuple[float, float]], drawing: Dict) -> Optional[GeometricElement]:
        """Strict rectangle detection."""
        if len(points) != 4:
            return None
        
        try:
            edges = []
            edge_lengths = []
            
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
                edge_length = sqrt(edge_vector[0]**2 + edge_vector[1]**2)
                edges.append(edge_vector)
                edge_lengths.append(edge_length)
            
            # Check side lengths
            sorted_lengths = sorted(edge_lengths)
            if not (abs(sorted_lengths[0] - sorted_lengths[1]) < 1.0 and 
                   abs(sorted_lengths[2] - sorted_lengths[3]) < 1.0):
                return None
            
            # Check right-angle property
            for i in range(4):
                edge1 = edges[i]
                edge2 = edges[(i + 1) % 4]
                
                dot_product = edge1[0] * edge2[0] + edge1[1] * edge2[1]
                edge1_length = sqrt(edge1[0]**2 + edge1[1]**2)
                edge2_length = sqrt(edge2[0]**2 + edge2[1]**2)
                
                if edge1_length > 0 and edge2_length > 0:
                    cos_angle = abs(dot_product) / (edge1_length * edge2_length)
                    if cos_angle > 0.04:  # cos(88.7°) ≈ 0.04
                        return None
            
            # Build the rectangle element
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            centroid = (min_x + width/2, min_y + height/2)
            
            if area < 1.0:
                return None
            
            return GeometricElement(
                element_type='rectangle',
                coordinates=[min_x, min_y, width, height],
                properties=drawing.copy(),
                bbox=(min_x, min_y, max_x, max_y),
                centroid=centroid,
                area=area,
                orientation=0.0
            )
            
        except (ValueError, ZeroDivisionError):
            return None
    
    def _create_closed_curve_group(self, path_segments: List[Dict], drawing: Dict) -> GeometricElement:
        """Build a closed curve-group element."""
        all_points = []
        
        for segment in path_segments:
            if segment['type'] == 'line':
                all_points.extend([segment['start'], segment['end']])
            elif segment['type'] == 'curve':
                all_points.extend(segment['points'])
        
        if not all_points:
            bbox = (0, 0, 1, 1)
            centroid = (0.5, 0.5)
            area = 1.0
            element_type = 'closed_curve'
            coordinates = []
        else:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            element_type = 'closed_curve'
            coordinates = []
            for p in all_points:
                coordinates.extend(p)
        
        return GeometricElement(
            element_type=element_type,
            coordinates=coordinates,
            properties=drawing.copy(),
            bbox=bbox,
            centroid=centroid,
            area=area,
            orientation=0.0
        )
    
    def _fit_strict_circle_to_points(self, points: List[Tuple[float, float]], drawing: Dict) -> Optional[GeometricElement]:
        """Strict circle fitting."""
        if len(points) < 12:
            return None
        
        try:
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # Fit circle
            A = np.column_stack([x_coords, y_coords, np.ones(len(points))])
            B = x_coords**2 + y_coords**2
            
            coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
            
            center_x = coeffs[0] / 2
            center_y = coeffs[1] / 2
            radius = np.sqrt(coeffs[2] + center_x**2 + center_y**2)
            
            if radius < 8.0 or radius > 300:
                return None
            
            # Check fit residual
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            errors = np.abs(distances - radius)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            if max_error / radius > 0.003 or mean_error / radius > 0.0015:
                return None
            
            bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            area = np.pi * radius**2
            
            return GeometricElement(
                element_type='circle',
                coordinates=[center_x, center_y, radius],
                properties=drawing.copy(),
                bbox=bbox,
                centroid=(center_x, center_y),
                area=area,
                orientation=0.0
            )
            
        except (np.linalg.LinAlgError, ValueError, IndexError):
            return None
    
    def _create_polygon_element(self, points: List[Tuple[float, float]], drawing: Dict) -> GeometricElement:
        """Build a polygon element."""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        
        # Compute area via the shoelace formula
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        
        coordinates = []
        for p in points:
            coordinates.extend(p)
        
        return GeometricElement(
            element_type='polygon',
            coordinates=coordinates,
            properties=drawing.copy(),
            bbox=bbox,
            centroid=centroid,
            area=area,
            orientation=0.0
        )

    def _create_line_element(self, start: List[float], end: List[float], drawing: Dict) -> GeometricElement:
        """Build a line element."""
        x1, y1 = start
        x2, y2 = end
        
        length = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        orientation = atan2(y2 - y1, x2 - x1)
        if orientation < 0:
            orientation += pi
        
        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        return GeometricElement(
            element_type="line",
            coordinates=[x1, y1, x2, y2],
            properties=drawing,
            bbox=bbox,
            centroid=centroid,
            area=length,
            orientation=orientation
        )
    
    def _create_curve_element(self, points: List[List[float]], drawing: Dict) -> GeometricElement:
        """Build a curve element."""
        length = 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            length += sqrt(dx*dx + dy*dy)
        
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)
        centroid = (centroid_x, centroid_y)
        
        if len(points) >= 2:
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            orientation = atan2(dy, dx)
            if orientation < 0:
                orientation += pi
        else:
            orientation = 0
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        return GeometricElement(
            element_type="curve",
            coordinates=[coord for point in points for coord in point],
            properties=drawing,
            bbox=bbox,
            centroid=centroid,
            area=length,
            orientation=orientation
        )
    
    def _create_rectangle_element(self, rect: List[float], drawing: Dict) -> GeometricElement:
        """Build a rectangle element."""
        x, y, w, h = rect
        area = abs(w * h)
        centroid = (x + w/2, y + h/2)
        
        if h != 0:
            aspect_ratio = abs(w / h)
            if aspect_ratio > 1:
                orientation = 0  # horizontal rectangle
            else:
                orientation = pi/2  # vertical rectangle
        else:
            orientation = 0
        
        bbox = (x, y, x + w, y + h)
        
        return GeometricElement(
            element_type="rectangle",
            coordinates=list(rect),
            properties=drawing,
            bbox=bbox,
            centroid=centroid,
            area=area,
            orientation=orientation
        )
    
    def _post_process_detect_circles(self, elements: List[GeometricElement]):
        """Post-pass: discover circles among the curve segments."""
        curve_elements = [elem for elem in elements if elem.element_type == 'curve']
        
        if len(curve_elements) < 3:
            return
        
        groups = self._group_nearby_curves(curve_elements)
        
        for group in groups:
            if len(group) >= 3:
                all_points = []
                for elem in group:
                    coords = elem.coordinates
                    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    all_points.extend(points)
                
                if len(all_points) >= 12:
                    shape = self._fit_strict_circle_to_points(all_points, group[0].properties)
                    if shape:
                        for elem in group:
                            if elem in elements:
                                elements.remove(elem)
                        elements.append(shape)
    
    def _group_nearby_curves(self, curve_elements: List[GeometricElement]) -> List[List[GeometricElement]]:
        """Group adjacent curve segments."""
        if not curve_elements:
            return []
        
        groups = []
        used = set()
        
        for i, elem1 in enumerate(curve_elements):
            if i in used:
                continue
                
            group = [elem1]
            used.add(i)
            
            coords1 = elem1.coordinates
            start1 = (coords1[0], coords1[1])
            end1 = (coords1[-2], coords1[-1])
            
            for j, elem2 in enumerate(curve_elements):
                if j in used:
                    continue
                
                coords2 = elem2.coordinates
                start2 = (coords2[0], coords2[1])
                end2 = (coords2[-2], coords2[-1])
                
                connection_threshold = 5.0
                
                connections = [
                    sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2),
                    sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2),
                    sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2),
                    sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2)
                ]
                
                min_connection_dist = min(connections)
                
                if min_connection_dist < connection_threshold:
                    group.append(elem2)
                    used.add(j)
            
            if len(group) >= 3:
                groups.append(group)
        
        return groups
    
    def _remove_duplicate_elements(self, elements: List[GeometricElement]) -> List[GeometricElement]:
        """Drop duplicate geometric elements."""
        if not elements:
            return elements
        
        unique_elements = []
        tolerance = 2.0
        
        for elem in elements:
            is_duplicate = False
            
            for unique_elem in unique_elements:
                if self._are_elements_similar(elem, unique_elem, tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(elem)
        
        return unique_elements
    
    def _are_elements_similar(self, elem1: GeometricElement, elem2: GeometricElement, tolerance: float) -> bool:
        """Check whether two elements are similar."""
        if elem1.element_type != elem2.element_type:
            return False
        
        bbox1 = elem1.bbox
        bbox2 = elem2.bbox
        
        bbox_diff = max(
            abs(bbox1[0] - bbox2[0]),
            abs(bbox1[1] - bbox2[1]),
            abs(bbox1[2] - bbox2[2]),
            abs(bbox1[3] - bbox2[3])
        )
        
        if bbox_diff > tolerance:
            return False
        
        area_diff = abs(elem1.area - elem2.area)
        max_area = max(elem1.area, elem2.area)
        if max_area > 0 and area_diff / max_area > 0.1:
            return False
        
        if len(elem1.coordinates) != len(elem2.coordinates):
            return False
        
        coords_diff = 0
        for i in range(min(8, len(elem1.coordinates))):
            coords_diff += abs(elem1.coordinates[i] - elem2.coordinates[i])
        
        avg_coord_diff = coords_diff / min(8, len(elem1.coordinates))
        
        return avg_coord_diff < tolerance

    def _calculate_similarity_with_matching(self,
                                          candidate_elements: List[GeometricElement],
                                          reference_elements: List[GeometricElement]) -> GeometrySimilarityResult:
        """Type-wise Hungarian matching, then paper-aligned reward aggregation.
        Matches are produced per shape type by the Hungarian algorithm using
        the type-specific cost function C.
        """
        total_candidate_count = len(candidate_elements)
        total_reference_count = len(reference_elements)

        if total_candidate_count == 0 and total_reference_count == 0:
            return GeometrySimilarityResult(
                overall_similarity=1.0,
                type_similarities={},
                element_counts={},
                total_elements=(0, 0)
            )

        if total_candidate_count == 0 or total_reference_count == 0:
            return GeometrySimilarityResult(
                overall_similarity=0.0,
                type_similarities={},
                element_counts={},
                total_elements=(total_candidate_count, total_reference_count)
            )

        # Group by shape type.
        candidate_groups = self._group_elements_by_type(candidate_elements)
        reference_groups = self._group_elements_by_type(reference_elements)

        # Canvas size (paper: diagonal of the page-level bounding box).
        canvas_size = self._calculate_canvas_size(candidate_elements + reference_elements)

        # Accumulate per-pair scores across types (Algorithm 2 lines 22-27).
        type_similarities: Dict[str, float] = {}
        element_counts: Dict[str, Tuple[int, int]] = {}
        global_sum_exp = 0.0
        global_matches = 0

        all_types = set(candidate_groups.keys()) | set(reference_groups.keys())
        for shape_type in all_types:
            candidate_shapes = candidate_groups.get(shape_type, [])
            reference_shapes = reference_groups.get(shape_type, [])
            element_counts[shape_type] = (len(candidate_shapes), len(reference_shapes))

            sum_exp_t, num_matches_t = self._calculate_type_specific_similarity_with_matches(
                candidate_shapes, reference_shapes, shape_type, canvas_size
            )

            global_sum_exp += sum_exp_t
            global_matches += num_matches_t

            # For diagnostic logging: average exp(-k*cost) per matched pair.
            type_similarities[shape_type] = (
                sum_exp_t / num_matches_t if num_matches_t > 0 else 0.0
            )

        # Final reward (Algorithm 2 lines 28-32 + paper eq. 4).
        denom = max(total_candidate_count, total_reference_count)
        overall_similarity = global_sum_exp / denom if denom > 0 else 1.0

        return GeometrySimilarityResult(
            overall_similarity=overall_similarity,
            type_similarities=type_similarities,
            element_counts=element_counts,
            total_elements=(total_candidate_count, total_reference_count)
        )
    
    def _group_elements_by_type(self, elements: List[GeometricElement]) -> Dict[str, List[GeometricElement]]:
        """Group elements by shape type."""
        groups = {}
        for elem in elements:
            shape_type = elem.element_type
            if shape_type not in groups:
                groups[shape_type] = []
            groups[shape_type].append(elem)
        return groups
    
    def _calculate_canvas_size(self, all_elements: List[GeometricElement]) -> float:
        """Diagonal of the bounding box enclosing all elements.
        """
        if not all_elements:
            return 1.0

        all_bboxes = [elem.bbox for elem in all_elements]
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)

        w = max_x - min_x
        h = max_y - min_y
        return max(sqrt(w * w + h * h), 1.0)
    
    def _solve_and_compute_similarity(self, cost_matrix: np.ndarray, n_cand: int, n_ref: int, k: float = 1.5) -> Tuple[float, int]:
        """Solve the optimal one-to-one matching with the Hungarian algorithm
        and return its contribution to the geometry reward.

        Returns
        -------
        sum_exp : float
            Sum of ``exp(-k * cost)`` over all matched pairs for this type.
        num_matches : int
            Number of pairs returned by the Hungarian solver.
        """
        if cost_matrix.size == 0 or n_cand == 0 or n_ref == 0:
            return 0.0, 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_costs = cost_matrix[row_ind, col_ind]
        sum_exp = float(np.exp(-k * matched_costs).sum())
        return sum_exp, int(len(row_ind))
    
    def _calculate_type_specific_similarity_with_matches(self,
                                          candidate_shapes: List[GeometricElement],
                                          reference_shapes: List[GeometricElement],
                                          shape_type: str,
                                          canvas_size: float) -> Tuple[float, int]:
        """Compute the (sum_exp, num_matches) contribution for one shape type.

        Returns ``(0.0, 0)`` when either side has no element of this type
        (no matches are possible); the missing elements are still
        penalised by the global normaliser ``max(|E_p|, |E_g|)``.
        """
        if not candidate_shapes or not reference_shapes:
            return 0.0, 0

        # Dispatch to per-type similarity functions
        if shape_type == 'circle':
            return self._calculate_circle_similarity(candidate_shapes, reference_shapes, canvas_size)
        elif shape_type == 'line':
            return self._calculate_line_similarity(candidate_shapes, reference_shapes, canvas_size)
        elif shape_type == 'rectangle':
            return self._calculate_rectangle_similarity(candidate_shapes, reference_shapes, canvas_size)
        elif shape_type in ['curve', 'closed_curve']:
            return self._calculate_curve_similarity(candidate_shapes, reference_shapes, canvas_size)
        elif shape_type == 'polygon':
            return self._calculate_polygon_similarity(candidate_shapes, reference_shapes, canvas_size)
        else:
            return self._calculate_generic_similarity(candidate_shapes, reference_shapes, canvas_size)
    
    def _calculate_circle_similarity(self, candidate_circles: List[GeometricElement], 
                                   reference_circles: List[GeometricElement], 
                                   canvas_size: float) -> Tuple[float, int]:
        """Circle similarity: position + radius."""
        n_cand = len(candidate_circles)
        n_ref = len(reference_circles)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_circle in enumerate(candidate_circles):
            for j, ref_circle in enumerate(reference_circles):
                cand_x, cand_y, cand_r = cand_circle.coordinates
                ref_x, ref_y, ref_r = ref_circle.coordinates
                
                pos_dist = sqrt((cand_x - ref_x)**2 + (cand_y - ref_y)**2) / canvas_size
                radius_diff = abs(cand_r - ref_r) / max(cand_r, ref_r, 1e-6)
                
                cost = 0.6 * pos_dist + 0.4 * radius_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=2.0)
    
    def _calculate_line_similarity(self, candidate_lines: List[GeometricElement],
                                 reference_lines: List[GeometricElement],
                                 canvas_size: float) -> Tuple[float, int]:
        """Line similarity: position, length, orientation."""
        n_cand = len(candidate_lines)
        n_ref = len(reference_lines)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_line in enumerate(candidate_lines):
            for j, ref_line in enumerate(reference_lines):
                cand_center = cand_line.centroid
                ref_center = ref_line.centroid
                pos_dist = sqrt((cand_center[0] - ref_center[0])**2 + 
                               (cand_center[1] - ref_center[1])**2) / canvas_size
                
                cand_length = cand_line.area
                ref_length = ref_line.area
                length_diff = abs(cand_length - ref_length) / max(cand_length, ref_length, 1e-6)
                
                orient_diff = abs(cand_line.orientation - ref_line.orientation)
                orient_diff = min(orient_diff, pi - orient_diff) / (pi / 2)
                
                cost = 0.4 * pos_dist + 0.3 * length_diff + 0.3 * orient_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=1.5)
    
    def _calculate_rectangle_similarity(self, candidate_rects: List[GeometricElement],
                                      reference_rects: List[GeometricElement], 
                                      canvas_size: float) -> Tuple[float, int]:
        """Rectangle similarity: position, aspect ratio, area."""
        n_cand = len(candidate_rects)
        n_ref = len(reference_rects)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_rect in enumerate(candidate_rects):
            for j, ref_rect in enumerate(reference_rects):
                cand_x, cand_y, cand_w, cand_h = cand_rect.coordinates
                ref_x, ref_y, ref_w, ref_h = ref_rect.coordinates
                
                cand_center = (cand_x + cand_w/2, cand_y + cand_h/2)
                ref_center = (ref_x + ref_w/2, ref_y + ref_h/2)
                pos_dist = sqrt((cand_center[0] - ref_center[0])**2 + 
                               (cand_center[1] - ref_center[1])**2) / canvas_size
                
                cand_area = abs(cand_w * cand_h)
                ref_area = abs(ref_w * ref_h)
                area_diff = abs(cand_area - ref_area) / max(cand_area, ref_area, 1e-6)
                
                cand_aspect = abs(cand_w) / max(abs(cand_h), 1e-6)
                ref_aspect = abs(ref_w) / max(abs(ref_h), 1e-6)
                aspect_diff = abs(cand_aspect - ref_aspect) / max(cand_aspect, ref_aspect, 1e-6)
                
                cost = 0.4 * pos_dist + 0.3 * area_diff + 0.3 * aspect_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=1.5)
    
    def _calculate_curve_similarity(self, candidate_curves: List[GeometricElement],
                                  reference_curves: List[GeometricElement],
                                  canvas_size: float) -> Tuple[float, int]:
        """Curve similarity: position, length, shape complexity."""
        n_cand = len(candidate_curves)
        n_ref = len(reference_curves)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_curve in enumerate(candidate_curves):
            for j, ref_curve in enumerate(reference_curves):
                pos_dist = sqrt((cand_curve.centroid[0] - ref_curve.centroid[0])**2 + 
                               (cand_curve.centroid[1] - ref_curve.centroid[1])**2) / canvas_size
                
                length_diff = abs(cand_curve.area - ref_curve.area) / max(cand_curve.area, ref_curve.area, 1e-6)
                
                cand_bbox_size = (cand_curve.bbox[2] - cand_curve.bbox[0]) * (cand_curve.bbox[3] - cand_curve.bbox[1])
                ref_bbox_size = (ref_curve.bbox[2] - ref_curve.bbox[0]) * (ref_curve.bbox[3] - ref_curve.bbox[1])
                size_diff = abs(cand_bbox_size - ref_bbox_size) / max(cand_bbox_size, ref_bbox_size, 1e-6)
                
                cost = 0.4 * pos_dist + 0.3 * length_diff + 0.3 * size_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=1.5)
    
    def _calculate_polygon_similarity(self, candidate_polygons: List[GeometricElement],
                                    reference_polygons: List[GeometricElement],
                                    canvas_size: float) -> Tuple[float, int]:
        """Polygon similarity: position, area, vertex count."""
        n_cand = len(candidate_polygons)
        n_ref = len(reference_polygons)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_poly in enumerate(candidate_polygons):
            for j, ref_poly in enumerate(reference_polygons):
                pos_dist = sqrt((cand_poly.centroid[0] - ref_poly.centroid[0])**2 + 
                               (cand_poly.centroid[1] - ref_poly.centroid[1])**2) / canvas_size
                
                area_diff = abs(cand_poly.area - ref_poly.area) / max(cand_poly.area, ref_poly.area, 1e-6)
                
                cand_vertices = len(cand_poly.coordinates) // 2
                ref_vertices = len(ref_poly.coordinates) // 2
                vertex_diff = abs(cand_vertices - ref_vertices) / max(cand_vertices, ref_vertices, 1)
                
                cost = 0.4 * pos_dist + 0.3 * area_diff + 0.3 * vertex_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=1.5)
    
    def _calculate_generic_similarity(self, candidate_shapes: List[GeometricElement],
                                    reference_shapes: List[GeometricElement],
                                    canvas_size: float) -> Tuple[float, int]:
        """Generic shape similarity: position + size."""
        n_cand = len(candidate_shapes)
        n_ref = len(reference_shapes)
        
        cost_matrix = np.zeros((n_cand, n_ref))
        
        for i, cand_shape in enumerate(candidate_shapes):
            for j, ref_shape in enumerate(reference_shapes):
                pos_dist = sqrt((cand_shape.centroid[0] - ref_shape.centroid[0])**2 + 
                               (cand_shape.centroid[1] - ref_shape.centroid[1])**2) / canvas_size
                
                size_diff = abs(cand_shape.area - ref_shape.area) / max(cand_shape.area, ref_shape.area, 1e-6)
                
                cost = 0.6 * pos_dist + 0.4 * size_diff
                cost_matrix[i, j] = cost
        
        return self._solve_and_compute_similarity(cost_matrix, n_cand, n_ref, k=1.5)


def process_single_pair(args):
    """Compute geometry similarity for a single PDF pair."""
    try:
        pair_data, index = args
        
        # Decode the PDF bytes
        rendered_pdf_base64 = pair_data["rendered_pdf_base64"]
        ground_truth_pdf_base64 = pair_data["ground_truth_pdf_base64"]
        
        rendered_pdf_bytes = base64.b64decode(rendered_pdf_base64)
        ground_truth_pdf_bytes = base64.b64decode(ground_truth_pdf_base64)
        
        # Compute geometry similarity
        calculator = SimplifiedGeometrySimilarity()
        similarity_score = calculator.calculate_similarity_from_pdf_bytes(
            rendered_pdf_bytes, ground_truth_pdf_bytes
        )
        
        return {
            "index": index,
            "geometry_similarity": similarity_score,
            "success": True
        }
        
    except Exception as e:
        print(f"Error processing geometry similarity for pair {index}: {e}")
        return {
            "index": index,
            "geometry_similarity": 0.0,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Batch geometry similarity calculation worker')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--num_processes', type=int, default=48, help='Number of processes for parallel computation')
    
    args = parser.parse_args()
    
    try:
        # Read input data
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
        
        pdf_pairs = input_data["image_pairs"]  # field name kept consistent across workers
        
        print(f"Processing {len(pdf_pairs)} PDF pairs for geometry similarity...")
        
        # Prepare per-task arguments for the pool
        process_args = [(pair, i) for i, pair in enumerate(pdf_pairs)]
        
        # Parallel processing
        results = []
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            future_to_index = {
                executor.submit(process_single_pair, arg): i 
                for i, arg in enumerate(process_args)
            }
            
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"Error in geometry similarity calculation for index {idx}: {e}")
                    results.append({
                        "index": idx,
                        "geometry_similarity": 0.0,
                        "success": False,
                        "error": str(e)
                    })
        
        # Sort results by index
        results.sort(key=lambda x: x["index"])
        
        # Write the results
        output_data = {
            "success": True,
            "results": results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Geometry similarity calculation completed. Results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error in geometry similarity batch worker: {e}")
        error_output = {
            "success": False,
            "error": str(e),
            "results": []
        }
        
        try:
            with open(args.output_file, 'w') as f:
                json.dump(error_output, f, indent=2)
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
