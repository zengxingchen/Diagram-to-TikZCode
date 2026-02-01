#!/usr/bin/env python3
"""
PDF text extraction worker script using PyMuPDF for text detection and layout analysis.
This script receives batch PDF data via input file and outputs text detection results via output file.
Uses Hungarian algorithm for optimal text box matching and Distance IoU (DIoU) calculation for layout evaluation.
"""

import sys
import json
import base64
import numpy as np
import fitz  # PyMuPDF
from scipy.optimize import linear_sum_assignment
import argparse
import multiprocessing as mp


def decode_base64_to_bytes(base64_str):
    """Decode base64 string to bytes (for PDF or image data)"""
    return base64.b64decode(base64_str.encode('utf-8'))


def levenshtein_distance(s1, s2):
    """
    Calculate the edit distance (Levenshtein distance) between two strings
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        int: Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def text_similarity_with_edit_distance(text1, text2, adaptive_threshold=True, max_edit_distance=1):
    """
    Determine if two texts are similar based on edit distance
    
    Args:
        text1: First text
        text2: Second text
        adaptive_threshold: Whether to use adaptive threshold, default True
        max_edit_distance: Fixed maximum edit distance (used when adaptive_threshold=False)
        
    Returns:
        tuple: (is_similar, edit_distance, similarity_type)
            - is_similar: Whether similar
            - edit_distance: Edit distance
            - similarity_type: Similarity type ('exact', 'edit_distance', 'none')
    """
    # Preprocessing: remove whitespace and convert to lowercase
    text1_clean = text1.strip().lower()
    text2_clean = text2.strip().lower()
    
    # Skip empty text
    if not text1_clean or not text2_clean:
        return False, -1, 'none'
    
    # Exact match (highest priority)
    if text1_clean == text2_clean:
        return True, 0, 'exact'
    
    # Edit distance matching
    edit_dist = levenshtein_distance(text1_clean, text2_clean)
    
    # Calculate adaptive threshold
    if adaptive_threshold:
        min_length = min(len(text1_clean), len(text2_clean))
        max_length = max(len(text1_clean), len(text2_clean))
        avg_length = (len(text1_clean) + len(text2_clean)) / 2
        
        # Adaptive threshold strategy:
        # 0. Both single character: must match exactly
        # 1. Short text (<=3 chars): allow at most 1 error
        # 2. Medium text (4-10 chars): allow 20% error rate
        # 3. Long text (>10 chars): allow 15% error rate, but at least 2, at most 5
        if max_length == 1 and min_length == 1:
            max_allowed_distance = 0  # Two single characters must match exactly
        elif min_length <= 3:
            max_allowed_distance = 1
        elif avg_length <= 10:
            max_allowed_distance = max(1, int(avg_length * 0.2))
        else:
            max_allowed_distance = max(2, min(5, int(avg_length * 0.15)))
        
        # Additional constraint: edit distance cannot exceed 50% of shorter text length
        # But keep max_allowed_distance = 0 for single character case
        if not (max_length == 1 and min_length == 1):
            max_allowed_distance = min(max_allowed_distance, max(1, min_length // 2))
        
        # Additional constraint: text length difference cannot be too large (prevent short-long text mismatch)
        length_diff = abs(len(text1_clean) - len(text2_clean))
        if length_diff > max_allowed_distance:
            return False, edit_dist, 'none'
            
    else:
        # Use fixed threshold (backward compatibility)
        min_length = min(len(text1_clean), len(text2_clean))
        max_allowed_distance = min(max_edit_distance, max(1, min_length // 2))
    
    if edit_dist <= max_allowed_distance:
        return True, edit_dist, 'edit_distance'
    
    return False, edit_dist, 'none'


def bbox_diou_rect(box1_rect, box2_rect):
    """
    Calculate Distance IoU (DIoU) between two rectangular bounding boxes.
    Input format: [x_min, y_min, x_max, y_max]
    """
    try:
        x1_min, y1_min, x1_max, y1_max = box1_rect
        x2_min, y2_min, x2_max, y2_max = box2_rect
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            inter_area = 0.0
        else:
            inter_width = np.clip(inter_x_max - inter_x_min, 0, 1e6)
            inter_height = np.clip(inter_y_max - inter_y_min, 0, 1e6)
            inter_area = inter_width * inter_height
        
        # Calculate union and IoU with overflow protection
        width1 = np.clip(x1_max - x1_min, 0, 1e6)
        height1 = np.clip(y1_max - y1_min, 0, 1e6)
        width2 = np.clip(x2_max - x2_min, 0, 1e6)
        height2 = np.clip(y2_max - y2_min, 0, 1e6)
        
        area1 = width1 * height1
        area2 = width2 * height2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        iou = inter_area / union_area
        
        # Calculate center points
        center1_x = (x1_min + x1_max) / 2
        center1_y = (y1_min + y1_max) / 2
        center2_x = (x2_min + x2_max) / 2
        center2_y = (y2_min + y2_max) / 2
        
        # Calculate distance between centers (ρ²) with clipping to prevent overflow
        center_dx = np.clip(center1_x - center2_x, -1e6, 1e6)
        center_dy = np.clip(center1_y - center2_y, -1e6, 1e6)
        center_distance_sq = center_dx ** 2 + center_dy ** 2
        
        # Calculate diagonal of smallest enclosing box (c²)
        enclose_x_min = min(x1_min, x2_min)
        enclose_y_min = min(y1_min, y2_min)
        enclose_x_max = max(x1_max, x2_max)
        enclose_y_max = max(y1_max, y2_max)
        
        # Use np.float64 to prevent overflow and clip extreme values
        enclose_width = np.clip(enclose_x_max - enclose_x_min, 0, 1e6)
        enclose_height = np.clip(enclose_y_max - enclose_y_min, 0, 1e6)
        enclose_diagonal_sq = enclose_width ** 2 + enclose_height ** 2
        
        if enclose_diagonal_sq <= 0:
            return iou
        
        # Calculate DIoU with numerical stability
        distance_ratio = center_distance_sq / enclose_diagonal_sq
        distance_ratio = np.clip(distance_ratio, 0, 2.0)  # Prevent extreme penalties
        diou = iou - distance_ratio
        
        # DIoU should be in range [-1, 1], preserve full range for distance information
        return float(np.clip(diou, -1.0, 1.0))
    except Exception:
        return 0.0


def stage1a_exact_text_matching(pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices):
    """
    Stage 1A: Exact text matching (strict matching)
    """
    exact_matches = []
    updated_pred_indices = remaining_pred_indices.copy()
    updated_gt_indices = remaining_gt_indices.copy()
    
    # Build exact match candidate dictionary for each GT text
    gt_to_pred_exact_candidates = {}
    
    for gt_idx in list(updated_gt_indices):
        gt_text = gt_texts[gt_idx].strip().lower()
        
        if not gt_text:
            continue
            
        exact_candidates = []
        
        for pred_idx in list(updated_pred_indices):
            pred_text = pred_texts[pred_idx].strip().lower()
            
            if pred_text and pred_text == gt_text:
                exact_candidates.append(pred_idx)
        
        if exact_candidates:
            gt_to_pred_exact_candidates[gt_idx] = exact_candidates
    
    sorted_gt_candidates = sorted(gt_to_pred_exact_candidates.items(), 
                                key=lambda x: len(x[1]))
    
    for gt_idx, candidates in sorted_gt_candidates:
        available_candidates = [c for c in candidates if c in updated_pred_indices]
        
        if len(available_candidates) == 1:
            pred_idx = available_candidates[0]
            
            diou = bbox_diou_rect(pred_boxes[pred_idx], gt_boxes[gt_idx])
            transformed_diou = (diou + 1.0) / 2.0
            
            match = {
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'diou': diou,
                'transformed_diou': transformed_diou,
                'pred_box': pred_boxes[pred_idx],
                'gt_box': gt_boxes[gt_idx],
                'pred_text': pred_texts[pred_idx],
                'gt_text': gt_texts[gt_idx],
                'match_type': 'text_exact',
                'edit_distance': 0,
                'similarity_type': 'exact'
            }
            
            exact_matches.append(match)
            updated_pred_indices.remove(pred_idx)
            updated_gt_indices.remove(gt_idx)
        elif len(available_candidates) > 1:
            # When multiple exact candidates exist, use DIoU to select the geometrically closest one
            candidates_with_diou = []
            for pred_idx in available_candidates:
                diou = bbox_diou_rect(pred_boxes[pred_idx], gt_boxes[gt_idx])
                candidates_with_diou.append((pred_idx, diou))
            
            # Sort by DIoU in descending order, select the geometrically closest one
            candidates_with_diou.sort(key=lambda x: x[1], reverse=True)
            pred_idx, diou = candidates_with_diou[0]
            
            transformed_diou = (diou + 1.0) / 2.0
            
            match = {
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'diou': diou,
                'transformed_diou': transformed_diou,
                'pred_box': pred_boxes[pred_idx],
                'gt_box': gt_boxes[gt_idx],
                'pred_text': pred_texts[pred_idx],
                'gt_text': gt_texts[gt_idx],
                'match_type': 'text_exact',
                'edit_distance': 0,
                'similarity_type': 'exact'
            }
            
            exact_matches.append(match)
            updated_pred_indices.remove(pred_idx)
            updated_gt_indices.remove(gt_idx)
    
    return exact_matches, updated_pred_indices, updated_gt_indices


def stage1b_edit_distance_matching(pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices):
    """
    Stage 1B: Edit distance text matching
    """
    edit_distance_matches = []
    updated_pred_indices = remaining_pred_indices.copy()
    updated_gt_indices = remaining_gt_indices.copy()
    
    gt_to_pred_candidates = {}
    
    for gt_idx in list(updated_gt_indices):
        gt_text = gt_texts[gt_idx].strip()
        
        if not gt_text:
            continue
            
        candidates = []
        
        for pred_idx in list(updated_pred_indices):
            pred_text = pred_texts[pred_idx].strip()
            
            is_similar, edit_dist, similarity_type = text_similarity_with_edit_distance(
                pred_text, gt_text, adaptive_threshold=True
            )
            
            if is_similar:
                candidates.append((pred_idx, edit_dist, similarity_type))
        
        if candidates:
            gt_to_pred_candidates[gt_idx] = candidates
    
    sorted_gt_candidates = sorted(gt_to_pred_candidates.items(), 
                                key=lambda x: len(x[1]))
    
    for gt_idx, candidates in sorted_gt_candidates:
        available_candidates = [(pred_idx, edit_dist, sim_type) for pred_idx, edit_dist, sim_type in candidates 
                               if pred_idx in updated_pred_indices]
        
        if len(available_candidates) == 1:
            pred_idx, edit_dist, similarity_type = available_candidates[0]
            
            diou = bbox_diou_rect(pred_boxes[pred_idx], gt_boxes[gt_idx])
            transformed_diou = (diou + 1.0) / 2.0
            
            match_type = 'text_exact' if similarity_type == 'exact' else 'text_edit_distance'
            
            match = {
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'diou': diou,
                'transformed_diou': transformed_diou,
                'pred_box': pred_boxes[pred_idx],
                'gt_box': gt_boxes[gt_idx],
                'pred_text': pred_texts[pred_idx],
                'gt_text': gt_texts[gt_idx],
                'match_type': match_type,
                'edit_distance': edit_dist,
                'similarity_type': similarity_type
            }
            
            edit_distance_matches.append(match)
            updated_pred_indices.remove(pred_idx)
            updated_gt_indices.remove(gt_idx)
        
        elif len(available_candidates) > 1:
            available_candidates.sort(key=lambda x: (0 if x[2] == 'exact' else 1, x[1]))
            
            best_edit_dist = available_candidates[0][1]
            best_candidates = [c for c in available_candidates if c[1] == best_edit_dist and c[2] == available_candidates[0][2]]
            
            if len(best_candidates) == 1:
                pred_idx, edit_dist, similarity_type = best_candidates[0]
                
                diou = bbox_diou_rect(pred_boxes[pred_idx], gt_boxes[gt_idx])
                transformed_diou = (diou + 1.0) / 2.0
                
                match_type = 'text_exact' if similarity_type == 'exact' else 'text_edit_distance'
                
                match = {
                    'pred_idx': pred_idx,
                    'gt_idx': gt_idx,
                    'diou': diou,
                    'transformed_diou': transformed_diou,
                    'pred_box': pred_boxes[pred_idx],
                    'gt_box': gt_boxes[gt_idx],
                    'pred_text': pred_texts[pred_idx],
                    'gt_text': gt_texts[gt_idx],
                    'match_type': match_type,
                    'edit_distance': edit_dist,
                    'similarity_type': similarity_type
                }
                
                edit_distance_matches.append(match)
                updated_pred_indices.remove(pred_idx)
                updated_gt_indices.remove(gt_idx)
    
    return edit_distance_matches, updated_pred_indices, updated_gt_indices


def stage1_text_based_matching(pred_boxes, pred_texts, gt_boxes, gt_texts):
    """
    Stage 1: Hierarchical matching based on text content (exact matching first, then edit distance matching)
    """
    remaining_pred_indices = set(range(len(pred_boxes)))
    remaining_gt_indices = set(range(len(gt_boxes)))
    
    exact_matches, remaining_pred_indices, remaining_gt_indices = stage1a_exact_text_matching(
        pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices
    )
    
    edit_distance_matches, remaining_pred_indices, remaining_gt_indices = stage1b_edit_distance_matching(
        pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices
    )
    
    text_matches = exact_matches + edit_distance_matches
    
    return text_matches, remaining_pred_indices, remaining_gt_indices


def stage2_geometric_matching(pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices):
    """
    Stage 2: Hungarian algorithm matching based on geometric position
    """
    geometric_matches = []
    
    remaining_pred_list = list(remaining_pred_indices)
    remaining_gt_list = list(remaining_gt_indices)
    
    if not remaining_pred_list or not remaining_gt_list:
        return geometric_matches
    
    cost_matrix = np.zeros((len(remaining_pred_list), len(remaining_gt_list)))
    
    for i, pred_idx in enumerate(remaining_pred_list):
        for j, gt_idx in enumerate(remaining_gt_list):
            try:
                diou = bbox_diou_rect(pred_boxes[pred_idx], gt_boxes[gt_idx])
                cost_matrix[i][j] = -diou  # Negative for minimization
            except Exception as e:
                cost_matrix[i][j] = 0.0
    
    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_indices, col_indices):
            pred_idx = remaining_pred_list[i]
            gt_idx = remaining_gt_list[j]
            diou = -cost_matrix[i][j]  # Convert back to positive
            
            if diou > -0.5:
                transformed_diou = (diou + 1.0) / 2.0
                
                match = {
                    'pred_idx': pred_idx,
                    'gt_idx': gt_idx,
                    'diou': diou,
                    'transformed_diou': transformed_diou,
                    'pred_box': pred_boxes[pred_idx],
                    'gt_box': gt_boxes[gt_idx],
                    'pred_text': pred_texts[pred_idx] if pred_idx < len(pred_texts) else '',
                    'gt_text': gt_texts[gt_idx] if gt_idx < len(gt_texts) else '',
                    'match_type': 'geometric'
                }
                
                geometric_matches.append(match)
        
    except Exception as e:
        print(f"Error in Hungarian algorithm: {e}")
    
    return geometric_matches


def calculate_layout_similarity_two_stage(pred_boxes, pred_texts, gt_boxes, gt_texts, skip_stage2=False):
    """Two-stage matching algorithm: text matching first, then geometric matching
    
    Args:
        pred_boxes: Predicted bounding boxes
        pred_texts: Predicted texts
        gt_boxes: Ground truth bounding boxes
        gt_texts: Ground truth texts
        skip_stage2: Whether to skip stage2 geometric matching, default False
    """
    # Handle edge cases: correct empty set should get full score
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            # GT has no text, model also generated no text -> completely correct, give full score
            return 1.0, []
        else:
            # GT has no text, but model generated text -> hallucination, give zero score
            return 0.0, []
    
    # GT has text boxes, but pred has no text boxes -> missed detection, give zero score
    if len(pred_boxes) == 0:
        return 0.0, []
    
    text_matches, remaining_pred_indices, remaining_gt_indices = stage1_text_based_matching(
        pred_boxes, pred_texts, gt_boxes, gt_texts
    )
    
    if skip_stage2:
        # Use only text matching results
        all_matches = text_matches
    else:
        # Normal execution of two-stage matching
        geometric_matches = stage2_geometric_matching(
            pred_boxes, pred_texts, gt_boxes, gt_texts, remaining_pred_indices, remaining_gt_indices
        )
        all_matches = text_matches + geometric_matches
    
    if not all_matches:
        return 0.0, []
    
    total_transformed_diou = sum(match['transformed_diou'] for match in all_matches)
    max_possible_matches = max(len(pred_boxes), len(gt_boxes))
    final_score = total_transformed_diou / max_possible_matches
    
    return final_score, all_matches


def extract_text_and_bbox_from_pdf_bytes(pdf_bytes, target_size=384):
    """
    Extract text and bounding box information from PDF byte data, and normalize coordinates to fixed size
    
    Args:
        pdf_bytes: PDF file byte data
        target_size: Target normalization size, default 384 (consistent with image processing)
        
    Returns:
        tuple: (boxes, texts) where:
        - boxes: list of normalized boxes in format [x_min, y_min, x_max, y_max] (0-target_size range)
        - texts: list of corresponding text strings
    """
    boxes = []
    texts = []
    
    try:
        # Create PyMuPDF document from byte data
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Due to task limitations, we only process the first page (and only one page)
        if len(doc) == 0:
            print("Warning: PDF has no pages")
            doc.close()
            return [], []
            
        page = doc[0]  # Only take the first page
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Calculate scale factor to normalize to target_size x target_size
        # Use the same logic as image processing: scale by longest side, then center padding
        max_dim = max(page_width, page_height)
        scale_factor = target_size / max_dim
        
        # Calculate scaled dimensions
        scaled_width = page_width * scale_factor
        scaled_height = page_height * scale_factor
        
        # Calculate center offset (simulate ImageOps.pad padding operation)
        offset_x = (target_size - scaled_width) / 2
        offset_y = (target_size - scaled_height) / 2
        
        
        # Get text dictionary (contains detailed line and span information)
        text_dict = page.get_text("dict")
        
        # Iterate through all text blocks
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                # Iterate through lines in each text block
                for line in block["lines"]:
                    # Get line bounding box
                    line_bbox = line["bbox"]  # [x0, y0, x1, y1]
                    x0, y0, x1, y1 = line_bbox
                    
                    # Collect all text fragments in the line
                    line_text_parts = []
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if span_text:
                            line_text_parts.append(span_text)
                    
                    # Merge line text
                    line_text = " ".join(line_text_parts).strip()
                    
                    if line_text:  # Only process non-empty lines
                        # Normalize coordinates: scale first, then add center offset
                        norm_x0 = x0 * scale_factor + offset_x
                        norm_y0 = y0 * scale_factor + offset_y
                        norm_x1 = x1 * scale_factor + offset_x
                        norm_y1 = y1 * scale_factor + offset_y
                        
                        # Ensure coordinates are within valid range
                        norm_x0 = max(0, min(target_size, norm_x0))
                        norm_y0 = max(0, min(target_size, norm_y0))
                        norm_x1 = max(0, min(target_size, norm_x1))
                        norm_y1 = max(0, min(target_size, norm_y1))
                        
                        # Only add valid bounding boxes (avoid degenerate boxes)
                        if norm_x1 > norm_x0 and norm_y1 > norm_y0:
                            boxes.append([float(norm_x0), float(norm_y0), float(norm_x1), float(norm_y1)])
                            texts.append(line_text)
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return [], []
    
    return boxes, texts


def process_pdf_pair(pair_data, skip_stage2=False):
    """
    Process a single PDF pair for text extraction and layout analysis.
    
    Args:
        pair_data: PDF pair data containing base64 encoded PDF and ground truth data
        skip_stage2: Whether to skip stage2 geometric matching, default False
    """
    try:
        # Decode PDF data for both predicted and ground truth
        rendered_pdf_bytes = decode_base64_to_bytes(pair_data["rendered_pdf_base64"])
        ground_truth_pdf_bytes = decode_base64_to_bytes(pair_data["ground_truth_pdf_base64"])
        
        # Extract text boxes and texts from rendered PDF and ground truth PDF
        pred_boxes, pred_texts = extract_text_and_bbox_from_pdf_bytes(rendered_pdf_bytes)
        gt_boxes, gt_texts = extract_text_and_bbox_from_pdf_bytes(ground_truth_pdf_bytes)
        
        # Use two-stage layout similarity calculation
        layout_score, _ = calculate_layout_similarity_two_stage(
            pred_boxes, pred_texts, gt_boxes, gt_texts, skip_stage2=skip_stage2
        )
                
        return {
            "layout_score": float(layout_score),
            "success": True
        }
        
    except Exception as e:
        print(f"Error processing PDF pair: {e}")
        return {
            "layout_score": 0.0,
            "success": False,
            "error": str(e)
        }


def process_cpu_worker(pdf_pairs_chunk, process_id, total_processes, skip_stage2=False):
    """
    Process a subset of PDF pairs on CPU
    
    Args:
        pdf_pairs_chunk: List of PDF pairs to process
        process_id: Process identifier
        total_processes: Total number of processes
        skip_stage2: Whether to skip stage2 geometric matching, default False
    """
    
    try:
        print(f"Process {process_id}/{total_processes} starting with {len(pdf_pairs_chunk)} PDF pairs")
        
        results = []
        
        # Process PDF pairs with periodic progress updates
        total_pairs = len(pdf_pairs_chunk)
        progress_interval = max(1, total_pairs // 3)  # Report every 33% or at least every item
        
        for i, pair_data in enumerate(pdf_pairs_chunk):
            try:
                # Print progress at intervals
                if i % progress_interval == 0 or i == total_pairs - 1:
                    progress_pct = (i + 1) / total_pairs * 100
                    print(f"Process {process_id}: Processing progress {i+1}/{total_pairs} ({progress_pct:.1f}%)")
                
                result = process_pdf_pair(pair_data, skip_stage2=skip_stage2)
                results.append(result)
                    
            except Exception as e:
                results.append({
                    "layout_score": 0.0,
                    "success": False
                })
        
        # Statistics for current process
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        print(f"Process {process_id} completed: {successful} success, {failed} failed")
        
        return results
        
    except Exception as e:
        print(f"Error: Process {process_id} initialization failed: {e}")
        return [{
            "layout_score": 0.0,
            "success": False,
        } for _ in pdf_pairs_chunk]


def split_data_for_processes(pdf_pairs, num_processes):
    """
    Split PDF pair data into multiple chunks for different processes
    """
    if num_processes <= 0:
        return [pdf_pairs]
    
    chunk_size = len(pdf_pairs) // num_processes
    remainder = len(pdf_pairs) % num_processes
    
    chunks = []
    start_idx = 0
    
    for i in range(num_processes):
        # First few processes get one extra data point to handle remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunks.append(pdf_pairs[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PDF text extraction worker for text layout analysis')
    parser.add_argument('input_file', help='Path to input JSON file containing image pairs')
    parser.add_argument('output_file', help='Path to output JSON file for results')
    parser.add_argument('--num_processes', type=int, default=48, help='Number of processes to use')
    parser.add_argument('--skip_stage2', default=False,
                       help='Skip stage2 geometric matching (use only text matching)')
    args = parser.parse_args()
    
    # Multi-process mode setup
    num_processes = args.num_processes 
    print(f"Using multi-process mode: {num_processes} processes")
    print(f"Skip Stage2 geometric matching: {args.skip_stage2}")
    
    try:
        print(f"Reading input file: {args.input_file}")
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
        
        pdf_pairs = input_data.get("image_pairs", [])  # Keep "image_pairs" key for compatibility
        if not pdf_pairs:
            output_error = {"error": "No PDF pairs provided", "success": False}
            with open(args.output_file, 'w') as f:
                json.dump(output_error, f)
            sys.exit(1)
        
        print(f"Successfully loaded {len(pdf_pairs)} PDF pairs")
        print("Starting initialization process...")
        
    except json.JSONDecodeError as e:
        output_error = {"error": f"Failed to parse input JSON: {e}", "success": False}
        with open(args.output_file, 'w') as f:
            json.dump(output_error, f)
        sys.exit(1)
    except Exception as e:
        output_error = {"error": f"Failed to read input file: {e}", "success": False}
        with open(args.output_file, 'w') as f:
            json.dump(output_error, f)
        sys.exit(1)
    
    try:
        
        # Multi-process parallel processing
        print(f"Starting multi-process parallel processing with {num_processes} processes...")
        
        # Split data
        data_chunks = split_data_for_processes(pdf_pairs, num_processes)
        print(f"Data splitting completed, chunk sizes: {[len(chunk) for chunk in data_chunks]}")
        
        # Create process pool with improved shutdown handling
        pool = mp.Pool(processes=num_processes)
        try:
            # Create task arguments
            tasks = []
            for i, chunk in enumerate(data_chunks):
                if len(chunk) > 0:  # Only process non-empty chunks
                    task_args = (chunk, i+1, num_processes, args.skip_stage2)
                    tasks.append(task_args)
            
            # Execute tasks in parallel
            print(f"Starting {len(tasks)} processes...")
            chunk_results = pool.starmap(process_cpu_worker, tasks)
            
            print("All processes completed, closing process pool...")
            
        finally:
            # Graceful shutdown with timeout
            pool.close()  # No more work
            print("Waiting for worker processes to finish...")
            pool.join()   # Wait for workers to finish
            print("All worker processes finished")
        
        # Merge all results
        print("Merging all processing results...")
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        # Add index to maintain original order
        for i, result in enumerate(results):
            result["index"] = i
        print(f"Results merged successfully, processed {len(results)} PDF pairs")
        
            
        output = {
            "results": results,
            "success": True,
        }
        print("Saving results to output file...")
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print("Processing completed! Results saved.")
        
    except Exception as e:
        output_error = {
            "error": f"Failed to compute PDF text layout analysis: {e}",
            "success": False
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_error, f)
        sys.exit(1)

if __name__ == "__main__":
    main()
