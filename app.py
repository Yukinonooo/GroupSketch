import torch
import pydiffvg
import os
import re
import subprocess
import threading
import tempfile
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, send_file
import xml.etree.ElementTree as ET
from xml.etree import ElementTree as etree

# 获取项目根目录，确保可移植性
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_center_of_mass(shapes):
    """Compute the centroid of all shapes.
    Args:
        shapes: List of all shape objects in the SVG
    Returns:
        center: The average position (centroid) coordinates of all points
    """
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)  # Vertically stack all points into a large matrix
    center = points_vars.mean(dim=0)  # Compute the average position of all points
    return center

def parse_path_commands(d):
    """Parse the 'd' attribute of an SVG path.
    Args:
        d: The 'd' attribute string of the path
    Returns:
        commands: List of parsed commands, each containing type and coordinates
    """
    # Split commands using regular expression
    pattern = r'([MLHVCSQTAZ])([^MLHVCSQTAZ]*)'
    matches = re.findall(pattern, d)
    commands = []
    
    for cmd_type, args in matches:
        # Convert the argument string into a list of floats
        args = [float(x) for x in args.strip().replace(',', ' ').split()]
        commands.append((cmd_type, args))
    
    return commands

def split_complex_paths(shapes):
    """Split complex paths containing multiple Bezier curves into individual paths.
    Args:
        shapes: List of original shapes
    Returns:
        new_shapes: List of split shapes
    """
    new_shapes = []
    
    for shape in shapes:
        if isinstance(shape, pydiffvg.Path):
            # Get the starting point of the path
            current_point = shape.points[0]
            points_per_curve = 4  # A cubic Bezier curve uses 4 points (start + 2 control points + end)
            
            # If number of points is greater than 4, it's a complex path
            if len(shape.points) > points_per_curve:
                # Process in groups of 4 points
                for i in range(0, len(shape.points) - 3, 3):
                    curve_points = torch.stack([
                        shape.points[i],      # start point
                        shape.points[i + 1],  # control point 1
                        shape.points[i + 2],  # control point 2
                        shape.points[i + 3]   # end point
                    ])
                    
                    # Create a new Path object, keeping only necessary attributes
                    new_path = pydiffvg.Path(
                        points=curve_points,
                        stroke_width=shape.stroke_width,
                        is_closed=False,  # A single Bezier segment is not closed
                        num_control_points=torch.tensor([2])  # A cubic Bezier has 2 control points
                    )
                    new_shapes.append(new_path)
            else:
                new_shapes.append(shape)
        else:
            new_shapes.append(shape)
            
    return new_shapes

def resize_canvas(svg_path, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5):
    """Resize the SVG canvas and reposition shapes.
    Args:
        svg_path: Path to the SVG file
        target_width: Target canvas width
        target_height: Target canvas height
        obj_scale: Object scaling factor
        target_stroke_width: Target stroke width
    """
    # Read SVG scene info
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_path)
    
    # Split complex paths
    shapes_init = split_complex_paths(shapes_init)
    
    # Compute center scaling factors
    width_factor_center = canvas_width / target_width
    height_factor_center = canvas_height / target_height
    
    # Compute the overall scaling factor (preserve aspect ratio)
    if canvas_width > canvas_height:
        width_factor = canvas_width / target_width
        height_factor = width_factor
    else:
        height_factor = canvas_height / target_height  # preserve height
        width_factor = height_factor
        
    # Get centroid of the original sketch
    center_of_orig_sketch = get_center_of_mass(shapes_init)
    
    # Compute new canvas center and new shape center positions
    new_canvas_orig = torch.tensor((target_width / 2, target_height / 2))
    new_center = (center_of_orig_sketch[0] / width_factor_center, center_of_orig_sketch[1] / height_factor_center)
    
    # Process each path object
    for j,path in enumerate(shapes_init):
        # Step 1: Move the shape to the origin (0,0)
        path.points[:, 0] -= center_of_orig_sketch[0]
        path.points[:, 1] -= center_of_orig_sketch[1]
        
        # Step 2: Resize according to the scaling factor
        path.points[:, 1] /= height_factor
        path.points[:, 0] /= width_factor
        path.points[:, 1] *= obj_scale
        path.points[:, 0] *= obj_scale
        
        # Step 3: Move to the new center position
        path.points[:, 0] += new_center[0]
        path.points[:, 1] += new_center[1]
        
        # Set the new stroke width
        path.stroke_width = torch.tensor((target_stroke_width))   
    
    # Serialize scene and save new SVG
    scene_args = pydiffvg.RenderFunction.serialize_scene(target_width, target_height, shapes_init, shape_groups_init)
    render = pydiffvg.RenderFunction.apply
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    pydiffvg.save_svg(os.path.join(uploads_dir, "scaled.svg"), target_width, target_height, shapes_init, shape_groups_init)
    
    
def process_file(svg_path, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5):
    """Process a single SVG file."""
    # Read SVG scene info
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_path)
    
    # Split complex paths
    shapes_init = split_complex_paths(shapes_init)
    
    # Compute center scaling factors
    width_factor_center = canvas_width / target_width
    height_factor_center = canvas_height / target_height
    
    # Compute the overall scaling factor (preserve aspect ratio)
    if canvas_width > canvas_height:
        width_factor = canvas_width / target_width
        height_factor = width_factor
    else:
        height_factor = canvas_height / target_height  # preserve height
        width_factor = height_factor
        
    # Get centroid of the original sketch
    center_of_orig_sketch = get_center_of_mass(shapes_init)
    
    # Compute new canvas center and new shape center positions
    new_canvas_orig = torch.tensor((target_width / 2, target_height / 2))
    new_center = (center_of_orig_sketch[0] / width_factor_center, center_of_orig_sketch[1] / height_factor_center)
    
    # Process each path object
    for j,path in enumerate(shapes_init):
        # Step 1: Move the shape to the origin (0,0)
        path.points[:, 0] -= center_of_orig_sketch[0]
        path.points[:, 1] -= center_of_orig_sketch[1]
        
        # Step 2: Resize according to the scaling factor
        path.points[:, 1] /= height_factor
        path.points[:, 0] /= width_factor
        path.points[:, 1] *= obj_scale
        path.points[:, 0] *= obj_scale
        
        # Step 3: Move to the new center position
        path.points[:, 0] += new_center[0]
        path.points[:, 1] += new_center[1]
        
        # Set the new stroke width
        path.stroke_width = torch.tensor((target_stroke_width))   
    
    # Serialize scene and save new SVG
    scene_args = pydiffvg.RenderFunction.serialize_scene(target_width, target_height, shapes_init, shape_groups_init)
    render = pydiffvg.RenderFunction.apply
    
    new_input_dir = os.path.join(PROJECT_ROOT, "new_input")
    os.makedirs(new_input_dir, exist_ok=True)
    output_file = os.path.join(new_input_dir, f"{os.path.splitext(os.path.basename(svg_path))[0]}_scaled.svg")
    pydiffvg.save_svg(output_file, target_width, target_height, shapes_init, shape_groups_init)

def process_folder(folder_path, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5):
    """Process all SVG files in a folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".svg"):
            svg_path = os.path.join(folder_path, filename)
            process_file(svg_path, target_width, target_height, obj_scale, target_stroke_width)

def clean_empty_tags(svg_path):
    """Clean empty tags in the SVG file, such as empty <g></g>.
    
    Args:
        svg_path: Path to the SVG file
        
    Returns:
        cleaned_svg_path: Path to the cleaned SVG file
    """
    # Read SVG file
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    # Remove namespace prefixes
    svg_content = svg_content.replace('ns0:', '')
    svg_content = svg_content.replace('xmlns:ns0=', 'xmlns=')
    svg_content = svg_content.replace('svg:', '')
    svg_content = svg_content.replace('xmlns:svg=', 'xmlns=')
    # Use regex to find and replace empty g tags
    # This regex matches <g> tags (with attributes) followed immediately by </g>
    cleaned_content = re.sub(r'<g[^>]*>\s*</g>', '', svg_content)
    
    # Create a new file name
    base_name = os.path.splitext(os.path.basename(svg_path))[0]
    dir_name = os.path.dirname(svg_path)
    cleaned_svg_path = os.path.join(dir_name, f"{base_name}_cleaned.svg")
    
    # Save the cleaned SVG file
    with open(cleaned_svg_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"cleaned empty tags and saved to: {cleaned_svg_path}")
    return cleaned_svg_path

from flask import Flask, request, jsonify, render_template, send_from_directory, Response, send_file
import xml.etree.ElementTree as ET
from xml.etree import ElementTree as etree
etree.register_namespace("", "http://www.w3.org/2000/svg")

import os
import uuid
import json
from werkzeug.utils import secure_filename
import copy
import zipfile
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(PROJECT_ROOT, 'outputs')
app.config['AFTER_MOVE_FOLDER'] = os.path.join(PROJECT_ROOT, 'afterMove')

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['AFTER_MOVE_FOLDER'], exist_ok=True)

# Dictionaries to store uploaded SVG files, grouping information, and frame positions
svg_files = {}
groups_info = {}
frame_positions = {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_svg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.svg'):
        # Generate a unique ID
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(original_path)
        
        # Call resize_canvas to preprocess the SVG file
        # Note: target_width, target_height, obj_scale, and target_stroke_width can be adjusted as needed
        resize_canvas(original_path, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5)
        
        # The resize_canvas function saves the processed SVG to the 'uploads' directory as 'scaled.svg'
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"scaled.svg")
        
        # Build parse info using the processed SVG file
        tree = ET.parse(processed_path)
        root = tree.getroot()
        
        # Update global SVG file dict to save processed path and related info
        svg_files[file_id] = {
            'filename': filename,
            'path': processed_path,
            'tree': tree,
            'root': root,
            'namespace': get_namespace(root)
        }
        
        # Extract path elements from SVG
        namespace = svg_files[file_id]['namespace']
        paths = []
        for i, path in enumerate(root.findall(f'.//{{{namespace}}}path')):
            path_id = path.get('id', f'path_{i}')
            if not path.get('id'):
                path.set('id', path_id)
            
            paths.append({
                'id': path_id,
                'd': path.get('d', ''),
                'style': path.get('style', ''),
                'fill': path.get('fill', ''),
                'stroke': path.get('stroke', ''),
                'stroke-width': path.get('stroke-width', '')
            })
        
        # Initialize groups and frame positions for this file
        groups_info[file_id] = []
        frame_positions[file_id] = {}
        
        return jsonify({
            'file_id': file_id,
            'paths': paths,
            'svg_width': root.get('width', '800'),
            'svg_height': root.get('height', '600')
        })
    
    return jsonify({'error': 'File must be SVG'}), 400

@app.route('/create_group', methods=['POST'])
def create_group():
    data = request.json
    file_id = data.get('file_id')
    group_name = data.get('group_name')
    path_ids = data.get('path_ids', [])
    
    if not file_id or not group_name or not path_ids:
        return jsonify({'error': 'Missing required fields'}), 400
    
    if file_id not in svg_files:
        return jsonify({'error': 'File not found'}), 404
    
    # Create a new group record
    group_id = f"group_{len(groups_info[file_id])}"
    groups_info[file_id].append({
        'id': group_id,
        'name': group_name,
        'path_ids': path_ids
    })
    
    # Initialize frame positions for this group
    if file_id not in frame_positions:
        frame_positions[file_id] = {}
    frame_positions[file_id][group_id] = {}
    
    return jsonify({
        'group_id': group_id,
        'message': f'Group "{group_name}" created successfully'
    })

@app.route('/get_groups', methods=['GET'])
def get_groups():
    file_id = request.args.get('file_id')
    
    if not file_id:
        return jsonify({'error': 'Missing file_id'}), 400
    
    if file_id not in groups_info:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify({'groups': groups_info[file_id]})

@app.route('/record_position', methods=['POST'])
def record_position():
    data = request.json
    file_id = data.get('file_id')
    group_id = data.get('group_id')
    frame = int(data.get('frame', 1))
    position = data.get('position', {})
    
    if not file_id or not group_id or not position:
        return jsonify({'error': 'Missing required fields'}), 400
    
    if file_id not in frame_positions:
        return jsonify({'error': 'File not found'}), 404
    
    if group_id not in frame_positions[file_id]:
        frame_positions[file_id][group_id] = {}
    
    # Record position
    frame_positions[file_id][group_id][str(frame)] = position
    
    return jsonify({'message': f'Position recorded for frame {frame}'})

def calculate_position(frame, keyframes):
    """Compute the position for the specified frame, ensuring frame 1 is at the origin (0,0)."""
    # Frame 1 is always at the origin (0,0)
    if frame == 1:
        return {'x': 0, 'y': 0}
    
    # If the current frame is a keyframe, return its position directly
    if frame in keyframes:
        return keyframes[frame]
    
    # Find surrounding keyframes
    prev_frames = [f for f in keyframes if f < frame]
    next_frames = [f for f in keyframes if f > frame]
    
    # If only a following keyframe exists (e.g., between frame 1 and the first keyframe)
    if not prev_frames and next_frames:
        next_frame = min(next_frames)
        next_pos = keyframes[next_frame]
        
        # Interpolate from frame 1 (0,0) to the next keyframe
        t = (frame - 1) / (next_frame - 1)
        return {
            'x': t * next_pos['x'],
            'y': t * next_pos['y']
        }
    
    # Interpolation between two keyframes
    if prev_frames and next_frames:
        prev_frame = max(prev_frames)
        next_frame = min(next_frames)
        
        prev_pos = keyframes[prev_frame]
        next_pos = keyframes[next_frame]
        
        t = (frame - prev_frame) / (next_frame - prev_frame)
        return {
            'x': prev_pos['x'] + t * (next_pos['x'] - prev_pos['x']),
            'y': prev_pos['y'] + t * (next_pos['y'] - prev_pos['y'])
        }
    
    # If only previous keyframes exist, use the last keyframe's position
    if prev_frames:
        prev_frame = max(prev_frames)
        return keyframes[prev_frame]
    
    # Default to origin
    return {'x': 0, 'y': 0}

def copy_file(src_path, dst_path):
    with open(src_path, 'rb') as src_file:
        with open(dst_path, 'wb') as dst_file:
            dst_file.write(src_file.read())
            

@app.route('/generate_animation', methods=['POST'])
def generate_animation():
    data = request.json
    file_id = data.get('file_id')
    total_frames = int(data.get('total_frames', 1))
    
    if not file_id:
        return jsonify({'error': 'Missing file_id'}), 400
    
    if file_id not in svg_files:
        return jsonify({'error': 'File not found'}), 404
    
    # Get original SVG info
    svg_info = svg_files[file_id]
    original_tree = svg_info['tree']
    original_root = svg_info['root']
    namespace = svg_info['namespace']
    
    # Get the original filename (without extension)
    original_filename = os.path.splitext(svg_info['filename'])[0]
    
    # Create the target directory
    output_dir = os.path.join(app.config['AFTER_MOVE_FOLDER'], original_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create motion_data directory (sibling to SVG file)
    motion_data_dir = os.path.join(output_dir, 'motion_data')
    os.makedirs(motion_data_dir, exist_ok=True)
    # Create input directory (sibling to SVG file)
    motion_data_dir = os.path.join(output_dir, 'input')
    os.makedirs(motion_data_dir, exist_ok=True)
    
    output_files = []
    
    # Ensure the root node has a viewBox attribute
    if 'viewBox' not in original_root.attrib:
        width = original_root.get('width', '256')
        height = original_root.get('height', '256')
        original_root.set('viewBox', f"0 0 {width} {height}")
    
    # Debug print
    print(f"Frame positions data: {frame_positions[file_id]}")
    
    # Create motion data record for each group
    motion_data = {}
    for i, group_info in enumerate(groups_info[file_id]):
        group_id = group_info['id']
        motion_data[i] = []  # Use index as key for easier saving by index later
    
    # Generate SVG for each frame
    for frame in range(1, total_frames + 1):
        # Create an independent SVG copy for each frame
        frame_tree = copy.deepcopy(original_tree)
        frame_root = frame_tree.getroot()
        
        # Remove all original path elements to rebuild groupings
        for group_info in groups_info[file_id]:
            for path_id in group_info['path_ids']:
                for path_elem in frame_root.findall(f'.//{{{namespace}}}path[@id="{path_id}"]'):
                    for parent in frame_root.iter():
                        if path_elem in parent:
                            parent.remove(path_elem)
                            break
        
        # Create a <g> element for each group and set correct translation
        for i, group_info in enumerate(groups_info[file_id]):
            group_id = group_info['id']
            group_name = group_info['name']
            path_ids = group_info['path_ids']
            
            # Create group element
            g_element = ET.Element(f"{{{namespace}}}g", {"id": group_id, "class": group_name})
            
            # Copy paths from the original SVG into the group
            for path_id in path_ids:
                original_paths = original_tree.findall(f'.//{{{namespace}}}path[@id="{path_id}"]')
                if original_paths:
                    path_copy = copy.deepcopy(original_paths[0])
                    g_element.append(path_copy)
            
            # Compute the position of the current frame
            current_pos = {'x': 0, 'y': 0}  # default position
            if group_id in frame_positions[file_id]:
                positions = frame_positions[file_id][group_id]
                
                # Ensure integer keys are used
                keyframes = {}
                for k, v in positions.items():
                    try:
                        keyframes[int(k)] = v
                    except:
                        print(f"Error converting key {k} to integer")
                
                # Compute and apply current frame position
                current_pos = calculate_position(frame, keyframes)
                print(f"Frame {frame}, Group {group_id}: Position {current_pos}")
                
                # Only set transform if not at origin
                if current_pos['x'] != 0 or current_pos['y'] != 0:
                    g_element.set('transform', f'translate({current_pos["x"]}, {current_pos["y"]})')
            
            # Record motion data (as integers)
            motion_data[i].append((frame, int(current_pos['x']), int(current_pos['y'])))
            
            # Append group to SVG
            frame_root.append(g_element)

        # Save current frame SVG with new naming format and path
        output_filename = f"frame-{frame}.svg"
        output_path = os.path.join(output_dir, output_filename)
        frame_tree.write(output_path)
        
        # Record frame info without .svg suffix
        display_filename = f"frame-{frame}"
        output_files.append({'frame': frame, 'filename': display_filename, 'path': os.path.join(original_filename, display_filename)})
        
    # Preprocess the first frame
    output_path = os.path.join(output_dir, "frame-1.svg")
    cleaned_svg_path = clean_empty_tags(output_path)
    process_file(cleaned_svg_path)
    copy_file(cleaned_svg_path, os.path.join(output_dir, "input", "postprocess.svg"))
    
    # Save motion data to text files
    for i, positions in motion_data.items():
        motion_file_path = os.path.join(motion_data_dir, f"motion_data_{i}.txt")
        with open(motion_file_path, 'w') as f:
            for frame, x, y in positions:
                f.write(f"frame {frame}: ({x}, {y})\n")
    
    return jsonify({
        'message': 'Animation generated successfully',
        'frames': output_files,
        'total_frames': total_frames,
        'motion_data': 'Motion data files saved in motion_data directory'
    })


@app.route('/download/<path:filepath>', methods=['GET'])
def download_file(filepath):
    # If a filename without extension is provided, append .svg
    if not filepath.endswith('.svg'):
        filepath = f"{filepath}.svg"
    
    # Build the full file path
    full_path = os.path.join(app.config['AFTER_MOVE_FOLDER'], filepath)
    
    # Only handle SVG files
    if filepath.endswith('.svg'):
        with open(full_path, 'r') as f:
            svg_content = f.read()
        
        # Remove namespace prefixes
        svg_content = svg_content.replace('ns0:', '')
        svg_content = svg_content.replace('xmlns:ns0=', 'xmlns=')
        
        return Response(
            svg_content,
            mimetype='image/svg+xml',
            headers={'Content-Disposition': f'attachment; filename={os.path.basename(filepath)}'}
        )
    
    # Other file types
    directory, filename = os.path.split(full_path)
    return send_from_directory(directory, filename)

@app.route('/download_all/<file_id>', methods=['GET'])
def download_all_frames(file_id):
    if file_id not in svg_files:
        return jsonify({'error': 'can not find the file'}), 404
    
    # Get the original filename (without extension)
    original_filename = os.path.splitext(svg_files[file_id]['filename'])[0]
    
    # Target directory
    output_dir = os.path.join(app.config['AFTER_MOVE_FOLDER'], original_filename)
    
    if not os.path.exists(output_dir):
        return jsonify({'error': 'can not find the generated frame animation'}), 404
    
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Get all frames in this directory
        frame_files = [f for f in os.listdir(output_dir) if f.startswith("frame-") and f.endswith(".svg")]
        
        # Sort by frame number
        frame_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
        
        # Add to ZIP (without .svg suffix)
        for frame_file in frame_files:
            file_path = os.path.join(output_dir, frame_file)
            with open(file_path, 'r') as f:
                svg_content = f.read()
            
            svg_content = svg_content.replace('ns0:', '')
            svg_content = svg_content.replace('xmlns:ns0=', 'xmlns=')
            
            frame_name = os.path.splitext(frame_file)[0]
            zf.writestr(frame_name, svg_content)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{original_filename}_frames.zip"
    )

# store the training task status
training_tasks = {}

@app.route('/training_form/<file_id>')
def training_form(file_id):
    if file_id not in svg_files:
        return jsonify({'error': 'File not found'}), 404
        
    svg_info = svg_files[file_id] 
    original_filename = os.path.splitext(svg_info['filename'])[0]
    
    target_path = f"afterMove/{original_filename}/input"
    
    return render_template('training_form.html', 
                         file_id=file_id, 
                         target_path=target_path,
                         filename=original_filename)

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    file_id = data.get('file_id')
    
    if file_id not in svg_files:
        return jsonify({'error': 'File not found'}), 404
    
    params = {
        'caption': data.get('caption', ''),
        'output_folder': data.get('output_folder', 'web_training_output'),
        'num_iter': int(data.get('num_iter', 1000)),
        'rotation_weight': float(data.get('rotation_weight', 0.01)),
        'scale_weight': float(data.get('scale_weight', 0.05)),
        'shear_weight': float(data.get('shear_weight', 0.1)),
        'translation_weight': float(data.get('translation_weight', 0.01)),
        'dino': data.get('dino', False),
        'timeE': data.get('timeE', False)
    }
    
    svg_info = svg_files[file_id]
    original_filename = os.path.splitext(svg_info['filename'])[0]
    target_path = f"afterMove/{original_filename}/input"
    
    if not os.path.exists(target_path):
        return jsonify({'error': f'Target path {target_path} does not exist. Please generate animation first.'}), 400
    
    progress_file = tempfile.mktemp(suffix='.json')
    
    cmd = [
        'python', 'animate_svg.py',
        '--target', target_path,
        '--caption', params['caption'],
        '--output_folder', params['output_folder'],
        '--num_iter', str(params['num_iter']),
        '--rotation_weight', str(params['rotation_weight']),
        '--scale_weight', str(params['scale_weight']),
        '--shear_weight', str(params['shear_weight']),
        '--translation_weight', str(params['translation_weight']),
        '--progress_callback_file', progress_file
    ]
    
    if params['dino']:
        cmd.append('--dino')
    if params['timeE']:
        cmd.append('--timeE')
    
    task_id = f"{file_id}_{int(time.time())}"
    training_tasks[task_id] = {
        'status': 'starting',
        'progress_file': progress_file,
        'output_folder': params['output_folder'],
        'process': None,
        'file_id': file_id
    }
    
    def run_training():
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  
            
            # 使用项目根目录作为工作目录，确保可移植性
            process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            training_tasks[task_id]['process'] = process
            training_tasks[task_id]['status'] = 'running'
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                training_tasks[task_id]['status'] = 'completed'
            else:
                training_tasks[task_id]['status'] = 'error'
                training_tasks[task_id]['error'] = stderr.decode('utf-8')
                
        except Exception as e:
            training_tasks[task_id]['status'] = 'error'
            training_tasks[task_id]['error'] = str(e)
    
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'task_id': task_id,
        'message': 'Training started successfully'
    })

@app.route('/training_progress/<task_id>')
def training_progress(task_id):
    def generate():
        while True:
            if task_id not in training_tasks:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                break
                
            task = training_tasks[task_id]
            
            # read the progress file
            progress_data = {'status': task['status']}
            if os.path.exists(task['progress_file']):
                try:
                    with open(task['progress_file'], 'r') as f:
                        progress_data.update(json.load(f))
                except:
                    pass
            
            # check the task status
            if task['status'] == 'completed':
                progress_data['status'] = 'completed'
                yield f"data: {json.dumps(progress_data)}\n\n"
                break
            elif task['status'] == 'error':
                progress_data['status'] = 'error'
                progress_data['error'] = task.get('error', 'Unknown error')
                yield f"data: {json.dumps(progress_data)}\n\n"
                break
            
            yield f"data: {json.dumps(progress_data)}\n\n"
            time.sleep(2)  # update every 2 seconds
    
    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache'})

@app.route('/get_result_gif/<task_id>')
def get_result_gif(task_id):
    """get the gif file of the training result"""
    if task_id not in training_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = training_tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Training not completed'}), 400
    
    # find the generated gif file
    output_folder = task['output_folder']
    # animate_svg.py will save the result in '/output_videos' directory
    actual_output_path = os.path.join('output_videos', output_folder)
    gif_path = os.path.join(actual_output_path, 'HQ_gif.gif')
    
    if not os.path.exists(gif_path):
        return jsonify({'error': f'Gif file not found at {gif_path}'}), 404
    
    return send_file(gif_path, mimetype='image/gif')

def get_namespace(root):
    # Get SVG namespace
    ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
    return ns if ns else 'http://www.w3.org/2000/svg'

if __name__ == '__main__':
    app.run(port=8081, debug=True)