# Spark-DSG Interactive Scene Graph Demo

An interactive, zero-3D-dizziness **Linked Dual-View WebUI** built directly on top of **[MIT-SPARK/Spark-DSG](https://github.com/MIT-SPARK/Spark-DSG)** (`spark_dsg`) to intuitively explain **3D Scene Graphs**, **Spatial Concepts**, and **Topological Connectivity**.

## Why Linked Dual-View?

Traditional 3D exploded scene graph figures work well in papers, but free-orbiting 3D stacks often cause disorientation and overlapping labels during interactive exploration. This demo decouples physical geometry from graph structure into two synchronized views:

- **Left Pane — Fixed North-Up 2D Floorplan (`16m × 12m`)**: Unmistakable spatial landmarks (5 rooms, furniture icons, walls, solid no-door barrier, directly clickable doors on the walls, optional ESDF clearance circles & robot path animation).
- **Right Pane — Orthogonal 4-Layer Scene Graph**:
  - **Vertical dashed/dotted edges (`↑↓`)** strictly represent **inter-layer hierarchy** (`Building → Rooms → Places → Objects`).
  - **Horizontal solid edges (`←→`)** strictly represent **intra-layer topological connectivity** (`Room ↔ Room`, `Place ↔ Place`).
- **Bidirectional Real-Time Linkage**:
  - Hover or click any node/edge on either view to highlight its full vertical ancestry chain and horizontal topological neighbors on both sides.
  - Click any doorway pill (`🔓开 / 🔒关`) directly on the left floorplan to execute real `G.remove_edge()` / `G.insert_edge()` calls on the live `spark_dsg.DynamicSceneGraph` instance and watch both the graph edges and planned path update immediately.

## Built-in Interactive Experiments

1. **🧱 Wall Trap (`Geometric Proximity ≠ Topological Connectivity`)**: `Laptop O(8)` → `Bedside Lamp O(11)` — only `1.6 m` apart across a solid wall, yet `19.36 m` (`12.1x`) via topological routing (`Office → Corridor → Bedroom`).
2. **🚪 Dynamic Door Rerouting**: `Sofa O(0)` → `Coffee Mug O(3)` — `10.73 m` direct when the Living Room ↔ Kitchen door is open; click the door on the map to close it and watch the path immediately reroute through the Corridor (`20.08 m`).
3. **🤖 Long-Range Semantic Navigation**: Two-stage hierarchical planning (`DsgLayers.ROOMS` macro plan → `DsgLayers.PLACES` free-space Dijkstra) plus keyword semantic query.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Regenerate data/smart_lab_dsg.json using spark_dsg
python build_scene_graph.py

# Start the interactive WebUI server
uvicorn server:app --host 127.0.0.1 --port 8765
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

## Project Structure

- `build_scene_graph.py` — Constructs the 4-layer `spark_dsg.DynamicSceneGraph` (37 nodes, 62 edges) and saves it via `G.save()` to `data/smart_lab_dsg.json`.
- `server.py` — FastAPI backend managing the live `DynamicSceneGraph`, dynamic door mutations, hierarchical Dijkstra planning, and semantic queries.
- `static/index.html` — Zero-dependency SVG Linked Dual-View WebUI.
- `data/smart_lab_dsg.json` — Serialized `Spark-DSG` JSON file.

## Credits

Powered by [MIT-SPARK/Spark-DSG](https://github.com/MIT-SPARK/Spark-DSG) (MIT SPARK Lab).
