"""
Build a pedagogical 3D Dynamic Scene Graph using the official MIT-SPARK/Spark-DSG library.

Demonstrates:
- Layer 5 (BUILDINGS): Building-level spatial concept
- Layer 4 (ROOMS): Room-level spatial concepts + intra-layer topological connectivity
- Layer 3 (PLACES): Free-space traversable nodes (with ESDF clearance radius) + topological edges
- Layer 2 (OBJECTS): Semantic 3D objects with bounding boxes
- Inter-layer edges: Hierarchical containment (Building -> Room -> Place -> Object)
- Intra-layer edges: Topological connectivity (traversability through doorways, blocked by walls)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import spark_dsg as dsg


# Architectural floorplan (16m x 12m)
ROOM_SPECS = [
    {
        "idx": 0,
        "name": "Living Room",
        "name_zh": "客厅",
        "semantic_label": 1,
        "bounds": [0.0, 0.0, 6.0, 6.0],  # xmin, ymin, xmax, ymax
        "center": [3.0, 3.0, 0.0],
        "color": [59, 130, 246],
    },
    {
        "idx": 1,
        "name": "Kitchen",
        "name_zh": "厨房",
        "semantic_label": 2,
        "bounds": [0.0, 6.0, 6.0, 12.0],
        "center": [3.0, 9.0, 0.0],
        "color": [245, 158, 11],
    },
    {
        "idx": 2,
        "name": "Corridor",
        "name_zh": "中央走廊",
        "semantic_label": 3,
        "bounds": [6.0, 0.0, 10.0, 12.0],
        "center": [8.0, 6.0, 0.0],
        "color": [16, 185, 129],
    },
    {
        "idx": 3,
        "name": "Office",
        "name_zh": "办公室",
        "semantic_label": 4,
        "bounds": [10.0, 0.0, 16.0, 6.0],
        "center": [13.0, 3.0, 0.0],
        "color": [168, 85, 247],
    },
    {
        "idx": 4,
        "name": "Bedroom",
        "name_zh": "卧室",
        "semantic_label": 5,
        "bounds": [10.0, 6.0, 16.0, 12.0],
        "center": [13.0, 9.0, 0.0],
        "color": [236, 72, 153],
    },
]

# Solid walls as line segments [x1, y1, x2, y2], leaving doorway gaps
WALLS = [
    # Outer boundary
    [0.0, 0.0, 16.0, 0.0],
    [16.0, 0.0, 16.0, 12.0],
    [16.0, 12.0, 0.0, 12.0],
    [0.0, 12.0, 0.0, 0.0],
    # Living Room <-> Kitchen wall (y=6, x in [0, 6]) with doorway at x in [3.7, 4.9]
    [0.0, 6.0, 3.7, 6.0],
    [4.9, 6.0, 6.0, 6.0],
    # Living Room & Kitchen <-> Corridor wall (x=6, y in [0, 12]) with doors at y in [2.4, 3.6] and [8.5, 9.7]
    [6.0, 0.0, 6.0, 2.4],
    [6.0, 3.6, 6.0, 8.5],
    [6.0, 9.7, 6.0, 12.0],
    # Corridor <-> Office & Bedroom wall (x=10, y in [0, 12]) with doors at y in [2.5, 3.7] and [8.4, 9.6]
    [10.0, 0.0, 10.0, 2.5],
    [10.0, 3.7, 10.0, 8.4],
    [10.0, 9.6, 10.0, 12.0],
    # SOLID WALL between Office and Bedroom (y=6, x in [10, 16]) -> NO DOOR!
    [10.0, 6.0, 16.0, 6.0],
]

# Free-space Place nodes (Layer 3)
PLACE_SPECS = [
    # Living Room (R0)
    {"idx": 0, "room": 0, "pos": [2.2, 2.8, 0.0], "distance": 1.35, "desc": "客厅西侧空地"},
    {"idx": 1, "room": 0, "pos": [4.3, 3.0, 0.0], "distance": 1.25, "desc": "客厅东侧通道"},
    {"idx": 2, "room": 0, "pos": [4.3, 4.9, 0.0], "distance": 0.95, "desc": "客厅北门前"},
    # Kitchen (R1)
    {"idx": 3, "room": 1, "pos": [4.3, 7.1, 0.0], "distance": 0.95, "desc": "厨房南门前"},
    {"idx": 4, "room": 1, "pos": [2.5, 9.1, 0.0], "distance": 1.30, "desc": "厨房操作区"},
    {"idx": 5, "room": 1, "pos": [4.6, 9.1, 0.0], "distance": 1.10, "desc": "厨房东侧通道"},
    # Corridor (R2)
    {"idx": 6, "room": 2, "pos": [8.0, 1.5, 0.0], "distance": 1.15, "desc": "走廊南端玄关"},
    {"idx": 7, "room": 2, "pos": [8.0, 3.1, 0.0], "distance": 1.30, "desc": "走廊南交叉口"},
    {"idx": 8, "room": 2, "pos": [8.0, 6.0, 0.0], "distance": 1.35, "desc": "走廊中段"},
    {"idx": 9, "room": 2, "pos": [8.0, 9.0, 0.0], "distance": 1.30, "desc": "走廊北交叉口"},
    {"idx": 10, "room": 2, "pos": [8.0, 10.7, 0.0], "distance": 1.05, "desc": "走廊北端"},
    # Office (R3)
    {"idx": 11, "room": 3, "pos": [11.5, 3.1, 0.0], "distance": 1.15, "desc": "办公室入口"},
    {"idx": 12, "room": 3, "pos": [13.7, 2.5, 0.0], "distance": 1.25, "desc": "办公室白板前"},
    {"idx": 13, "room": 3, "pos": [13.5, 4.5, 0.0], "distance": 0.95, "desc": "办公桌旁（紧邻实墙）"},
    # Bedroom (R4)
    {"idx": 14, "room": 4, "pos": [11.5, 9.0, 0.0], "distance": 1.15, "desc": "卧室入口"},
    {"idx": 15, "room": 4, "pos": [13.4, 9.5, 0.0], "distance": 1.25, "desc": "卧室床边"},
    {"idx": 16, "room": 4, "pos": [13.5, 7.5, 0.0], "distance": 0.95, "desc": "床头柜旁（紧邻实墙）"},
]

# Intra-room free-space edges (always open)
INTRA_ROOM_PLACE_EDGES = [
    # Living room
    (0, 1),
    (1, 2),
    (0, 2),
    # Kitchen
    (3, 4),
    (3, 5),
    (4, 5),
    # Corridor
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    # Office
    (11, 12),
    (11, 13),
    (12, 13),
    # Bedroom
    (14, 15),
    (14, 16),
    (15, 16),
]

# Doorway edges between rooms (dynamically toggleable!)
DOOR_SPECS = [
    {
        "id": "door_lr_kit",
        "name": "Living Room ↔ Kitchen",
        "name_zh": "客厅 ↔ 厨房门",
        "places": [2, 3],
        "rooms": [0, 1],
        "segment": [3.7, 6.0, 4.9, 6.0],
        "pos": [4.3, 6.0],
        "open": True,
    },
    {
        "id": "door_lr_cor",
        "name": "Living Room ↔ Corridor",
        "name_zh": "客厅 ↔ 走廊门",
        "places": [1, 7],
        "rooms": [0, 2],
        "segment": [6.0, 2.4, 6.0, 3.6],
        "pos": [6.0, 3.0],
        "open": True,
    },
    {
        "id": "door_kit_cor",
        "name": "Kitchen ↔ Corridor",
        "name_zh": "厨房 ↔ 走廊门",
        "places": [5, 9],
        "rooms": [1, 2],
        "segment": [6.0, 8.5, 6.0, 9.7],
        "pos": [6.0, 9.1],
        "open": True,
    },
    {
        "id": "door_cor_off",
        "name": "Corridor ↔ Office",
        "name_zh": "走廊 ↔ 办公室门",
        "places": [7, 11],
        "rooms": [2, 3],
        "segment": [10.0, 2.5, 10.0, 3.7],
        "pos": [10.0, 3.1],
        "open": True,
    },
    {
        "id": "door_cor_bed",
        "name": "Corridor ↔ Bedroom",
        "name_zh": "走廊 ↔ 卧室门",
        "places": [9, 14],
        "rooms": [2, 4],
        "segment": [10.0, 8.4, 10.0, 9.6],
        "pos": [10.0, 9.0],
        "open": True,
    },
]

# Semantic Object nodes (Layer 2)
OBJECT_SPECS = [
    {"idx": 0, "place": 0, "name": "Sofa", "name_zh": "沙发", "label": 10, "pos": [1.3, 1.4, 0.45], "dims": [2.0, 0.95, 0.85], "color": [96, 165, 250]},
    {"idx": 1, "place": 0, "name": "TV Stand", "name_zh": "电视柜", "label": 11, "pos": [1.2, 4.9, 0.35], "dims": [1.6, 0.55, 0.65], "color": [129, 140, 248]},
    {"idx": 2, "place": 1, "name": "Coffee Table", "name_zh": "茶几", "label": 12, "pos": [3.5, 1.7, 0.25], "dims": [1.1, 0.7, 0.45], "color": [56, 189, 248]},
    {"idx": 3, "place": 4, "name": "Coffee Mug", "name_zh": "咖啡杯", "label": 13, "pos": [1.4, 8.6, 0.90], "dims": [0.25, 0.25, 0.25], "color": [251, 191, 36]},
    {"idx": 4, "place": 4, "name": "Fridge", "name_zh": "冰箱", "label": 14, "pos": [1.1, 11.0, 0.95], "dims": [0.85, 0.85, 1.85], "color": [245, 158, 11]},
    {"idx": 5, "place": 5, "name": "Dining Table", "name_zh": "餐桌", "label": 15, "pos": [4.3, 10.9, 0.42], "dims": [1.5, 0.95, 0.78], "color": [251, 146, 60]},
    {"idx": 6, "place": 6, "name": "Charging Dock", "name_zh": "机器人充电桩", "label": 16, "pos": [8.0, 0.6, 0.25], "dims": [0.7, 0.5, 0.45], "color": [52, 211, 153]},
    {"idx": 7, "place": 8, "name": "Fire Extinguisher", "name_zh": "灭火器", "label": 17, "pos": [6.6, 6.0, 0.35], "dims": [0.3, 0.3, 0.65], "color": [248, 113, 113]},
    {"idx": 8, "place": 13, "name": "Laptop", "name_zh": "笔记本电脑", "label": 18, "pos": [13.8, 5.2, 0.80], "dims": [0.55, 0.4, 0.3], "color": [192, 132, 252]},
    {"idx": 9, "place": 12, "name": "Whiteboard", "name_zh": "白板", "label": 19, "pos": [15.3, 2.5, 1.15], "dims": [0.25, 1.8, 1.1], "color": [167, 139, 250]},
    {"idx": 10, "place": 11, "name": "Printer", "name_zh": "打印机", "label": 20, "pos": [11.6, 1.0, 0.45], "dims": [0.7, 0.6, 0.75], "color": [139, 92, 246]},
    {"idx": 11, "place": 16, "name": "Bedside Lamp", "name_zh": "床头台灯", "label": 21, "pos": [13.8, 6.8, 0.70], "dims": [0.35, 0.35, 0.55], "color": [244, 114, 182]},
    {"idx": 12, "place": 15, "name": "Bed", "name_zh": "双人床", "label": 22, "pos": [14.7, 10.2, 0.35], "dims": [1.8, 2.1, 0.6], "color": [236, 72, 153]},
    {"idx": 13, "place": 14, "name": "Wardrobe", "name_zh": "衣柜", "label": 23, "pos": [11.3, 11.2, 1.00], "dims": [1.4, 0.65, 1.95], "color": [251, 113, 133]},
]


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_edge_attr(weight: float) -> dsg.EdgeAttributes:
    e = dsg.EdgeAttributes()
    e.weight = float(weight)
    e.weighted = True
    return e


def build_dynamic_scene_graph() -> dsg.DynamicSceneGraph:
    """Construct a full hierarchical 3D DynamicSceneGraph via spark_dsg."""
    G = dsg.DynamicSceneGraph()

    # 1. Layer 5: Building (Top-level Spatial Concept)
    b_attr = dsg.SemanticNodeAttributes()
    b_attr.position = np.array([8.0, 6.0, 0.0], dtype=float)
    b_attr.name = "Smart Lab Building"
    b_attr.semantic_label = 0
    b_attr.color = np.array([226, 232, 240], dtype=np.uint8)
    b_sym = dsg.NodeSymbol("B", 0)
    G.add_node(dsg.DsgLayers.BUILDINGS, b_sym.value, b_attr)

    # 2. Layer 4: Rooms (Room-level Spatial Concepts)
    for r in ROOM_SPECS:
        r_attr = dsg.RoomNodeAttributes()
        r_attr.position = np.array(r["center"], dtype=float)
        r_attr.name = r["name"]
        r_attr.semantic_label = r["semantic_label"]
        r_attr.color = np.array(r["color"], dtype=np.uint8)
        r_sym = dsg.NodeSymbol("R", r["idx"])
        G.add_node(dsg.DsgLayers.ROOMS, r_sym.value, r_attr)
        # Inter-layer containment: Building -> Room
        G.insert_edge(b_sym.value, r_sym.value)

    # 3. Layer 3: Places (Free-space traversable nodes)
    place_by_idx = {p["idx"]: p for p in PLACE_SPECS}
    for p in PLACE_SPECS:
        p_attr = dsg.PlaceNodeAttributes()
        p_attr.position = np.array(p["pos"], dtype=float)
        p_attr.distance = float(p["distance"])
        p_attr.name = f"p{p['idx']}"
        p_attr.real_place = True
        p_sym = dsg.NodeSymbol("p", p["idx"])
        G.add_node(dsg.DsgLayers.PLACES, p_sym.value, p_attr)
        # Inter-layer containment: Room -> Place
        r_sym = dsg.NodeSymbol("R", p["room"])
        G.insert_edge(r_sym.value, p_sym.value)

    # Intra-layer Place edges (within rooms)
    for u, v in INTRA_ROOM_PLACE_EDGES:
        w = _dist(place_by_idx[u]["pos"], place_by_idx[v]["pos"])
        G.insert_edge(
            dsg.NodeSymbol("p", u).value,
            dsg.NodeSymbol("p", v).value,
            make_edge_attr(w),
        )

    # Intra-layer Place edges (doorways) & Room edges
    for door in DOOR_SPECS:
        if door["open"]:
            u, v = door["places"]
            w = _dist(place_by_idx[u]["pos"], place_by_idx[v]["pos"])
            G.insert_edge(
                dsg.NodeSymbol("p", u).value,
                dsg.NodeSymbol("p", v).value,
                make_edge_attr(w),
            )
            ru, rv = door["rooms"]
            rw = _dist(ROOM_SPECS[ru]["center"], ROOM_SPECS[rv]["center"])
            G.insert_edge(
                dsg.NodeSymbol("R", ru).value,
                dsg.NodeSymbol("R", rv).value,
                make_edge_attr(rw),
            )

    # 4. Layer 2: Objects (Semantic 3D entities)
    for o in OBJECT_SPECS:
        o_attr = dsg.ObjectNodeAttributes()
        o_attr.position = np.array(o["pos"], dtype=float)
        o_attr.name = o["name"]
        o_attr.semantic_label = o["label"]
        o_attr.color = np.array(o["color"], dtype=np.uint8)
        o_attr.bounding_box = dsg.BoundingBox(
            np.array(o["dims"], dtype=np.float32),
            np.array(o["pos"], dtype=np.float32),
        )
        o_sym = dsg.NodeSymbol("O", o["idx"])
        G.add_node(dsg.DsgLayers.OBJECTS, o_sym.value, o_attr)
        # Inter-layer containment: Place -> Object
        p_sym = dsg.NodeSymbol("p", o["place"])
        G.insert_edge(p_sym.value, o_sym.value)

    # Compute ancestor bounding boxes for Rooms using official spark_dsg utility
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
    return G


def save_demo_graph(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "smart_lab_dsg.json"
    G = build_dynamic_scene_graph()
    G.save(str(out_path))
    return out_path


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    saved = save_demo_graph(data_dir)
    G = dsg.DynamicSceneGraph.load(str(saved))
    print(f"Saved Spark-DSG JSON to: {saved}")
    print(f"Nodes: {G.num_nodes()}, Total Edges: {G.num_edges()}, Interlayer Edges: {len(list(G.interlayer_edges))}")
    for layer in G.layers:
        print(f"  Layer {layer.id}: {len(list(layer.nodes))} nodes, {len(list(layer.edges))} intra-edges")
