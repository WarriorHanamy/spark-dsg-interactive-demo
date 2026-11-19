"""
Build a pedagogical 5-Layer 3D Dynamic Scene Graph using the official MIT-SPARK/Spark-DSG library.

Demonstrates the complete Spark-DSG hierarchy:
- Layer 5 (BUILDINGS): Building-level spatial concept
- Layer 4 (ROOMS): Room-level spatial concepts + intra-layer topological connectivity
- Layer 3 (PLACES): Free-space traversable nodes (with ESDF clearance radius) + topological edges
- Layer 2 (OBJECTS): Semantic 3D objects with bounding boxes
- Layer 1 (SEGMENTS): Pre-object 3D geometric/semantic mesh segments linked to G.mesh vertices
- Underlying Mesh (G.mesh): Real spark_dsg.Mesh vertices & triangular faces for each segment
"""

from __future__ import annotations

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
    (0, 1), (1, 2), (0, 2),
    (3, 4), (3, 5), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10),
    (11, 12), (11, 13), (12, 13),
    (14, 15), (14, 16), (15, 16),
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
    {"idx": 3, "place": 4, "name": "Coffee Mug", "name_zh": "咖啡杯", "label": 13, "pos": [1.4, 8.6, 0.90], "dims": [0.45, 0.45, 0.25], "color": [251, 191, 36]},
    {"idx": 4, "place": 4, "name": "Fridge", "name_zh": "冰箱", "label": 14, "pos": [1.1, 11.0, 0.95], "dims": [0.95, 0.90, 1.85], "color": [245, 158, 11]},
    {"idx": 5, "place": 5, "name": "Dining Table", "name_zh": "餐桌", "label": 15, "pos": [4.3, 10.9, 0.42], "dims": [1.5, 0.95, 0.78], "color": [251, 146, 60]},
    {"idx": 6, "place": 6, "name": "Charging Dock", "name_zh": "机器人充电桩", "label": 16, "pos": [8.0, 0.6, 0.25], "dims": [0.8, 0.55, 0.45], "color": [52, 211, 153]},
    {"idx": 7, "place": 8, "name": "Fire Extinguisher", "name_zh": "灭火器", "label": 17, "pos": [6.6, 6.0, 0.35], "dims": [0.4, 0.4, 0.65], "color": [248, 113, 113]},
    {"idx": 8, "place": 13, "name": "Laptop", "name_zh": "笔记本电脑", "label": 18, "pos": [13.8, 5.2, 0.80], "dims": [0.85, 0.55, 0.3], "color": [192, 132, 252]},
    {"idx": 9, "place": 12, "name": "Whiteboard", "name_zh": "白板", "label": 19, "pos": [15.3, 2.5, 1.15], "dims": [0.4, 1.8, 1.1], "color": [167, 139, 250]},
    {"idx": 10, "place": 11, "name": "Printer", "name_zh": "打印机", "label": 20, "pos": [11.6, 1.0, 0.45], "dims": [0.85, 0.65, 0.75], "color": [139, 92, 246]},
    {"idx": 11, "place": 16, "name": "Bedside Lamp", "name_zh": "床头台灯", "label": 21, "pos": [13.8, 6.8, 0.70], "dims": [0.5, 0.5, 0.55], "color": [244, 114, 182]},
    {"idx": 12, "place": 15, "name": "Bed", "name_zh": "双人床", "label": 22, "pos": [14.7, 10.2, 0.35], "dims": [1.8, 2.1, 0.6], "color": [236, 72, 153]},
    {"idx": 13, "place": 14, "name": "Wardrobe", "name_zh": "衣柜", "label": 23, "pos": [11.3, 11.2, 1.00], "dims": [1.4, 0.65, 1.95], "color": [251, 113, 133]},
]

# Layer 1: Pre-Object 3D Geometric / Semantic Segments (DsgLayers.SEGMENTS)
# Each segment owns a real triangulated mesh patch in G.mesh (referenced by mesh_connections)
# and clusters upward into a parent Layer 2 Object!
# patch: [xmin, ymin, xmax, ymax, z_height]
SEGMENT_SPECS = [
    # O(0) Sofa -> 3 geometric segments (Seat Cushion, Backrest, Armrest)
    {"idx": 0, "object": 0, "name": "sofa_seat", "name_zh": "沙发座垫", "patch": [0.45, 1.22, 2.15, 1.86, 0.42], "color": [56, 189, 248], "sim": 0.95},
    {"idx": 1, "object": 0, "name": "sofa_backrest", "name_zh": "沙发靠背", "patch": [0.45, 0.90, 2.15, 1.22, 0.85], "color": [96, 165, 250], "sim": 0.93},
    {"idx": 2, "object": 0, "name": "sofa_armrest", "name_zh": "沙发扶手", "patch": [2.15, 0.90, 2.38, 1.86, 0.62], "color": [129, 140, 248], "sim": 0.89},

    # O(1) TV Stand -> 2 segments
    {"idx": 3, "object": 1, "name": "tv_cabinet", "name_zh": "电视柜体", "patch": [0.40, 4.60, 2.00, 5.18, 0.45], "color": [99, 102, 241], "sim": 0.92},
    {"idx": 4, "object": 1, "name": "tv_screen", "name_zh": "屏幕面板", "patch": [0.60, 5.00, 1.80, 5.24, 1.05], "color": [165, 180, 252], "sim": 0.88},

    # O(2) Coffee Table -> 2 segments
    {"idx": 5, "object": 2, "name": "table_top", "name_zh": "茶几台面", "patch": [2.95, 1.35, 4.05, 2.05, 0.45], "color": [14, 165, 233], "sim": 0.94},
    {"idx": 6, "object": 2, "name": "table_frame", "name_zh": "金属底座", "patch": [3.10, 1.48, 3.90, 1.92, 0.15], "color": [56, 189, 248], "sim": 0.87},

    # O(3) Coffee Mug -> 1 segment (small single-patch object)
    {"idx": 7, "object": 3, "name": "mug_body", "name_zh": "陶瓷杯身", "patch": [1.18, 8.38, 1.62, 8.82, 0.92], "color": [251, 191, 36], "sim": 0.98},

    # O(4) Fridge -> 2 segments
    {"idx": 8, "object": 4, "name": "fridge_upper", "name_zh": "冷藏箱门", "patch": [0.62, 10.55, 1.58, 11.10, 1.75], "color": [245, 158, 11], "sim": 0.96},
    {"idx": 9, "object": 4, "name": "fridge_lower", "name_zh": "冷冻抽屉", "patch": [0.62, 11.10, 1.58, 11.45, 0.85], "color": [252, 211, 77], "sim": 0.91},

    # O(5) Dining Table -> 2 segments
    {"idx": 10, "object": 5, "name": "dining_surface", "name_zh": "餐桌面板", "patch": [3.55, 10.42, 5.05, 11.38, 0.76], "color": [251, 146, 60], "sim": 0.95},
    {"idx": 11, "object": 5, "name": "dining_legs", "name_zh": "桌腿支架", "patch": [3.70, 10.56, 4.90, 11.24, 0.30], "color": [253, 186, 116], "sim": 0.86},

    # O(6) Charging Dock -> 2 segments
    {"idx": 12, "object": 6, "name": "dock_base", "name_zh": "充电底板", "patch": [7.55, 0.35, 8.45, 0.92, 0.12], "color": [16, 185, 129], "sim": 0.93},
    {"idx": 13, "object": 6, "name": "dock_tower", "name_zh": "集尘立柱", "patch": [7.72, 0.28, 8.28, 0.62, 0.45], "color": [52, 211, 153], "sim": 0.90},

    # O(7) Fire Extinguisher -> 1 segment
    {"idx": 14, "object": 7, "name": "extinguisher_tank", "name_zh": "红色罐体", "patch": [6.38, 5.78, 6.82, 6.22, 0.62], "color": [248, 113, 113], "sim": 0.97},

    # O(8) Laptop -> 2 segments (Screen + Keyboard Base)
    {"idx": 15, "object": 8, "name": "laptop_screen", "name_zh": "LCD屏幕", "patch": [13.38, 5.26, 14.22, 5.52, 1.02], "color": [192, 132, 252], "sim": 0.94},
    {"idx": 16, "object": 8, "name": "laptop_keyboard", "name_zh": "键盘底座", "patch": [13.38, 4.90, 14.22, 5.26, 0.78], "color": [168, 85, 247], "sim": 0.92},

    # O(9) Whiteboard -> 2 segments
    {"idx": 17, "object": 9, "name": "board_surface", "name_zh": "书写面板", "patch": [15.12, 1.65, 15.52, 3.35, 1.45], "color": [167, 139, 250], "sim": 0.95},
    {"idx": 18, "object": 9, "name": "board_tray", "name_zh": "笔槽边框", "patch": [15.02, 1.65, 15.16, 3.35, 0.85], "color": [196, 181, 253], "sim": 0.88},

    # O(10) Printer -> 2 segments
    {"idx": 19, "object": 10, "name": "printer_body", "name_zh": "打印机身", "patch": [11.18, 0.68, 12.02, 1.32, 0.55], "color": [139, 92, 246], "sim": 0.93},
    {"idx": 20, "object": 10, "name": "scanner_lid", "name_zh": "扫描顶盖", "patch": [11.24, 0.75, 11.96, 1.25, 0.78], "color": [167, 139, 250], "sim": 0.90},

    # O(11) Bedside Lamp -> 2 segments
    {"idx": 21, "object": 11, "name": "lamp_shade", "name_zh": "布艺灯罩", "patch": [13.55, 6.55, 14.05, 7.05, 0.95], "color": [244, 114, 182], "sim": 0.94},
    {"idx": 22, "object": 11, "name": "lamp_base", "name_zh": "底座灯柱", "patch": [13.65, 6.65, 13.95, 6.95, 0.55], "color": [251, 182, 206], "sim": 0.89},

    # O(12) Bed -> 3 geometric segments (Mattress, Headboard, Pillows)
    {"idx": 23, "object": 12, "name": "bed_mattress", "name_zh": "床垫主体", "patch": [13.80, 9.15, 15.15, 11.25, 0.45], "color": [236, 72, 153], "sim": 0.96},
    {"idx": 24, "object": 12, "name": "bed_headboard", "name_zh": "床头靠背", "patch": [15.35, 9.15, 15.65, 11.25, 0.95], "color": [219, 39, 119], "sim": 0.91},
    {"idx": 25, "object": 12, "name": "bed_pillows", "name_zh": "枕头区域", "patch": [15.15, 9.35, 15.35, 11.05, 0.62], "color": [249, 168, 212], "sim": 0.88},

    # O(13) Wardrobe -> 2 segments
    {"idx": 26, "object": 13, "name": "wardrobe_left", "name_zh": "左推拉门", "patch": [10.60, 10.88, 11.30, 11.55, 1.95], "color": [251, 113, 133], "sim": 0.95},
    {"idx": 27, "object": 13, "name": "wardrobe_right", "name_zh": "右推拉门", "patch": [11.30, 10.88, 12.00, 11.55, 1.95], "color": [253, 164, 175], "sim": 0.95},
]

# Intra-layer L1 Segment geometric adjacency edges (touching mesh patches within an object)
SEGMENT_ADJ_EDGES = [
    (0, 1), (0, 2), (1, 2),  # Sofa parts touch
    (3, 4),                   # TV stand + screen
    (5, 6),                   # Coffee table top + frame
    (8, 9),                   # Fridge upper + lower
    (10, 11),                 # Dining table top + legs
    (12, 13),                 # Dock base + tower
    (15, 16),                 # Laptop screen + keyboard
    (17, 18),                 # Whiteboard surface + tray
    (19, 20),                 # Printer body + lid
    (21, 22),                 # Lamp shade + base
    (23, 25), (25, 24),       # Bed mattress <-> pillows <-> headboard
    (26, 27),                 # Wardrobe left <-> right door
]


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_edge_attr(weight: float) -> dsg.EdgeAttributes:
    e = dsg.EdgeAttributes()
    e.weight = float(weight)
    e.weighted = True
    return e


def _make_color(rgb: list[int]) -> dsg.Color:
    c = dsg.Color()
    c.r, c.g, c.b, c.a = int(rgb[0]), int(rgb[1]), int(rgb[2]), 255
    return c


def build_dynamic_scene_graph() -> dsg.DynamicSceneGraph:
    """Construct a complete 5-layer hierarchical 3D DynamicSceneGraph + Mesh via spark_dsg."""
    G = dsg.DynamicSceneGraph()
    # Explicitly register Layer 1: DsgLayers.SEGMENTS (not added by default constructor!)
    G.add_layer(1, 0, dsg.DsgLayers.SEGMENTS)

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
        r_sym = dsg.NodeSymbol("R", p["room"])
        G.insert_edge(r_sym.value, p_sym.value)

    for u, v in INTRA_ROOM_PLACE_EDGES:
        w = _dist(place_by_idx[u]["pos"], place_by_idx[v]["pos"])
        G.insert_edge(dsg.NodeSymbol("p", u).value, dsg.NodeSymbol("p", v).value, make_edge_attr(w))

    for door in DOOR_SPECS:
        if door["open"]:
            u, v = door["places"]
            w = _dist(place_by_idx[u]["pos"], place_by_idx[v]["pos"])
            G.insert_edge(dsg.NodeSymbol("p", u).value, dsg.NodeSymbol("p", v).value, make_edge_attr(w))
            ru, rv = door["rooms"]
            rw = _dist(ROOM_SPECS[ru]["center"], ROOM_SPECS[rv]["center"])
            G.insert_edge(dsg.NodeSymbol("R", ru).value, dsg.NodeSymbol("R", rv).value, make_edge_attr(rw))

    # 4. Layer 2: Objects (Semantic 3D entities)
    obj_by_idx = {o["idx"]: o for o in OBJECT_SPECS}
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
        p_sym = dsg.NodeSymbol("p", o["place"])
        G.insert_edge(p_sym.value, o_sym.value)

    # 5. Layer 1: SEGMENTS + Underlying 3D Mesh (G.mesh)
    mesh = dsg.Mesh()
    num_segs = len(SEGMENT_SPECS)
    mesh.resize_vertices(num_segs * 4)
    faces_list = []

    seg_centers: dict[int, list[float]] = {}
    for seg in SEGMENT_SPECS:
        idx = seg["idx"]
        x0, y0, x1, y1, zh = seg["patch"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        seg_centers[idx] = [cx, cy, zh]
        parent_obj = obj_by_idx[seg["object"]]

        # Create 4 mesh vertices (2 triangular faces) for this segment patch
        v_base = idx * 4
        quad_pts = [
            [x0, y0, zh],
            [x1, y0, zh],
            [x1, y1, zh],
            [x0, y1, zh],
        ]
        col = _make_color(seg["color"])
        for k, pt in enumerate(quad_pts):
            vid = v_base + k
            mesh.set_pos(vid, np.array(pt, dtype=np.float32))
            mesh.set_color(vid, col)
            mesh.set_label(vid, int(parent_obj["label"]))

        faces_list.append([v_base, v_base + 1, v_base + 2])
        faces_list.append([v_base, v_base + 2, v_base + 3])

        # Create real Spark-DSG L1 Segment node with mesh_connections & bounding box
        s_attr = dsg.ObjectNodeAttributes()
        s_attr.position = np.array([cx, cy, zh], dtype=float)
        s_attr.name = seg["name"]
        s_attr.semantic_label = int(parent_obj["label"])
        s_attr.color = np.array(seg["color"], dtype=np.uint8)
        s_attr.mesh_connections = [v_base, v_base + 1, v_base + 2, v_base + 3]
        s_attr.bounding_box = dsg.BoundingBox(
            np.array([max(0.05, x1 - x0), max(0.05, y1 - y0), max(0.05, zh)], dtype=np.float32),
            np.array([cx, cy, zh / 2.0], dtype=np.float32),
        )
        # Compact mock semantic feature vector (simulating CLIP/VLM embedding per segment)
        ang = float(seg["object"]) * 0.42 + (1.0 - seg["sim"]) * 0.6
        feat = np.array([math.cos(ang), math.sin(ang), seg["sim"], 0.25], dtype=np.float32)
        feat = feat / np.linalg.norm(feat)
        s_attr.semantic_feature = feat

        s_sym = dsg.NodeSymbol("s", idx)
        G.add_node(dsg.DsgLayers.SEGMENTS, s_sym.value, s_attr)
        # Inter-layer containment: Object (L2) -> Segment (L1)
        o_sym = dsg.NodeSymbol("O", seg["object"])
        G.insert_edge(o_sym.value, s_sym.value)

    mesh.set_faces(np.array(faces_list, dtype=np.uint32).T)
    G.mesh = mesh

    # Intra-layer L1 Segment geometric adjacency edges
    for u, v in SEGMENT_ADJ_EDGES:
        w = _dist(seg_centers[u], seg_centers[v])
        G.insert_edge(dsg.NodeSymbol("s", u).value, dsg.NodeSymbol("s", v).value, make_edge_attr(w))

    # Compute ancestor bounding boxes for Rooms using official spark_dsg utility
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
    return G


def save_demo_graph(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "smart_lab_dsg.json"
    G = build_dynamic_scene_graph()
    G.save(str(out_path), include_mesh=True)
    return out_path


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    saved = save_demo_graph(data_dir)
    G = dsg.DynamicSceneGraph.load(str(saved))
    print(f"Saved 5-Layer Spark-DSG JSON (with Mesh) to: {saved}")
    print(
        f"Nodes: {G.num_nodes()}, Total Edges: {G.num_edges()}, "
        f"Interlayer Edges: {len(list(G.interlayer_edges))}, "
        f"Mesh: {G.mesh.num_vertices()} verts / {G.mesh.num_faces()} faces"
    )
    for layer in G.layers:
        print(f"  Layer {layer.id}: {len(list(layer.nodes))} nodes, {len(list(layer.edges))} intra-edges")
