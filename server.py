"""
FastAPI backend powered by MIT-SPARK/Spark-DSG (Full 5-Layer DSG + Mesh).

Loads a live spark_dsg.DynamicSceneGraph with:
- L5 BUILDINGS, L4 ROOMS, L3 PLACES, L2 OBJECTS, L1 SEGMENTS + G.mesh
Supports dynamic topological mutations (G.insert_edge / G.remove_edge),
hierarchical planning, L1->L2 segment-to-object inspection, and semantic querying.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import spark_dsg as dsg

from build_scene_graph import (
    DOOR_SPECS,
    OBJECT_SPECS,
    PLACE_SPECS,
    ROOM_SPECS,
    SEGMENT_SPECS,
    WALLS,
    make_edge_attr,
    save_demo_graph,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
DSG_PATH = DATA_DIR / "smart_lab_dsg.json"

LAYER_NAMES = {
    1: "SEGMENTS",
    2: "OBJECTS",
    3: "PLACES",
    4: "ROOMS",
    5: "BUILDINGS",
}

LAYER_NAMES_ZH = {
    1: "几何片段层 (Segments)",
    2: "物体层 (Objects)",
    3: "自由空间层 (Places)",
    4: "房间概念层 (Rooms)",
    5: "建筑概念层 (Buildings)",
}

ROOM_META = {f"R({r['idx']})": r for r in ROOM_SPECS}
PLACE_META = {f"p({p['idx']})": p for p in PLACE_SPECS}
OBJECT_META = {f"O({o['idx']})": o for o in OBJECT_SPECS}
SEGMENT_META = {f"s({s['idx']})": s for s in SEGMENT_SPECS}


class SceneGraphService:
    def __init__(self) -> None:
        if not DSG_PATH.exists():
            save_demo_graph(DATA_DIR)
        self.reset()

    def reset(self) -> None:
        self.G: dsg.DynamicSceneGraph = dsg.DynamicSceneGraph.load(str(DSG_PATH))
        if not self.G.has_layer(dsg.DsgLayers.SEGMENTS):
            save_demo_graph(DATA_DIR)
            self.G = dsg.DynamicSceneGraph.load(str(DSG_PATH))
        self.doors: list[dict[str, Any]] = copy.deepcopy(DOOR_SPECS)

    def _sym_str(self, node_id: int) -> str:
        return str(dsg.NodeSymbol(node_id))

    @staticmethod
    def _layer_int(layer_obj: Any) -> int:
        return int(layer_obj.layer) if hasattr(layer_obj, "layer") else int(layer_obj)

    def _parse_symbol(self, sym_str: str) -> dsg.NodeSymbol:
        s = sym_str.strip()
        if "(" in s and s.endswith(")"):
            return dsg.NodeSymbol(s[0], int(s[2:-1]))
        return dsg.NodeSymbol(s[0], int(s[1:]))

    def _node_label_zh(self, sym: str, fallback: str) -> str:
        if sym == "B(0)":
            return "智能实验室大楼"
        if sym in ROOM_META:
            return ROOM_META[sym]["name_zh"]
        if sym in PLACE_META:
            return PLACE_META[sym]["desc"]
        if sym in OBJECT_META:
            return OBJECT_META[sym]["name_zh"]
        if sym in SEGMENT_META:
            return SEGMENT_META[sym]["name_zh"]
        return fallback

    def _node_color(self, layer: int, sym: str) -> str:
        if sym == "B(0)":
            return "#e2e8f0"
        if sym in ROOM_META:
            r, g, b = ROOM_META[sym]["color"]
            return f"rgb({r},{g},{b})"
        if sym in OBJECT_META:
            r, g, b = OBJECT_META[sym]["color"]
            return f"rgb({r},{g},{b})"
        if sym in SEGMENT_META:
            r, g, b = SEGMENT_META[sym]["color"]
            return f"rgb({r},{g},{b})"
        if layer == 3:
            return "#22d3ee"
        return "#94a3b8"

    def serialize_graph(self) -> dict[str, Any]:
        nodes_out = []
        has_mesh = self.G.has_mesh()
        mesh = self.G.mesh if has_mesh else None

        for node in self.G.nodes:
            sym = str(node.id)
            attrs = node.attributes
            pos = [round(float(x), 3) for x in attrs.position]
            parent_raw = node.get_parent()
            parent_sym = self._sym_str(parent_raw) if parent_raw is not None else None
            children_syms = [self._sym_str(c) for c in node.children()]
            siblings_syms = [self._sym_str(s) for s in node.siblings()]
            layer_num = self._layer_int(node.layer)

            item: dict[str, Any] = {
                "id": sym,
                "raw_id": int(node.id.value),
                "category": node.id.category,
                "layer": layer_num,
                "layer_name": LAYER_NAMES.get(layer_num, str(layer_num)),
                "layer_name_zh": LAYER_NAMES_ZH.get(layer_num, str(layer_num)),
                "name": getattr(attrs, "name", sym) or sym,
                "name_zh": self._node_label_zh(sym, getattr(attrs, "name", sym)),
                "position": pos,
                "color": self._node_color(layer_num, sym),
                "parent": parent_sym,
                "children": children_syms,
                "siblings": siblings_syms,
            }
            if layer_num == 1:
                mconns = [int(v) for v in attrs.mesh_connections]
                item["mesh_connections"] = mconns
                item["semantic_label"] = int(attrs.semantic_label)
                if sym in SEGMENT_META:
                    item["cluster_sim"] = SEGMENT_META[sym]["sim"]
                if mesh is not None and mconns:
                    poly = []
                    for vid in mconns:
                        vp = mesh.pos(vid)
                        poly.append([round(float(vp[0]), 3), round(float(vp[1]), 3), round(float(vp[2]), 3)])
                    item["mesh_polygon"] = poly
                bbox = attrs.bounding_box
                item["bbox_dims"] = [round(float(x), 3) for x in bbox.dimensions]
            if layer_num == 3:
                item["distance"] = round(float(attrs.distance), 3)
            if layer_num in (2, 4):
                bbox = attrs.bounding_box
                item["bbox_dims"] = [round(float(x), 3) for x in bbox.dimensions]
                item["semantic_label"] = int(attrs.semantic_label)
            if sym in ROOM_META:
                item["bounds"] = ROOM_META[sym]["bounds"]
            nodes_out.append(item)

        intra_edges = []
        for layer in self.G.layers:
            for e in layer.edges:
                intra_edges.append(
                    {
                        "source": self._sym_str(e.source),
                        "target": self._sym_str(e.target),
                        "layer": int(layer.id),
                        "weight": round(float(e.info.weight), 3),
                        "kind": "intra",
                    }
                )

        inter_edges = []
        for e in self.G.interlayer_edges:
            u_node = self.G.get_node(e.source)
            v_node = self.G.get_node(e.target)
            inter_edges.append(
                {
                    "source": self._sym_str(e.source),
                    "target": self._sym_str(e.target),
                    "source_layer": self._layer_int(u_node.layer),
                    "target_layer": self._layer_int(v_node.layer),
                    "kind": "inter",
                }
            )

        layer_counts = {LAYER_NAMES[int(l.id)]: len(list(l.nodes)) for l in self.G.layers}

        return {
            "stats": {
                "total_nodes": int(self.G.num_nodes()),
                "total_edges": int(self.G.num_edges()),
                "intra_edges": len(intra_edges),
                "inter_edges": len(inter_edges),
                "layers": layer_counts,
                "mesh_vertices": int(mesh.num_vertices()) if mesh is not None else 0,
                "mesh_faces": int(mesh.num_faces()) if mesh is not None else 0,
            },
            "nodes": nodes_out,
            "intra_edges": intra_edges,
            "inter_edges": inter_edges,
            "walls": WALLS,
            "doors": self.doors,
            "rooms_floorplan": ROOM_SPECS,
        }

    def toggle_door(self, door_id: str) -> dict[str, Any]:
        door = next((d for d in self.doors if d["id"] == door_id), None)
        if door is None:
            raise HTTPException(status_code=404, detail=f"Unknown door: {door_id}")

        u_idx, v_idx = door["places"]
        ru_idx, rv_idx = door["rooms"]
        p_u, p_v = dsg.NodeSymbol("p", u_idx), dsg.NodeSymbol("p", v_idx)
        r_u, r_v = dsg.NodeSymbol("R", ru_idx), dsg.NodeSymbol("R", rv_idx)

        door["open"] = not door["open"]
        if door["open"]:
            pos_u = self.G.get_node(p_u.value).attributes.position
            pos_v = self.G.get_node(p_v.value).attributes.position
            w_p = float(math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1]))
            self.G.insert_edge(p_u.value, p_v.value, make_edge_attr(w_p))

            rpos_u = self.G.get_node(r_u.value).attributes.position
            rpos_v = self.G.get_node(r_v.value).attributes.position
            w_r = float(math.hypot(rpos_u[0] - rpos_v[0], rpos_u[1] - rpos_v[1]))
            if not self.G.has_edge(r_u.value, r_v.value):
                self.G.insert_edge(r_u.value, r_v.value, make_edge_attr(w_r))

            code = (
                f"# 打开【{door['name_zh']}】：在 Spark-DSG 中恢复拓扑连通边\n"
                f"G.insert_edge(NodeSymbol('p', {u_idx}).value, NodeSymbol('p', {v_idx}).value, weight={w_p:.2f})\n"
                f"G.insert_edge(NodeSymbol('R', {ru_idx}).value, NodeSymbol('R', {rv_idx}).value, weight={w_r:.2f})"
            )
        else:
            if self.G.has_edge(p_u.value, p_v.value):
                self.G.remove_edge(p_u.value, p_v.value)
            still_connected = any(
                d["open"] and set(d["rooms"]) == {ru_idx, rv_idx} for d in self.doors
            )
            if not still_connected and self.G.has_edge(r_u.value, r_v.value):
                self.G.remove_edge(r_u.value, r_v.value)

            code = (
                f"# 关闭【{door['name_zh']}】：在 Spark-DSG 中移除拓扑连通边\n"
                f"G.remove_edge(NodeSymbol('p', {u_idx}).value, NodeSymbol('p', {v_idx}).value)\n"
                f"G.remove_edge(NodeSymbol('R', {ru_idx}).value, NodeSymbol('R', {rv_idx}).value)"
            )

        return {
            "door": door,
            "executed_code": code,
            "graph": self.serialize_graph(),
        }

    def _resolve_to_place_and_room(self, sym_str: str) -> tuple[str, str, str]:
        """Return (original_sym, place_sym, room_sym), supporting L1 Segments -> L4 Rooms."""
        sym = self._parse_symbol(sym_str)
        if not self.G.has_node(sym.value):
            raise HTTPException(status_code=404, detail=f"Node not found: {sym_str}")
        node = self.G.get_node(sym.value)
        layer = self._layer_int(node.layer)
        if layer == 1:  # Segment -> Object -> Place -> Room
            o_node = self.G.get_node(node.get_parent())
            p_raw = o_node.get_parent()
            p_node = self.G.get_node(p_raw)
            r_raw = p_node.get_parent()
            return str(node.id), self._sym_str(p_raw), self._sym_str(r_raw)
        if layer == 2:  # Object -> Place -> Room
            p_raw = node.get_parent()
            p_node = self.G.get_node(p_raw)
            r_raw = p_node.get_parent()
            return str(node.id), self._sym_str(p_raw), self._sym_str(r_raw)
        if layer == 3:  # Place -> Room
            r_raw = node.get_parent()
            return str(node.id), str(node.id), self._sym_str(r_raw)
        if layer == 4:  # Room -> first child Place
            children = list(node.children())
            if not children:
                raise HTTPException(status_code=400, detail=f"Room {sym_str} has no places")
            return str(node.id), self._sym_str(children[0]), str(node.id)
        raise HTTPException(status_code=400, detail="请选择 Segment / Object / Place / Room 节点进行路径规划")

    def _dijkstra(self, layer_id: Any, start_sym: str, goal_sym: str) -> tuple[list[str], float]:
        layer = self.G.get_layer(layer_id)
        adj: dict[str, list[tuple[str, float]]] = {str(n.id): [] for n in layer.nodes}
        for e in layer.edges:
            u = self._sym_str(e.source)
            v = self._sym_str(e.target)
            w = float(e.info.weight) if e.info.weighted else 1.0
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

        dist: dict[str, float] = {start_sym: 0.0}
        prev: dict[str, str | None] = {start_sym: None}
        pq: list[tuple[float, str]] = [(0.0, start_sym)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            if u == goal_sym:
                break
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if goal_sym not in dist:
            return [], float("inf")

        path = []
        cur: str | None = goal_sym
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, dist[goal_sym]

    @staticmethod
    def _segments_intersect(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> bool:
        def ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        )

    def _check_straight_line_blocked(self, pos_a: list[float], pos_b: list[float]) -> tuple[bool, str]:
        p1 = (pos_a[0], pos_a[1])
        p2 = (pos_b[0], pos_b[1])
        for w in WALLS:
            if self._segments_intersect(p1, p2, (w[0], w[1]), (w[2], w[3])):
                return True, "实心墙体阻隔"
        for d in self.doors:
            if not d["open"]:
                s = d["segment"]
                if self._segments_intersect(p1, p2, (s[0], s[1]), (s[2], s[3])):
                    return True, f"已关闭的门（{d['name_zh']}）阻隔"
        return False, "直线无墙体阻挡"

    def plan_hierarchical(self, start_id: str, goal_id: str) -> dict[str, Any]:
        s_orig, s_place, s_room = self._resolve_to_place_and_room(start_id)
        g_orig, g_place, g_room = self._resolve_to_place_and_room(goal_id)

        s_node = self.G.get_node(self._parse_symbol(s_orig).value)
        g_node = self.G.get_node(self._parse_symbol(g_orig).value)
        s_pos = [float(x) for x in s_node.attributes.position]
        g_pos = [float(x) for x in g_node.attributes.position]

        euclidean_dist = round(math.hypot(s_pos[0] - g_pos[0], s_pos[1] - g_pos[1]), 2)
        blocked, block_reason = self._check_straight_line_blocked(s_pos, g_pos)

        room_path, _ = self._dijkstra(dsg.DsgLayers.ROOMS, s_room, g_room)
        place_path, place_dist = self._dijkstra(dsg.DsgLayers.PLACES, s_place, g_place)

        reachable = len(place_path) > 0
        topo_dist = None
        detour_ratio = None
        waypoints: list[list[float]] = []

        if reachable:
            if s_orig != s_place:
                waypoints.append([round(s_pos[0], 3), round(s_pos[1], 3), round(s_pos[2], 3)])
                sp_pos = self.G.get_node(self._parse_symbol(s_place).value).attributes.position
                extra_s = math.hypot(s_pos[0] - sp_pos[0], s_pos[1] - sp_pos[1])
            else:
                extra_s = 0.0

            for pid in place_path:
                pp = self.G.get_node(self._parse_symbol(pid).value).attributes.position
                waypoints.append([round(float(pp[0]), 3), round(float(pp[1]), 3), 0.0])

            if g_orig != g_place:
                gp_pos = self.G.get_node(self._parse_symbol(g_place).value).attributes.position
                extra_g = math.hypot(g_pos[0] - gp_pos[0], g_pos[1] - gp_pos[1])
                waypoints.append([round(g_pos[0], 3), round(g_pos[1], 3), round(g_pos[2], 3)])
            else:
                extra_g = 0.0

            topo_dist = round(place_dist + extra_s + extra_g, 2)
            detour_ratio = round(topo_dist / max(euclidean_dist, 0.01), 2)

        room_path_pretty = [
            {
                "id": rid,
                "name_zh": self._node_label_zh(rid, rid),
                "name": ROOM_META[rid]["name"] if rid in ROOM_META else rid,
            }
            for rid in room_path
        ]

        code = (
            f"# 1. 沿层间边将起点/目标上溯到自由空间 (L3) 与房间概念 (L4)\n"
            f"start_place = '{s_place}'  # room={s_room}, from {s_orig}\n"
            f"goal_place  = '{g_place}'  # room={g_room}, from {g_orig}\n"
            f"# 2. 高层空间概念规划（DsgLayers.ROOMS 拓扑图）\n"
            f"room_path   = {' -> '.join(room_path) if room_path else 'UNREACHABLE'}\n"
            f"# 3. 底层自由空间规划（DsgLayers.PLACES Dijkstra）\n"
            f"place_path  = {len(place_path)} 个无碰撞航点, 拓扑距离={topo_dist}m"
        )

        if not reachable:
            summary = (
                f"⚠️ 拓扑不连通（Disconnected）！尽管欧氏直线距离为 {euclidean_dist} m，"
                f"但在 Spark-DSG 的 Places/Rooms 层中不存在连通路径（检查是否关闭了必经门洞）。"
            )
        elif blocked and detour_ratio and detour_ratio >= 2.0:
            summary = (
                f"🧱 典型的【几何邻近 ≠ 拓扑连通】案例！直线距离仅 {euclidean_dist} m（被{block_reason}），"
                f"机器人必须沿拓扑边绕行 {topo_dist} m（绕行倍率 {detour_ratio}x），"
                f"依次经过：{' → '.join(r['name_zh'] for r in room_path_pretty)}。"
            )
        else:
            summary = (
                f"✅ 层次化规划成功：先在 L4 Rooms 层规划房间序列（{' → '.join(r['name_zh'] for r in room_path_pretty)}），"
                f"再在 L3 Places 层生成 {len(place_path)} 个无碰撞航点，总拓扑路程 {topo_dist} m（直线 {euclidean_dist} m）。"
            )

        return {
            "reachable": reachable,
            "start": {
                "id": s_orig,
                "name_zh": self._node_label_zh(s_orig, s_orig),
                "place": s_place,
                "room": s_room,
                "room_zh": self._node_label_zh(s_room, s_room),
                "position": s_pos,
            },
            "goal": {
                "id": g_orig,
                "name_zh": self._node_label_zh(g_orig, g_orig),
                "place": g_place,
                "room": g_room,
                "room_zh": self._node_label_zh(g_room, g_room),
                "position": g_pos,
            },
            "euclidean_distance": euclidean_dist,
            "straight_line_blocked": blocked,
            "block_reason": block_reason,
            "topological_distance": topo_dist,
            "detour_ratio": detour_ratio,
            "room_path": room_path_pretty,
            "place_path": place_path,
            "waypoints": waypoints,
            "summary": summary,
            "executed_code": code,
        }

    def semantic_query(self, query: str, start_id: str = "O(6)") -> dict[str, Any]:
        q = query.strip().lower()
        if not q:
            return {"matches": [], "plan": None}

        matches = []
        for node in self.G.nodes:
            layer = self._layer_int(node.layer)
            if layer not in (1, 2, 4):  # Match Segments, Objects, and Rooms!
                continue
            sym = str(node.id)
            name_en = (getattr(node.attributes, "name", "") or "").lower()
            name_zh = self._node_label_zh(sym, name_en).lower()
            if q in name_en or q in name_zh or name_zh in q or (name_en and name_en in q):
                chain = []
                cur = node
                while cur is not None:
                    csym = str(cur.id)
                    chain.append(
                        {
                            "id": csym,
                            "layer": LAYER_NAMES[self._layer_int(cur.layer)],
                            "name_zh": self._node_label_zh(csym, getattr(cur.attributes, "name", csym)),
                        }
                    )
                    p_raw = cur.get_parent()
                    cur = self.G.get_node(p_raw) if p_raw is not None else None
                matches.append(
                    {
                        "id": sym,
                        "layer": LAYER_NAMES[layer],
                        "name": getattr(node.attributes, "name", sym),
                        "name_zh": self._node_label_zh(sym, sym),
                        "hierarchy": chain,
                    }
                )

        # Prefer exact Segment/Object first, then Room
        priority = {"SEGMENTS": 0, "OBJECTS": 1, "ROOMS": 2}
        matches.sort(key=lambda m: (priority.get(m["layer"], 9), m["id"]))
        auto_plan = self.plan_hierarchical(start_id, matches[0]["id"]) if matches else None
        return {"query": query, "matches": matches, "plan": auto_plan}


service = SceneGraphService()
app = FastAPI(title="Spark-DSG 5-Layer 3D Scene Graph Interactive Demo")


class DoorToggleReq(BaseModel):
    door_id: str


class PlanReq(BaseModel):
    start_id: str
    goal_id: str


class QueryReq(BaseModel):
    query: str
    start_id: str = "O(6)"


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    return service.serialize_graph()


@app.post("/api/toggle_door")
def toggle_door(req: DoorToggleReq) -> dict[str, Any]:
    return service.toggle_door(req.door_id)


@app.post("/api/plan")
def plan_path(req: PlanReq) -> dict[str, Any]:
    return service.plan_hierarchical(req.start_id, req.goal_id)


@app.post("/api/query")
def query_scene(req: QueryReq) -> dict[str, Any]:
    return service.semantic_query(req.query, req.start_id)


@app.post("/api/reset")
def reset_graph() -> dict[str, Any]:
    service.reset()
    return service.serialize_graph()


@app.get("/api/dsg_json")
def get_dsg_json() -> dict[str, Any]:
    raw_text = DSG_PATH.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    seg_samples = [n for n in parsed.get("nodes", []) if n.get("layer") == 1][:2]
    obj_samples = [n for n in parsed.get("nodes", []) if n.get("layer") == 2][:2]
    return {
        "file": "data/smart_lab_dsg.json",
        "size_bytes": len(raw_text.encode("utf-8")),
        "top_level_keys": list(parsed.keys()),
        "mesh_summary": {
            "num_vertices": service.G.mesh.num_vertices() if service.G.has_mesh() else 0,
            "num_faces": service.G.mesh.num_faces() if service.G.has_mesh() else 0,
        },
        "sample_L1_segment_nodes": seg_samples,
        "sample_L2_object_nodes": obj_samples,
        "sample_edges": parsed.get("edges", [])[:4],
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
