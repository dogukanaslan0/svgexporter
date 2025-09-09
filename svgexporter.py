bl_info = {
    "name": "Independent SVG Exporter",
    "author": "Doğukan",
    "version": (1, 4),
    "blender": (4, 0, 0),
    "location": "View3D > N Panel > SVG Exporter",
    "description": "Mesh objeleri bağımsız olarak SVG olarak dışa aktarır (Grease Pencil ve Freestyle olmadan)",
    "category": "Import-Export",
}

import bpy
import os
from math import sqrt
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view


class SVGExporterProperties(bpy.types.PropertyGroup):
    line_width: bpy.props.FloatProperty(
        name="Çizgi Kalınlığı",
        default=2.0,
        min=0.1,
        max=10.0
    )

    export_path: bpy.props.StringProperty(
        name="Export Path",
        description="SVG dosyalarının kaydedileceği klasör",
        default="//svg_export/",
        subtype='DIR_PATH'
    )

    projection_type: bpy.props.EnumProperty(
        name="Projection",
        items=[
            ('ORTHO_XY', 'Top (XY)', 'Üstten görünüm'),
            ('ORTHO_XZ', 'Front (XZ)', 'Önden görünüm'),
            ('ORTHO_YZ', 'Side (YZ)', 'Yandan görünüm'),
            ('CAMERA', 'Camera View', 'Kamera görünümü'),
        ],
        default='ORTHO_XZ'
    )

    export_multi_views: bpy.props.BoolProperty(
        name="Multi Views",
        description="Birden fazla açıdan export et",
        default=False
    )

    canvas_size: bpy.props.IntProperty(
        name="Canvas Size",
        default=1000,
        min=100,
        max=5000
    )

    auto_scale: bpy.props.BoolProperty(
        name="Auto Scale",
        description="Objeyi canvas'a sığacak şekilde ölçekle",
        default=True
    )

    use_render_resolution: bpy.props.BoolProperty(
        name="Use Render Resolution",
        description="Camera View'da SVG boyutunu render çözünürlüğü ile eşle",
        default=True
    )

    clip_to_camera: bpy.props.BoolProperty(
        name="Clip to Camera",
        description="Camera View'da kadraj dışındaki/arkasındaki kenarları ele",
        default=True
    )

    # Figma optimizasyonları (isteğe bağlı)
    optimize_for_figma: bpy.props.BoolProperty(
        name="Optimize for Figma",
        description="DOM ve düğüm sayısını azaltır (seçilince etkin)",
        default=False
    )
    combine_single_path: bpy.props.BoolProperty(
        name="Combine as Single Path",
        description="Seçiliyse tüm kenarlar tek path içinde",
        default=True
    )
    min_edge_pixels: bpy.props.IntProperty(
        name="Min Edge Length (px)",
        description="Bu piksel altındaki kenarları yazma",
        default=1,
        min=0,
        max=100
    )
    decimal_precision: bpy.props.IntProperty(
        name="Decimal Precision",
        description="Koordinat ondalık basamak sayısı",
        default=1,
        min=0,
        max=3
    )
    deduplicate_edges: bpy.props.BoolProperty(
        name="Deduplicate Edges",
        description="Tekrarlayan kenarları kaldır",
        default=True
    )

    # Silhouette / Occlusion seçenekleri
    silhouette_only: bpy.props.BoolProperty(
        name="Silhouette Only",
        description="Sadece kameraya göre dış hat kenarlarını çiz",
        default=False
    )
    occlusion_culling: bpy.props.BoolProperty(
        name="Occlusion Culling",
        description="Arkada kalan (örtülen) kenarları ele",
        default=False
    )
    occlusion_scope: bpy.props.EnumProperty(
        name="Occlusion Scope",
        items=[
            ('SELECTED', 'Selected Only', 'Sadece seçili objeler örtme kaynağı'),
            ('ALL', 'All Meshes', 'Sahnedeki tüm meshler örtme kaynağı'),
        ],
        default='ALL'
    )
    occlusion_samples: bpy.props.IntProperty(
        name="Samples",
        description="Kenar görünürlüğü için ray örnek sayısı",
        default=3,
        min=1,
        max=9
    )
    occlusion_epsilon: bpy.props.FloatProperty(
        name="Epsilon",
        description="Self-hit ve tolerans için küçük ofset (metre)",
        default=1e-4,
        min=1e-6,
        max=1e-2,
        precision=6
    )

    # Gruplama / renk
    group_by_object: bpy.props.BoolProperty(
        name="Group by Object",
        description="SVG içinde objeye göre <g> grupları",
        default=False
    )
    color_by_object: bpy.props.BoolProperty(
        name="Color by Object",
        description="Her objeye farklı stroke rengi",
        default=False
    )


class OBJECT_OT_export_svg(bpy.types.Operator):
    bl_idname = "object.export_svg_independent"
    bl_label = "SVG Olarak Dışa Aktar"
    bl_description = "Seçili objeyi bağımsız SVG olarak export eder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.svg_exporter_props

        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_objects:
            self.report({'ERROR'}, "En az bir mesh obje seçmelisiniz")
            return {'CANCELLED'}

        export_path = bpy.path.abspath(props.export_path)
        if not export_path:
            self.report({'ERROR'}, "Geçersiz export path")
            return {'CANCELLED'}

        try:
            os.makedirs(export_path, exist_ok=True)
        except Exception as e:
            self.report({'ERROR'}, f"Klasör oluşturulamadı: {e}")
            return {'CANCELLED'}

        # Occlusion için BVH hazırlığı (opsiyonel)
        bvh_data = None
        if props.occlusion_culling and props.projection_type == 'CAMERA':
            bvh_data = self.build_scene_bvh(context, props)

        exported_count = 0

        if props.export_multi_views:
            views = ['front', 'side', 'top', 'iso']
            for obj in selected_objects:
                for view in views:
                    if self.export_single_view(obj, view, props, export_path, bvh_data):
                        exported_count += 1
        else:
            for obj in selected_objects:
                if self.export_single_view(obj, props.projection_type, props, export_path, bvh_data):
                    exported_count += 1

        self.report({'INFO'}, f"{exported_count} SVG dosyası export edildi: {export_path}")
        return {'FINISHED'}

    def export_single_view(self, obj, view_type, props, export_path, bvh_data):
        try:
            scene = bpy.context.scene

            # 3B kenarları hesapla ve gerekirse silhouette/occlusion uygula, sonra 2B projekte et
            projected_edges = self.get_projected_edges(obj, view_type, props, bvh_data)
            if not projected_edges:
                print(f"⚠️ Kenar bulunamadı: {obj.name} ({view_type})")
                return False

            # Kamera tuvali boyutu
            if view_type == 'CAMERA':
                if props.use_render_resolution:
                    width = max(1, int(scene.render.resolution_x))
                    height = max(1, int(scene.render.resolution_y))
                else:
                    width = height = int(props.canvas_size)
            else:
                width = height = int(props.canvas_size)

            # Figma optimizasyonları (isteğe bağlı)
            edges_2d = projected_edges
            if props.optimize_for_figma:
                edges_2d = self.optimize_edges(edges_2d, props)

            # SVG üret
            svg_content = self.create_svg_content(
                obj_edges={obj.name: edges_2d} if props.group_by_object else None,
                edges=edges_2d if not props.group_by_object else None,
                props=props,
                width=width,
                height=height,
                color_map={obj.name: self.object_color_hex(obj)} if props.color_by_object else None
            )

            filename = f"{obj.name}_{view_type}.svg" if props.export_multi_views else f"{obj.name}_export.svg"
            filepath = os.path.join(export_path, filename)

            with open(filepath, "w", encoding='utf-8') as f:
                f.write(svg_content)

            print(f"✅ Exported: {filename}")
            return True

        except Exception as e:
            print(f"❌ Export error for {obj.name}: {e}")
            return False

    # ------------------------ Projection + Filtering ------------------------

    def get_projected_edges(self, obj, view_type, props, bvh_data):
        # Evaluated mesh (modifiers dahil)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

        try:
            if not mesh or not mesh.vertices or not mesh.edges:
                return []

            obj_mx = obj.matrix_world
            scene = bpy.context.scene
            camera = scene.camera

            # Kenar->yüz eşlemesi (polygon indeksleri)
            edge_key_to_faces = self.build_edge_face_map(mesh)

            # Silhouette filtrelemesi (opsiyonel, sadece CAMERA’da anlamlı)
            candidate_edges = list(range(len(mesh.edges)))
            if props.silhouette_only and view_type == 'CAMERA' and camera:
                candidate_edges = self.filter_silhouette_edges(mesh, obj_mx, camera, edge_key_to_faces)

            # Occlusion filtrelemesi (opsiyonel, sadece CAMERA’da)
            if props.occlusion_culling and view_type == 'CAMERA' and camera and bvh_data is not None:
                candidate_edges = self.filter_occluded_edges(
                    mesh, obj_mx, camera,
                    candidate_edges,
                    edge_key_to_faces,
                    bvh_data,
                    samples=props.occlusion_samples,
                    eps=props.occlusion_epsilon
                )

            # 2B projeksiyon
            edges_2d = []
            for e_idx in candidate_edges:
                e = mesh.edges[e_idx]
                v1w = obj_mx @ mesh.vertices[e.vertices[0]].co
                v2w = obj_mx @ mesh.vertices[e.vertices[1]].co

                if view_type == 'ORTHO_XY' or view_type == 'top':
                    p1 = (v1w.x, v1w.y); p2 = (v2w.x, v2w.y)
                elif view_type == 'ORTHO_XZ' or view_type == 'front':
                    p1 = (v1w.x, v1w.z); p2 = (v2w.x, v2w.z)
                elif view_type == 'ORTHO_YZ' or view_type == 'side':
                    p1 = (v1w.y, v1w.z); p2 = (v2w.y, v2w.z)
                elif view_type == 'iso':
                    p1 = (v1w.x + v1w.y * 0.5, v1w.z + v1w.y * 0.5)
                    p2 = (v2w.x + v2w.y * 0.5, v2w.z + v2w.y * 0.5)
                elif view_type == 'CAMERA':
                    if camera:
                        c1 = world_to_camera_view(scene, camera, v1w)
                        c2 = world_to_camera_view(scene, camera, v2w)
                        # Kadraj kırpma (opsiyonel, kaba)
                        if props.clip_to_camera:
                            out_left   = (c1.x < 0.0 and c2.x < 0.0)
                            out_right  = (c1.x > 1.0 and c2.x > 1.0)
                            out_bottom = (c1.y < 0.0 and c2.y < 0.0)
                            out_top    = (c1.y > 1.0 and c2.y > 1.0)
                            back_both  = (c1.z < 0.0 and c2.z < 0.0)
                            if out_left or out_right or out_bottom or out_top or back_both:
                                continue
                        p1 = (float(c1.x), float(c1.y))
                        p2 = (float(c2.x), float(c2.y))
                    else:
                        p1 = (v1w.x, v1w.z); p2 = (v2w.x, v2w.z)
                else:
                    p1 = (v1w.x, v1w.z); p2 = (v2w.x, v2w.z)

                # CAMERA için piksel haritalama (0..1 -> piksel), diğerlerinde sonra normalize
                if view_type == 'CAMERA':
                    # 0..1 ekran -> piksel dönüşüm create_svg_content içinde yapılacak, önce bırak
                    edges_2d.append((p1, p2))
                else:
                    edges_2d.append((p1, p2))

            # CAMERA: 0..1 ekran koordinatlarını piksele export sırasında çevireceğiz.
            # Diğerleri: gerekirse normalize_coordinates ile canvas’a sığdırılacak (export aşamasında)
            if view_type != 'CAMERA' and props.auto_scale and edges_2d:
                edges_2d = self.normalize_coordinates(edges_2d, props.canvas_size)

            if view_type == 'CAMERA':
                # 0..1 -> piksel dönüşümü export sırasında yapabilmek için burada bırakıyoruz
                scene = bpy.context.scene
                w = int(scene.render.resolution_x) if props.use_render_resolution else props.canvas_size
                h = int(scene.render.resolution_y) if props.use_render_resolution else props.canvas_size
                edges_2d = self.map_camera_coords_to_canvas(edges_2d, w, h)

            return edges_2d

        finally:
            if mesh:
                eval_obj.to_mesh_clear()

    def build_edge_face_map(self, mesh):
        # edge key: tuple(sorted(v1, v2))
        edge_key_to_index = {}
        for i, e in enumerate(mesh.edges):
            key = tuple(sorted((e.vertices[0], e.vertices[1])))
            edge_key_to_index[key] = i
        edge_faces = {i: [] for i in range(len(mesh.edges))}
        for poly_idx, poly in enumerate(mesh.polygons):
            for li in poly.loop_indices:
                vi1 = mesh.loops[li].vertex_index
                vi2 = mesh.loops[(li + 1 - poly.loop_start) % poly.loop_total + poly.loop_start].vertex_index
                key = tuple(sorted((vi1, vi2)))
                e_idx = edge_key_to_index.get(key)
                if e_idx is not None:
                    edge_faces[e_idx].append(poly_idx)
        return edge_faces

    def filter_silhouette_edges(self, mesh, obj_mx, camera, edge_faces):
        cam_pos = camera.matrix_world.translation
        normal_mx = obj_mx.inverted().transposed()

        def face_center_normal(poly):
            verts = [obj_mx @ mesh.vertices[i].co for i in poly.vertices]
            c = sum(verts, Vector()) / len(verts)
            n = (normal_mx @ poly.normal).normalized()
            return c, n

        # Cache face centers/normals
        face_info = {}
        for poly in mesh.polygons:
            face_info[poly.index] = face_center_normal(poly)

        silhouette_edges = []
        for e_idx, faces in edge_faces.items():
            if len(faces) <= 1:
                silhouette_edges.append(e_idx)
                continue
            f1, f2 = faces[0], faces[1]
            c1, n1 = face_info[f1]
            c2, n2 = face_info[f2]
            v1 = (cam_pos - c1).normalized()
            v2 = (cam_pos - c2).normalized()
            if (n1.dot(v1) * n2.dot(v2)) < 0.0:
                silhouette_edges.append(e_idx)
        return silhouette_edges

    def build_scene_bvh(self, context, props):
        # BVH: sahnedeki (veya seçili) meshlerden dünya uzayında poligon seti
        if props.occlusion_scope == 'SELECTED':
            objs = [o for o in context.selected_objects if o.type == 'MESH']
        else:
            objs = [o for o in context.view_layer.objects if o.type == 'MESH' and o.visible_get()]

        depsgraph = context.evaluated_depsgraph_get()
        vertices = []
        polygons = []
        poly_map = []  # global poly index -> (obj_name, local_poly_idx)

        vert_offset = 0
        for obj in objs:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            if not mesh:
                continue
            mx = obj.matrix_world
            # world vertices
            for v in mesh.vertices:
                vertices.append(mx @ v.co)
            for poly in mesh.polygons:
                polygons.append([vi + vert_offset for vi in poly.vertices])
                poly_map.append((obj.name, poly.index))
            vert_offset += len(mesh.vertices)
            eval_obj.to_mesh_clear()

        if not polygons:
            return None

        bvh = BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
        return {
            "bvh": bvh,
            "poly_map": poly_map
        }

    def filter_occluded_edges(self, mesh, obj_mx, camera, edge_indices, edge_faces, bvh_data, samples=3, eps=1e-4):
        if bvh_data is None or bvh_data["bvh"] is None:
            return edge_indices

        cam_pos = camera.matrix_world.translation
        bvh = bvh_data["bvh"]
        poly_map = bvh_data["poly_map"]

        # Kenar boyunca t örnekleri (uçlardan uzak)
        if samples < 1:
            samples = 1
        ts = [(i + 1) / (samples + 1) for i in range(samples)]

        visible = []
        for e_idx in edge_indices:
            e = mesh.edges[e_idx]
            p0 = obj_mx @ mesh.vertices[e.vertices[0]].co
            p1 = obj_mx @ mesh.vertices[e.vertices[1]].co

            # Kenarın ait olduğu yüzler (self-hit’i filtrelemek için)
            own_faces = set(edge_faces.get(e_idx, []))

            hits_covered = 0
            for t in ts:
                pw = p0.lerp(p1, t)
                dir_vec = pw - cam_pos
                dist = dir_vec.length
                if dist <= 0.0:
                    continue
                dir_n = dir_vec / dist
                # küçük ofset ile kameradan çıkar
                hit = bvh.ray_cast(cam_pos + dir_n * eps, dir_n, dist - eps)
                if hit is not None:
                    _, _, hit_index, _ = hit
                    # Kendi yüzlerinden birine çarpıyorsa örtülü sayma
                    if hit_index is not None:
                        obj_name, poly_idx = poly_map[hit_index]
                        if poly_idx not in own_faces:
                            hits_covered += 1
                # erken çıkış
                if hits_covered > samples // 2:
                    break

            if hits_covered <= samples // 2:
                visible.append(e_idx)

        return visible

    # ------------------------ 2D helpers / optimization / svg ------------------------

    def map_camera_coords_to_canvas(self, edges, width, height):
        mapped = []
        for (x1, y1), (x2, y2) in edges:
            x1c = max(0.0, min(1.0, x1))
            y1c = max(0.0, min(1.0, y1))
            x2c = max(0.0, min(1.0, x2))
            y2c = max(0.0, min(1.0, y2))
            nx1 = x1c * width
            ny1 = (1.0 - y1c) * height
            nx2 = x2c * width
            ny2 = (1.0 - y2c) * height
            mapped.append(((nx1, ny1), (nx2, ny2)))
        return mapped

    def optimize_edges(self, edges, props):
        # 1) Kısa kenarları ele
        if props.min_edge_pixels > 0:
            edges = [e for e in edges if self.edge_length(e) >= props.min_edge_pixels]
        # 2) Ondalık hassasiyeti düşür
        if props.decimal_precision >= 0:
            edges = self.round_edges(edges, props.decimal_precision)
        # 3) Tekrarlayan kenarları ele (yönsüz)
        if props.deduplicate_edges:
            uniq = set()
            filtered = []
            for (p1, p2) in edges:
                key = self.edge_key_undirected(p1, p2)
                if key in uniq:
                    continue
                uniq.add(key)
                filtered.append((p1, p2))
            edges = filtered
        return edges

    def edge_length(self, edge):
        (x1, y1), (x2, y2) = edge
        dx = x2 - x1
        dy = y2 - y1
        return sqrt(dx * dx + dy * dy)

    def round_edges(self, edges, precision):
        factor = 10 ** precision
        def rnd(v):
            return round(v * factor) / factor
        out = []
        for (x1, y1), (x2, y2) in edges:
            out.append(((rnd(x1), rnd(y1)), (rnd(x2), rnd(y2))))
        return out

    def edge_key_undirected(self, p1, p2):
        a, b = (p1, p2) if p1 < p2 else (p2, p1)
        return (round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6))

    def normalize_coordinates(self, edges, canvas_size):
        if not edges:
            return edges
        xs, ys = [], []
        for (x1, y1), (x2, y2) in edges:
            xs.extend([x1, x2]); ys.extend([y1, y2])
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        rx = max_x - min_x
        ry = max_y - min_y
        mr = max(rx, ry) if max(rx, ry) > 0 else 1.0
        margin = 50
        usable = max(1.0, float(canvas_size) - 2.0 * margin)
        scale = usable / mr
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        cxs = canvas_size / 2.0
        cys = canvas_size / 2.0
        out = []
        for (x1, y1), (x2, y2) in edges:
            nx1 = cxs + (x1 - cx) * scale
            ny1 = cys - (y1 - cy) * scale
            nx2 = cxs + (x2 - cx) * scale
            ny2 = cys - (y2 - cy) * scale
            out.append(((nx1, ny1), (nx2, ny2)))
        return out

    def object_color_hex(self, obj):
        # Basit: Objeye göre deterministik renk
        h = hash(obj.name)
        r = ((h >> 0) & 255) / 255.0
        g = ((h >> 8) & 255) / 255.0
        b = ((h >> 16) & 255) / 255.0
        return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    def create_svg_content(self, obj_edges, edges, props, width, height, color_map=None):
        line_width = props.line_width
        precision = max(0, int(props.decimal_precision))

        def svg_header():
            return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
'''
        def svg_footer():
            return '</svg>'

        def lines_group(edges_list, stroke="black"):
            s = f'<g stroke="{stroke}" stroke-width="{line_width}" fill="none" stroke-linecap="round">\n'
            fmt = "{:." + str(precision) + "f}"
            for (x1, y1), (x2, y2) in edges_list:
                s += f'  <line x1="{fmt.format(x1)}" y1="{fmt.format(y1)}" x2="{fmt.format(x2)}" y2="{fmt.format(y2)}"/>\n'
            s += '</g>\n'
            return s

        def single_path(edges_list, stroke="black"):
            fmt = "{:." + str(precision) + "f}"
            d_parts = [f'M {fmt.format(x1)} {fmt.format(y1)} L {fmt.format(x2)} {fmt.format(y2)}' for (x1, y1), (x2, y2) in edges_list]
            d_attr = " ".join(d_parts)
            return f'  <path d="{d_attr}" stroke="{stroke}" stroke-width="{line_width}" fill="none" stroke-linecap="round" />\n'

        # Optimizasyon seçili değilse: her zaman ayrı line (eski davranış)
        if not props.optimize_for_figma:
            if obj_edges is None:
                return svg_header() + lines_group(edges) + svg_footer()
            else:
                out = svg_header()
                for obj_name, eds in obj_edges.items():
                    stroke = color_map.get(obj_name, "black") if color_map else "black"
                    out += lines_group(eds, stroke=stroke)
                out += svg_footer()
                return out

        # Optimizasyon açık:
        if props.combine_single_path:
            if obj_edges is None:
                return svg_header() + single_path(edges) + svg_footer()
            else:
                out = svg_header()
                for obj_name, eds in obj_edges.items():
                    stroke = color_map.get(obj_name, "black") if color_map else "black"
                    out += single_path(eds, stroke=stroke)
                out += svg_footer()
                return out
        else:
            if obj_edges is None:
                return svg_header() + lines_group(edges) + svg_footer()
            else:
                out = svg_header()
                for obj_name, eds in obj_edges.items():
                    stroke = color_map.get(obj_name, "black") if color_map else "black"
                    out += lines_group(eds, stroke=stroke)
                out += svg_footer()
                return out


class SVGExporterPanel(bpy.types.Panel):
    bl_label = "SVG Exporter"
    bl_idname = "VIEW3D_PT_svg_exporter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SVG Export"

    def draw(self, context):
        layout = self.layout
        props = context.scene.svg_exporter_props

        box = layout.box()
        box.label(text="Export Settings", icon='EXPORT')
        box.prop(props, "export_path")
        box.prop(props, "line_width")
        box.prop(props, "canvas_size")

        box = layout.box()
        box.label(text="View Settings", icon='CAMERA_DATA')
        box.prop(props, "export_multi_views")
        if not props.export_multi_views:
            box.prop(props, "projection_type")
            if props.projection_type == 'CAMERA':
                row = box.row(align=True)
                row.prop(props, "use_render_resolution")
                row.prop(props, "clip_to_camera")
        else:
            box.label(text="Will export: Front, Side, Top, Iso")

        box = layout.box()
        box.label(text="Silhouette / Occlusion", icon='MOD_WIREFRAME')
        row = box.row(align=True)
        row.prop(props, "silhouette_only")
        row.prop(props, "occlusion_culling")
        if props.occlusion_culling and props.projection_type == 'CAMERA':
            row2 = box.row(align=True)
            row2.prop(props, "occlusion_scope")
            row3 = box.row(align=True)
            row3.prop(props, "occlusion_samples")
            row3.prop(props, "occlusion_epsilon")

        box = layout.box()
        box.label(text="Grouping / Color", icon='OUTLINER_OB_GROUP_INSTANCE')
        row = box.row(align=True)
        row.prop(props, "group_by_object")
        row.prop(props, "color_by_object")

        box = layout.box()
        box.label(text="Figma Optimization (optional)", icon='MOD_SIMPLIFY')
        box.prop(props, "optimize_for_figma")
        if props.optimize_for_figma:
            box.prop(props, "combine_single_path")
            row = box.row(align=True)
            row.prop(props, "min_edge_pixels")
            row.prop(props, "decimal_precision")
            box.prop(props, "deduplicate_edges")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.6
        row.operator("object.export_svg_independent", text="Export SVG", icon="EXPORT")

        layout.separator()
        box = layout.box()
        box.label(text="Info", icon='INFO')
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        box.label(text=f"Selected meshes: {len(selected_meshes)}")
        if len(selected_meshes) == 0:
            box.label(text="⚠️ Mesh objesi seçin!", icon='ERROR')


classes = (
    SVGExporterProperties,
    OBJECT_OT_export_svg,
    SVGExporterPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.svg_exporter_props = bpy.props.PointerProperty(type=SVGExporterProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.svg_exporter_props


if __name__ == "__main__":
    register()