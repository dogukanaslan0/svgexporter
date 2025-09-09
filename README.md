# Independent SVG Exporter for Blender

![Blender Logo](https://upload.wikimedia.org/wikipedia/commons/0/0c/Blender_logo_no_text.svg)

A Blender add-on to export **Mesh** and **Curve/Bezier** objects as independent **SVG files**.  
Supports multiple views, camera projection with clipping, and automatic canvas scaling.

---

## Features

- Export **Mesh** and **Curve/Bezier** objects to SVG
- Top / Front / Side / Isometric views
- Camera view with **near/far clipping**
- Automatic scaling to fit canvas
- Multi-view export for all angles
- Simple and clean SVG output

---

## Installation

1. Download or clone this repository.
2. In Blender, go to `Edit > Preferences > Add-ons > Install`.
3. Select the `.zip` file or folder containing `__init__.py`.
4. Enable the add-on.
5. Open the **N Panel** in the 3D Viewport, find the **SVG Export** tab.

---

## Usage

1. Select one or more **Mesh** or **Curve/Bezier** objects.
2. Configure settings in the **SVG Export** panel:
   - Export path
   - Canvas size
   - Line width
   - Auto-scale option
   - Projection type or multi-view
3. Click **Export SVG**.
4. Check the export folder for the generated SVG files.

---

## Example

- Export a single mesh from **top view** → `MyMesh_export.svg`
- Export multi-view (Front, Side, Top, Iso) → `MyMesh_front.svg`, `MyMesh_side.svg`, etc.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
<img width="2683" height="745" alt="svgexport" src="https://github.com/user-attachments/assets/4ed024d6-1260-42d8-aa20-84121798eed9" />

## Screenshots

*Optional: Add some Blender N-Panel screenshots showing settings and exported SVG previews.*

---

