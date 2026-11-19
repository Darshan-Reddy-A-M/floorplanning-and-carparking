# FloorPlan Lite — Student Edition

A frontend-only React project for a simple site/floor-plan concept tool.

## Requirements
- Node.js 20+ recommended
- npm

## Run
```bash
npm install
npm run dev
```
Open the local URL printed by Vite (usually http://localhost:5173).

## Features
- Responsive dashboard and home page
- Plot dimensions and setback inputs
- Room count controls
- Three simple schematic layout options
- SVG 2D floor-plan drawing
- Basic isometric-style preview
- Save/open/delete projects in browser LocalStorage
- Export plan as SVG and project data as JSON
- Basic command parser for bedroom count, bathroom count and plot size

## Limitations
This is a learning prototype. Room arrangements are template-based and may overlap or exceed a valid architectural arrangement for some inputs. It does not validate building codes, room connectivity, structure, accessibility, doors/windows, or construction feasibility. Do not use the generated plan as a construction drawing.

## Suggested next steps
1. Add automated layout validation and visible warnings.
2. Add drag-to-move with geometry constraints.
3. Add room templates and wall/door/window editing.
4. Add a proper 3D view with Three.js or React Three Fiber.
5. Add tests and accessibility improvements.
