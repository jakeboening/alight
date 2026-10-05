# alight

A fork of [burnback-3d](https://codeberg.org/iff/burnback-3d) that adds `alight`, a command-line front end
to its solver. `alight` needs no Qt and no display: it reads a mesh, runs the burnback-3d kernels to a
steady burn-time field and writes the result, so the solver can be scripted, run in batch or driven by
another program. The Qt GUI and everything else in burnback-3d are unchanged and documented below.

## alight

Build (any platform with CMake and a C++17 compiler; no other dependencies):

```shell
git clone --recursive https://github.com/jakeboening/alight
cmake -S alight -B alight/build
cmake --build alight/build --config Release
ctest --test-dir alight/build -C Release      # solves a small cube and checks the answer
cmake --install alight/build                  # optional: puts alight on the PATH
```

Prebuilt Linux, Windows and macOS binaries are attached to the
[releases](https://github.com/jakeboening/alight/releases).

Run:

```shell
alight mesh.json out --tol 1e-4 --iters 20000
```

`mesh.json` is the burnback-3d mesh format described under [Usage](#usage) (`tools/mesh_convert.py`
makes one from a Gmsh mesh). Three files are written:

- `out.u.f64`: burn time at every node, in mesh node order, as raw little-endian 64-bit floats. With the
  default recession speed of 1 this is the distance burned (web) when the flame reaches the node.
- `out.meta.json`: node and element counts, iterations, time step, final residual, `converged`, `diverged`.
- `out.history.csv`: residual per iteration.

| Option | Default | Meaning |
| --- | --- | --- |
| `--cfl` | 1 | Pseudo-time step factor. The converged field does not depend on it. |
| `--iters` | 300 | Maximum number of iterations. |
| `--weight` | 1 | Diffusive weight. Values below about 0.75 diverge. |
| `--tol` | 0 | Stop when the largest nodal rate of change falls below this; 0 runs all iterations. |
| `--json` | | Also write the GUI's result file (mesh plus `burnbackResults`). |
| `--expect-max` | | Exit with an error unless the largest burn time is within 5 % of this value. |

The exit status is 0 on success, 3 if the iteration diverged.

The scheme is first-order accurate: the burn time lags by an amount proportional to the element size.
Solving on two meshes and extrapolating to zero element size removes that error; see
[burnback](https://github.com/jakeboening/burnback) for a pipeline that does this.

Changes to burnback-3d in this fork:

- `cli/alight.cpp`, `CMakeLists.txt` and a test mesh (`cli/tests/cube.json`).
- `src/headers/globals.h` and `src/iosystem.cpp` compile without Qt when `ALIGHT_HEADLESS` is defined.
- `src/interface.cpp`: the GUI's time step used a variable that no longer exists (`maxHeight`); it now
  uses `minHeight`.

Licensed under the GNU Affero General Public License v3 or later, like burnback-3d.

---

# Burnback-3d

Analysis of 3D burn surfaces for solid propellant rockets using tetrahedra based Time Marching Method as an alternative of the Level Set Method. Key features:

- Combustion time computation with tetrahedra meshes for solid propellant rockets
- Supports per node configuration of the recession speed
- Supports isotropic and anisotropic propellants
- Graphical interface to evaluate the results
- Import and export data of any mesh formats with a python script

If you only need a 2D analysis, see [burnback-qt](https://codeberg.org/iff/burnback-qt).

Built binaries for Windows and MacOS can be found at [releases](https://github.com/iffse/burnback-3d/releases). For Linux is advisable to compile from source as Qt has no compatibility across different distributions (binaries built with Ubuntu CI didn't work on my Arch Linux). It is also possible to run the Windows binary through Wine, with minor flickers.

Supports both light and dark theme. Should use accordingly to your system theme. If you want dark theme, and it isn't, add `QT_QUICK_CONTROLS_MATERIAL_THEME=Dark` to your environment variables.

![burnback-3d interface](img/interface.png)

The contour visualization is only for preview purposes. For a much more detailed visualization, try [ParaView](https://www.paraview.org/):

![result visualization in paraview](img/paraview.png)

## Usage

First, you will need a mesh in order to use the program for analysis. For instance, [Gmsh](https://gmsh.info/) is an open source meshing software that can generate 2D and 3D finite element mesh.

Once you have the mesh, you will have to convert it into a Json file with tetrahedra based information. A python script that converts Gmsh mesh file to this format using [meshio](https://github.com/nschloe/meshio) can be found at tools directory: [mesh_convert.py](./tools/mesh_convert.py). The script is optimized for speed and can convert a mesh with 200K nodes in seconds. If you are using another meshing tool, feel free to edit it (probable you will only need to change the cell names).

When using Gmsh and the script, you can define boundary conditions with physical groups with the following naming conventions in surfaces:

- `inlet`: The boundary is an inlet, where the propellant starts to burn. You can later give it a value as the boundary in boundary panels, if you want a boundary starts to be already burnt for the given time.
- `outlet`: Used for boundaries where the combustion ends, like the shell of the container.
- `symmetry`: Used to indicate that a boundary defines a symmetry. The condition will automatically find the normal vector pointing outwards of the mesh if the boundary triangles are provided in a standard way (such as the output given by Gmsh). In other case, the vector is reversed and you should manually change its sign in the boundary conditions panel.
- `condition`: Placeholder for conditions that should be changed later in Burnback GUI (will be treated as outlet by default).

In volumes, you can define recession velocities:

- `recession 1 [0.5 0.2 45 30 25]` (optional): Used to indicate the recession velocity of a node, defaults to 1. When more than 1 number is specified, the velocity is considered to be anisotropic: First 3 numbers are the recession speed to the `x`, `y`, and `z` axis respectively, and the last 3 numbers are the rotation angles in degree with respect to axis `x`, `y`, and `z` respectively.

Everything after the names above will be added to a description field.

Example files of Gmsh can be found at [examples/gmsh](./examples/gmsh). The commands to be executed to obtain the Json file are:
```shell
gmsh -3 mesh.geo
python mesh_convert.py mesh.msh
```

If you want to use the exported results to another format other than Json (for instance `.CGNS`, or `.dat` for TecPlot/ParaView, etc.) you can use the [result_convert.py](./tools/result_convert.py) script. Usage is:
```shell
python result_convert.py result.json output.extension
```
You can provide only the extension name for the output file. In this case the name is inferred from the input file.

## Compiling

Can be compiled by either using command line or using the QtCreator. Binaries should be found at `<Project Dir>/target/debug|release`. When building with QtCreator, `<Project Dir>` equals to where the build location is set.

Qt modules dependencies:

- qt-3d
- qt-charts
- qt-declaratives
- qt-quickcontrols2

WARNING: Do not compile with Qt 5.15.2. The 3D model importer is broken for that version.

### Using command line

The project can be easily built with the `Makefile`. The `Makefile` is written with multiplatform compilation in mind, it should work with Linux, macOS, or Microsoft Windows:

- `make run`: Build the debug binary and run
	- `make run-sanitizer`: Build the debug binary with sanitizer and run
- `make debug`: Build the debug binary
- `make release`: Build the release binary

### Using QtCreator

Open `burnback-3d.pro` with QtCreator, set your compiling options if needed, and runs directly by clicking the play button at bottom-left.

