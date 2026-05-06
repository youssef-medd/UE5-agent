# Key Unreal Engine 5 Python Classes

## Editor Libraries

| Class | Purpose |
|---|---|
| `unreal.EditorLevelLibrary` | Spawn, destroy, move actors; save levels |
| `unreal.EditorAssetLibrary` | Load, duplicate, delete assets |
| `unreal.AssetToolsHelpers` | Import assets, create new assets |
| `unreal.AssetRegistryHelpers` | Query asset registry by path/class |
| `unreal.BlueprintEditorLibrary` | Compile Blueprints |

## Actor Classes

| Class | Purpose |
|---|---|
| `unreal.Actor` | Base class for all placeable objects |
| `unreal.StaticMeshActor` | Actor with a static mesh |
| `unreal.PointLight` | Omnidirectional light source |
| `unreal.DirectionalLight` | Sun/moon directional light |
| `unreal.SpotLight` | Cone-shaped light |
| `unreal.TriggerBox` | Invisible box that fires overlap events |
| `unreal.TriggerSphere` | Sphere-shaped trigger |
| `unreal.CineCameraActor` | Cinematic camera |

## Components

| Class | Purpose |
|---|---|
| `unreal.StaticMeshComponent` | Renders a static mesh |
| `unreal.PointLightComponent` | Point light parameters |
| `unreal.BoxComponent` | Collision box |
| `unreal.SphereComponent` | Collision sphere |
| `unreal.PhysicsHandleComponent` | Grab and move physics objects |
| `unreal.SplineComponent` | Spline path |

## Math Types

| Type | Example |
|---|---|
| `unreal.Vector` | `unreal.Vector(x, y, z)` |
| `unreal.Rotator` | `unreal.Rotator(pitch, yaw, roll)` |
| `unreal.Transform` | `unreal.Transform(location, rotation, scale)` |
| `unreal.LinearColor` | `unreal.LinearColor(r, g, b, a)` |

## Import / Factory Classes

| Class | Purpose |
|---|---|
| `unreal.AssetImportTask` | Configure a single file import |
| `unreal.BlueprintFactory` | Create new Blueprint assets |
| `unreal.StaticMeshFactory` | Import FBX/OBJ as static mesh |
