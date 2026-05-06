# Blueprint Patterns for UE5

## Point Light above spawn

```python
import unreal

light_class = unreal.PointLight.static_class()
location = unreal.Vector(0, 0, 300)
light = unreal.EditorLevelLibrary.spawn_actor_from_class(light_class, location)
light_component = light.get_component_by_class(unreal.PointLightComponent)
light_component.set_intensity(5000)
light_component.set_light_color(unreal.LinearColor(1.0, 0.9, 0.7, 1.0))
```

## Grid of static mesh actors

```python
import unreal

asset_path = '/Game/StarterContent/Props/SM_Chair'
rows, cols = 5, 5
spacing = 200.0

for r in range(rows):
    for c in range(cols):
        loc = unreal.Vector(r * spacing, c * spacing, 0)
        actor_class = unreal.EditorAssetLibrary.load_blueprint_class(asset_path)
        if actor_class:
            unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, loc)
```

## Trigger volume

```python
import unreal

trigger_class = unreal.TriggerBox.static_class()
location = unreal.Vector(0, 0, 50)
trigger = unreal.EditorLevelLibrary.spawn_actor_from_class(trigger_class, location)
box = trigger.get_component_by_class(unreal.BoxComponent)
box.set_box_extent(unreal.Vector(200, 200, 100))
```

## Camera fly-by path

```python
import unreal

spline_class = unreal.SplineActor if hasattr(unreal, 'SplineActor') else None
# Use CineCameraActor for cinematic shots
cam_class = unreal.CineCameraActor.static_class()
cam = unreal.EditorLevelLibrary.spawn_actor_from_class(cam_class, unreal.Vector(0, 0, 200))
```

## Gravity gun mechanic (Blueprint component)

The gravity gun requires a Physics Handle component. Typical setup:
1. Add `PhysicsHandleComponent` to the character Blueprint
2. On left-click: `PhysicsHandle.GrabComponentAtLocationWithRotation(hit_component, ...)`
3. On tick: update `PhysicsHandle.SetTargetLocationAndRotation(camera_forward * 300, ...)`
4. On release: `PhysicsHandle.ReleaseComponent()`
