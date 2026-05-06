# UE5 Python API Reference

## Core modules

```python
import unreal
```

All UE5 Python scripting uses the `unreal` module provided by the Python Script Plugin.

---

## Actor operations

### Spawn an actor from a Blueprint class
```python
actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/Blueprints/BP_MyActor')
location = unreal.Vector(0, 0, 100)
rotation = unreal.Rotator(0, 0, 0)
actor = unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, location, rotation)
```

### Get all actors in the level
```python
actors = unreal.EditorLevelLibrary.get_all_level_actors()
```

### Move an actor
```python
actor.set_actor_location(unreal.Vector(100, 200, 50), False, False)
```

### Delete an actor
```python
unreal.EditorLevelLibrary.destroy_actor(actor)
```

---

## Asset operations

### Load an asset
```python
asset = unreal.load_asset('/Game/MyFolder/MyAsset')
```

### List assets in a path
```python
ar = unreal.AssetRegistryHelpers.get_asset_registry()
assets = ar.get_assets_by_path('/Game', recursive=True)
```

### Import an asset
```python
task = unreal.AssetImportTask()
task.set_editor_property('filename', 'C:/path/to/file.fbx')
task.set_editor_property('destination_path', '/Game/Imported')
task.set_editor_property('replace_existing', True)
task.set_editor_property('automated', True)
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
```

---

## Blueprint operations

### Create a Blueprint
```python
factory = unreal.BlueprintFactory()
factory.set_editor_property('ParentClass', unreal.Actor)
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
bp = asset_tools.create_asset('BP_NewActor', '/Game/Blueprints', unreal.Blueprint, factory)
```

### Compile a Blueprint
```python
blueprint = unreal.load_asset('/Game/Blueprints/BP_MyActor')
unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
```

---

## World / Level

### Get the current editor world
```python
world = unreal.EditorLevelLibrary.get_editor_world()
```

### Save the current level
```python
unreal.EditorLevelLibrary.save_current_level()
```

---

## Logging

```python
unreal.log("Hello from Python")
unreal.log_warning("This is a warning")
unreal.log_error("This is an error")
```
