# How ARC-AGI-3 games are built (arcengine)

Every real game is one Python module with one subclass of `arcengine.ARCBaseGame`.
Your engine uses the same library, so its rendering and collision rules are
already exactly those of the real games. You only supply the content (sprites,
levels, UI) and the rules (`step`, `on_set_level`).

## Primitives

| Primitive | Fact |
| --- | --- |
| Screen | Every frame is 64x64 with 16 colours. A game draws onto a smaller logical grid, which is scaled up by a whole number (x1 to x5) and centred with a border colour around it. The budget bar is painted afterwards in screen pixels, so it can sit in that border. |
| Time | Nothing moves without an input. One action produces 1-N frames: the game keeps stepping until it marks the action complete, so extra frames are animation and the last frame is the resulting state. |
| Actions | RESET, ACTION1-4 (directions: 1 up, 2 down, 3 left, 4 right), 5 (interact), 6 (click x,y), 7 (undo). Each game advertises a fixed subset. |
| Levels | Each level is a fresh copy of pristine level data, rebuilt on entry, so the budget refills on a level restart. Completing a level is permanent. Losing triggers game over, after which only RESET is accepted. |
| Sprites | Pixel arrays with a layer, rotation, mirroring, scale and tags. Pixel -1 is transparent. Pixel -2 is invisible but solid: it isn't drawn but does collide. Collision is pixel-exact by default, and a blocked move is reverted. Objects are often "hidden" by switching them to a removed state, not deleted. |
| Determinism | The same state plus the same action always gives the same result. |

## The fixed main loop (`ARCBaseGame.perform_action`, never override it)

```
perform_action(action):
    if action is RESET: handle_reset()          # ONLY_RESET_LEVELS=true -> level_reset()
    elif state is GAME_OVER or WIN: return no frames (state unchanged, levels_completed=0, win_levels=0)
    state = NOT_FINISHED; self.action = action; action_count += 1 (not for RESET)
    frames = []
    while not (action complete and no level change pending):
        if a level change is pending: set_level(current + 1)   # instead of step(); this frame shows the new level
        else: step()                                           # your game logic, also called for RESET
        frames.append(camera.render(current_level.get_sprites()))
    return frames, state, levels_completed (= score), win_levels, available_actions
```

- `level_reset()`: replaces the current level with a fresh clone of its pristine
  copy, then `set_level(current)`, state NOT_FINISHED. `set_level(i)` resizes the
  camera to the level's `grid_size` and calls `on_set_level(level)`.
- `next_level()`: score += 1; if it was the last level, `win()`; otherwise a
  level change is pending, which adds one more frame showing the new level.
- `win()` sets state WIN, `lose()` sets state GAME_OVER (the action still
  returns the frames rendered so far). `complete_action()` ends the action after
  the current frame.
- `ARCBaseGame.__init__(game_id, levels, camera=None, debug=False, win_score=1,
  available_actions=[1,2,3,4,5,6], seed=0)`: clones the levels, win_levels =
  number of levels, calls `set_level(0)` (so `on_set_level` runs inside
  `super().__init__`).
- Handy members: `self.action` (`.id` is a `GameAction`, `.data` has `x`, `y` for
  ACTION6), `self.current_level`, `self.level_index`, `self.camera`,
  `self._score`, `self._state`, `self._action_count`, `self.is_last_level()`,
  `self.try_move(name, dx, dy)` / `self.try_move_sprite(sprite, dx, dy)`: move, and
  if it now collides with any sprite, move back and return the list of sprites hit
  (empty list = moved).

## Sprite

```python
Sprite(pixels, name=None, x=0, y=0, layer=0, scale=1, rotation=0, mirror_ud=False,
       mirror_lr=False, blocking=BlockingMode.PIXEL_PERFECT, interaction=None,
       visible=True, collidable=True, tags=[])
```

- `x`, `y` are logical-grid coordinates of the top-left corner; `move(dx, dy)`,
  `set_position(x, y)`, `width`, `height` (after transforms), `name`, `tags`,
  `layer`, `set_layer`, `pixels` (int8 array; may be reassigned).
- Rendering order of transforms: rotation (0/90/180/270, clockwise) then
  mirror_ud then mirror_lr then scale. `scale` > 1 repeats pixels; -1 halves, -2
  thirds (mode of each block).
- `interaction`: TANGIBLE (drawn, collides), INTANGIBLE (drawn, no collision),
  INVISIBLE (not drawn, collides), REMOVED (neither). `set_visible(bool)`,
  `set_collidable(bool)`, `set_interaction(mode)`, `is_visible`, `is_collidable`.
- `blocking`: PIXEL_PERFECT (default: pixels other than -1 overlap), BOUNDING_BOX,
  NOT_BLOCKED (never collides).
- `collides_with(other)`, `color_remap(old_or_None, new)`, `set_rotation`,
  `rotate(delta)`, `set_scale`, `set_mirror_ud`, `set_mirror_lr`, `clone(new_name=None)`
  (keeps the name), `merge(other)`.

## Level

```python
Level(sprites=None, grid_size=None, data={}, name="Level", placeable_areas=None)
```

- `grid_size=(width, height)` sets the camera size when the level is entered.
- `get_sprites()` (a copy, in insertion order), `get_sprites_by_name(name)`,
  `get_sprites_by_tag(tag)`, `get_sprites_by_tags([...])` (all), `get_sprites_by_any_tag([...])`,
  `get_sprite_at(x, y, tag=None, ignore_collidable=False)` (topmost by layer),
  `add_sprite(s)`, `remove_sprite(s)`, `collides_with(sprite)`, `get_data(key)`, `name`.
- On construction, PIXEL_PERFECT sprites tagged `sys_static` are merged into one
  sprite per layer.

## Camera and rendering

```python
Camera(x=0, y=0, width=64, height=64, background=5, letter_box=5, interfaces=[])
```

`render(sprites)`:
1. Fill a width x height grid with `background`; draw visible sprites sorted by
   `layer` (stable sort: equal layers keep list order, later on top); negative
   pixels are not drawn. The camera's `x`, `y` offset the view.
2. Scale by `s = min(64 // width, 64 // height)`, centre it on a 64x64 screen
   filled with `letter_box` (offset `(64 - width*s) // 2`, `(64 - height*s) // 2`).
3. Call each interface's `render_interface(frame)` in order (screen pixels; it
   returns the frame).

`camera.display_to_grid(x, y)` maps a click's screen pixel to grid coordinates
(None in the letterbox). `RenderableUserDisplay.draw_sprite(frame, sprite, x, y)`
draws a sprite in screen pixels.

## Colours

0 white, 1 light grey, 2 grey, 3 dark grey, 4 darker grey, 5 black, 6 magenta,
7 pink, 8 red, 9 blue, 10 light blue, 11 yellow, 12 orange, 13 maroon, 14 green,
15 purple.
