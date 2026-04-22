# Map Editor

Interactive visual editor for designing custom POGEMA grid maps.

<a id="editor-link" href="../../editor.html" target="_blank" class="md-button md-button--primary" style="font-size: 1.1em; padding: 0.7em 2em;">
  Open Map Editor
</a>

<script>
  // Pass current theme to editor via URL param
  var link = document.getElementById('editor-link');
  var base = '../../editor.html';
  function updateLink() {
    var theme = document.body.getAttribute('data-md-color-scheme') === 'slate' ? 'slate' : 'default';
    link.href = base + '?theme=' + theme;
  }
  updateLink();
  new MutationObserver(updateLink).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
</script>

---

## Features

- **Resize grid** — expand or shrink the map from any edge or corner
- **Quick presets** — instantly switch to 8x8, 16x16, or 32x32 grids
- **Obstacles** — click cells to toggle obstacles, or randomize with a density slider
- **Agents & targets** — add up to 16 agents, drag them and their targets into position
- **Auto-generate** — randomly place agents and targets with one click
- **Keyboard controls** — `Tab` to cycle agents, arrow keys / WASD / HJKL to move, `Space` to toggle agent/target focus
- **Copy snippet** — exports a ready-to-use `GridConfig(map=...)` Python snippet
- **Download SVG** — save the current map as a vector image
