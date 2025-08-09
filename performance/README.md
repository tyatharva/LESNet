# LESNet Viewer

An interactive website to explore machine‑learning predictions of lake‑effect snow and related weather maps. It presents four side‑by‑side maps you can customize, compare, and inspect.
Known issue: the credits can get hidden behind some layers. If you don't see the MAE for a panel, reload the page.
**Refer to info.txt for information on the paper's abstract and figures.**

---

## What you’ll see

- Sidebar (left)
  - Run Model: choose a lake and time, then start a model run
  - Available Data: a list of recent results you can open
  - Panel Controls: choose which layer appears in each of the four maps and adjust layer opacity
  - Credits: links and contact

- Map area (right)
  - Four map panels you can pan and zoom independently
  - A label showing the value beneath your cursor (updates as you move)
  - A colorbar that matches the selected layer so you know what the colors mean

---

## How to use the site (step by step)

1. Pick a lake
   - Use the “Lake” dropdown in the sidebar.

2. Pick a date and time (UTC)
   - Click the “Date & Time (UTC)” field.
   - The calendar activates after you choose a lake.
   - Dates with lake-effect are shaded by their use (train, val, test).
   - Dates with yellow shading (test) will be the best representation of performance.
   - Times are in UTC and accepted at the top of the hour (e.g., 2025‑01‑15 12:00).

3. Run the model
   - Click “Run Model.”
   - If someone else started a run, you’ll see your position in the queue. Only one run is processed at a time.
   - The status updates automatically until it’s done.
   - It should take about 30 seconds to a minute to run.

4. Open the results
   - When the run finishes, it shows up under “Available Data.”
   - It might take a few seconds to autmatically load onto the panels.
   - Click a result in the avaible data tab to load a different date/lake into the map panels.

5. Choose what each panel shows
   - In “Panel Controls,” pick a layer for Panel 1–4 (e.g., model outputs, radar reflectivity, wind, temperature).
   - Adjust “Opacity” to blend the colored layer with the base map.

6. Explore the maps
   - Pan and zoom in each panel.
   - Move your mouse over the map to see the value at that point (units depend on the layer; see below).

---

## What the layers mean (plain language)

- Model predictions
  - LESNet‑A, LESNet‑B: two versions of the model’s snowfall/precipitation prediction.

- Precipitation
  - QPE (past/target): estimates of recent precipitation.

- Radar
  - Reflectivity (dBZ): stronger colors usually indicate heavier precipitation.

- Wind
  - U/V components (knots): horizontal and vertical wind directions/speeds.

- Temperature and humidity
  - TMP (temperature), DPT (dew point): shown in °F or °C depending on the level.
  - THTE (theta‑e) and CAPE: indicators related to storm potential and air mass properties.

- Surface and static fields
  - Elevation, land/sea mask, ice coverage, and derived “flow” patterns.

Each layer uses a matching colorbar image so colors map to sensible ranges (for example, cold to warm, dry to moist, light to heavy).

Tip: The live value readout at the bottom of each panel reflects the same units and preprocessing used to draw the map, so the number you see should match what the color suggests.

---

## Helpful tips

- UTC time only
  - The site uses UTC everywhere so everyone sees the same time.

- “Missing data” message
  - Some dates can’t be processed (for example, if upstream sources were incomplete). Pick a different time if you see this.

- One at a time
  - Runs are processed one after another. If there’s a line, your position in the queue is shown and updates automatically.

- Older results may disappear
  - The site periodically clears older results to save space. If a result you used before is gone, just run the model again for that time.

---

## What’s under the hood (very high level)

- Web page (you see this): the four maps, calendar, controls, and legends.
- Server (works in the background): manages a simple queue, runs the model, and prepares map layers and value lookups.
- Data (temporary): when a run finishes, its layers are listed in “Available Data” for you to explore.

---

## Credits

- Model code: https://github.com/tyatharva/LESNet
- Dataset: https://huggingface.co/datasets/tyatharva/LESNet
- Colormaps reference: https://github.com/blaylockbk/Carpenter_Workshop/blob/main/paint/standard2_colormaps.ipynb
- Questions? Email: tyagiatharva11@gmail.com
