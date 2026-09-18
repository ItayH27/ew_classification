---
title: Environmental Warfare Classifier
emoji: 🔥
colorFrom: green
colorTo: red
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Verify and score conflict-driven vegetation damage
---

# Environmental Warfare Classifier

By Eran Shufaru and Itay Hazan.

Given a location and an event date, this demo verifies and quantifies environmental damage
(agricultural/forest) from conflicts:

1. **Retrieval**: fetches pre- and post-event Sentinel-2 imagery via Google Earth Engine.
2. **Segmentation**: a UNet trained on the HLS Burn Scar Scenes dataset marks burned vegetation.
3. **Severity**: a logistic-regression classifier over NDVI/NBR scores damage from 0 (none) to 3 (maximal).
4. **Attribution**: searches news coverage near the location and date, and asks Gemini whether
   each article describes a deliberate environmental-warfare incident.

You can also upload a local HLS `*_merged.tif` scene. This skips Earth Engine and attribution.
