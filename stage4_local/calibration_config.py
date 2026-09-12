# Calibrated from frame0_grid.png (ORIGINAL untrimmed test.mp4, frame 0 -- we moved off
# the 6s-trim approach to run the full 30-second clip instead, trading a bit more
# camera-drift error for ~5x the frames of usable heatmap/rating data).
#
# Same mixed-reference situation as the trimmed-clip calibration: these 4 points are
# NOT one rectangle's corners -- they combine the halfway line's two touchline ends
# (68m apart) with the near edge of the penalty box's two corners (40.3m apart, at a
# different point along the pitch). REAL_CORNERS below gives each point's actual
# real-world coordinate explicitly instead of assuming a single rectangle.
#
# Order (must match PIXEL_CORNERS <-> REAL_CORNERS pointwise):
#   1. top of halfway line    (halfway line x near touchline)
#   2. top of the 18-yard box (box's front-edge corner on the same side as point 1)
#   3. bottom of the 18-yard box (box's front-edge corner on the same side as point 4)
#   4. bottom of halfway line (halfway line x other touchline)
PIXEL_CORNERS = [
    [675, 300],   # top of halfway line
    [1425, 325],  # top of 18-yard box
    [1850, 575],  # bottom of 18-yard box
    [600, 800],   # bottom of halfway line
]

# Standard pitch coordinate system: x = 0-105m (pitch length), y = 0-68m (pitch width).
# Same real-world layout as the trimmed-clip calibration (same physical pitch/box,
# just read off an earlier frame of the same pan): halfway line at x=52.5 spanning the
# full width; penalty box front edge at x=88.5 (16.5m in from the x=105 goal line),
# centered on the pitch (y=34), so its two front corners sit at y=13.85 and y=54.15.
REAL_CORNERS = [
    [52.5, 0],
    [88.5, 13.85],
    [88.5, 54.15],
    [52.5, 68],
]

# Still required (used as the heatmap grid's extent) -- leave at the full pitch since
# REAL_CORNERS above is already expressed in the standard 105 x 68 coordinate system.
PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68
