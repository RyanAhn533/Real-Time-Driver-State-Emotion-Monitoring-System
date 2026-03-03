# External Context modality
# TODO: Implement for final integration
#
# Planned approach:
#   - External camera → YOLO object/scene detection
#   - Detected objects → text description (e.g., "person driving car, rain, night")
#   - Text → sentence embedding (e.g., sentence-transformers/all-MiniLM-L6-v2)
#   - Context embedding + emotion embedding → cross-attention or simple concat → final output
#
# This is the lowest priority modality.
