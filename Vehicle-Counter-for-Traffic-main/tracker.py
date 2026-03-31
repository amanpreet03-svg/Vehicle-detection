# tracker.py
# This file keeps track of vehicles across video frames
# Each vehicle gets a unique ID so we don't count it twice

class VehicleTracker:
    def __init__(self):
        # Dictionary to store center points of each tracked vehicle
        # Key = vehicle ID, Value = list of center positions
        self.tracked_vehicles = {}
        self.next_id = 0  # ID counter for new vehicles

    def update(self, detections):
        """
        detections = list of (cx, cy) center points from current frame
        Returns updated dictionary of tracked vehicles
        """
        updated = {}

        for (cx, cy) in detections:
            matched_id = None

            # Check if this detection matches any existing tracked vehicle
            for vehicle_id, positions in self.tracked_vehicles.items():
                last_cx, last_cy = positions[-1]

                # If the new detection is within 50 pixels of a tracked vehicle
                # we consider it the same vehicle
                distance = ((cx - last_cx)**2 + (cy - last_cy)**2) ** 0.5

                if distance < 50:
                    matched_id = vehicle_id
                    break

            if matched_id is None:
                # New vehicle detected — give it a new ID
                matched_id = self.next_id
                self.next_id += 1

            # Update position history for this vehicle
            if matched_id not in updated:
                updated[matched_id] = self.tracked_vehicles.get(matched_id, [])
            updated[matched_id].append((cx, cy))

        self.tracked_vehicles = updated
        return self.tracked_vehicles