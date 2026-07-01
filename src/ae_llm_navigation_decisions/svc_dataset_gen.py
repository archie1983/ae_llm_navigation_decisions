import json
import random
from typing import List, Dict, Optional

class ProcTHORObjectSampler:
    """
    A class to sample objects for specific room types based on ProcTHOR's
    placement-annotations.json weights.
    """

    def __init__(self, annotations_path: str, random_seed: Optional[int] = None):
        """
        Initialize the sampler with the placement annotations file.

        Args:
            annotations_path: Path to the placement-annotations.json file
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Map room type keys to their annotation keys
        self.room_mapping = {
            'kitchen': 'inKitchens',
            'living_room': 'inLivingRooms',
            'bedroom': 'inBedrooms',
            'bathroom': 'inBathrooms'
        }

        # Get all available object types
        self.all_objects = list(self.annotations['instances'].keys())

    def get_room_objects(self, room_type: str) -> Dict[str, int]:
        """
        Get the weight dictionary for a specific room type.

        Args:
            room_type: One of 'kitchen', 'living_room', 'bedroom', 'bathroom'

        Returns:
            Dictionary mapping object names to their weights
        """
        room_key = self.room_mapping.get(room_type.lower())
        if room_key is None:
            raise ValueError(f"Invalid room type: {room_type}. Must be one of {list(self.room_mapping.keys())}")

        room_objs_with_weights = self.annotations[room_key]
        # add back common objects
        room_objs_with_weights['Doorway'] = 1
        room_objs_with_weights['Doorframe'] = 1
        room_objs_with_weights['Window'] = 1

        return self.annotations[room_key]

    def sample_objects(
            self,
            room_type: str,
            num_objects: int = 10,
            min_weight: int = 1,
            include_zero_weight: bool = True
    ) -> List[str]:
        """
        Sample objects for a given room type based on their weights.

        Args:
            room_type: One of 'kitchen', 'living_room', 'bedroom', 'bathroom'
            num_objects: Number of objects to sample
            min_weight: Minimum weight threshold (objects with weight < min_weight are excluded)
            include_zero_weight: If True, include objects with weight 0 in the sampling pool

        Returns:
            List of sampled object names
        """
        room_weights = self.get_room_objects(room_type)

        # Filter objects based on weight criteria
        eligible_objects = []
        weights = []

        for obj, weight in room_weights.items():
            if weight is None:
                continue
            if include_zero_weight or weight >= min_weight:
                eligible_objects.append(obj)
                weights.append(weight)

        if not eligible_objects:
            raise ValueError(f"No eligible objects found for room type: {room_type}")

        # Ensure we don't sample more objects than available
        sample_size = min(num_objects, len(eligible_objects))

        # Sample objects using weights
        sampled = random.choices(
            population=eligible_objects,
            weights=weights,
            k=sample_size
        )

        return sampled

# Initialize the sampler
sampler = ProcTHORObjectSampler(
    annotations_path='placement-annotations.json',
    #random_seed=83
)

room_types = ['kitchen', 'bathroom', 'living_room', 'bedroom']
rt = random.randrange(0, len(room_types))

all_room_objects = [k for k,v in sampler.get_room_objects(room_types[rt]).items() if v > 0]

max_item_count_in_room = len(all_room_objects)  # how many classes of items do we have in this type of room
item_cnt_to_generate = random.randrange(1, max_item_count_in_room + 1)

room_objects = sampler.sample_objects(room_types[rt], num_objects=item_cnt_to_generate)

print(room_types[rt], " objects:", set(room_objects), len(set(room_objects)), item_cnt_to_generate)

all_objects = set([sample.upper() for sample in sampler.all_objects])
print(all_objects, len(all_objects))