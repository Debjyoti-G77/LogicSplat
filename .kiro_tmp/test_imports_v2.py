import sys
sys.path.insert(0, ".")
from src.models.relation_gnn import RelationGNN, CONTACT_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.training.augmentation import augment_graph
from src.repair.symbolic_repair import SceneGraphRepair
print("All imports OK")
print(f"NUM_RELATIONS={NUM_RELATIONS}")
print(f"CONTACT_INDICES={CONTACT_INDICES}")
