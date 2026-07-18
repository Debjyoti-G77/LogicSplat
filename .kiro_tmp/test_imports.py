import sys
sys.path.insert(0, ".")
from src.dataset.loader_3rscan_splat import Dataset3RScanSplat, process_scene, load_3dssg_annotations
print("loader_3rscan_splat imports OK")
from src.models.relation_gnn import RelationGNN
from src.training.augmentation import augment_graph
from src.repair.symbolic_repair import SceneGraphRepair
from src.relations.schema import DSSG_TO_SCHEMA, NUM_RELATIONS
print("All imports OK")
print(f"DSSG_TO_SCHEMA has {len(DSSG_TO_SCHEMA)} mappings")
print(f"NUM_RELATIONS = {NUM_RELATIONS}")
