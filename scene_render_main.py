
import os
from multiprocessing import freeze_support
import argparse

from util.dataset_utils import load_scene_data
from scene_render.create_table_top_scene import SceneDatasetGenerator, TableScene

if __name__ == "__main__":
    freeze_support()
    rood_dir = os.path.dirname(os.path.abspath(__file__))

    # Argument parser
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument('--data_dir', help='Root dir with grasps, meshes, mesh_contacts and splits',
                         type=str, default=os.path.join(rood_dir, "data"))
    parser.add_argument('--gripper_path', type=str,
                        default=None)
    parser.add_argument('--splits', type=str, default='test')
    parser.add_argument('--min_num_objects', type=int, default=8)
    parser.add_argument('--max_num_objects', type=int, default=13)
    parser.add_argument('--max_iterations', type=int, default=100)

    parser.add_argument('--start_index', type=int, default=0, help = "Where to start indexing scenes. Use in case of continuing dataset generation.")
    parser.add_argument('--end_idx', type=int, default=10)
    parser.add_argument('--n_processors', type=int, default=1)

    args = parser.parse_args()
    args.data_dir = "/hdd/datasets/suction_grasp_dataset"
    args.gripper_path = "/home/jure/programming/SuctionCupModel/meshes/EPick_extend_sg_collision.stl"

    # Create dataset
    if True:
        # Generate a dataset of 3D Scenes
        dg = SceneDatasetGenerator(args.data_dir,
                            args.gripper_path,
                            args.splits,
                            args.min_num_objects,
                            args.max_num_objects,
                            args.max_iterations,
        )

        # Generate a dataset of 3D Scenes
        dg.generate_N_scenes(args.start_index, args.end_idx, n_processors=args.n_processors)
    
    # Visualize dataset
    if True:
        scene_idx = "000000"
        table_scene = TableScene(args.splits, args.gripper_path, args.data_dir)
        scene_dir = os.path.join(args.data_dir, f"scenes_3d/{args.splits}")
        table_scene.load_scene(scene_idx, scene_dir, visualize=True)


