import os
from pathlib import Path
from typing import Optional

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, pairs_from_retrieval, pairs_from_covisibility, match_dense, triangulation

import pycolmap

# ref_path: path to empty colmap model. only with camera poses. (images.txt)
def SfM(images_path: Path, output_path: Path, ref_path: Optional[Path]=None, mode="triangulate", use_retrieval=False, fix_intrinsics=True):
    sfm_pairs = output_path / 'pairs-sfm.txt'
    sfm_dir = output_path / 'sfm'
    features = output_path / 'features.h5'
    matches = output_path / 'matches.h5'

    """Use feature point + feature matcher"""
    # feature_conf = extract_features.confs['superpoint_aachen']
    # feature_conf = extract_features.confs['superpoint_inloc']
    feature_conf = extract_features.confs['disk']
    # feature_conf = extract_features.confs['sift']
    # matcher_conf = match_features.confs['superglue']
    # matcher_conf = match_features.confs['superpoint+lightglue']
    matcher_conf = match_features.confs['disk+lightglue']
    # matcher_conf = match_features.confs['NN-ratio']


    if mode == "triangulate":
        with open(ref_path / 'images.txt', 'r') as f:
            lines = f.readlines()
            # references = ["images/"+line.strip().split(' ')[-1] for line in lines if line.strip()]
            references = [line.strip().split(' ')[-1] for line in lines if line.strip()]
    elif mode == "sfm":
        references = [p.relative_to(images_path).as_posix() for p in (images_path / 'images').iterdir()]
    else:
        raise ValueError(f"Invalid mode: {mode}")


    extract_features.main(feature_conf, images_path, image_list=references, feature_path=features)
    # extract_features.main(feature_conf, images_path, feature_path=features)

    if use_retrieval:
        retrieval_conf = extract_features.confs['netvlad']
        global_descriptors = extract_features.main(retrieval_conf, images_path, out_path, image_list=references)
        pairs_from_retrieval.main(global_descriptors, sfm_pairs, num_matched=20)
    else:
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        # pairs_from_exhaustive.main(sfm_pairs, features=features)

    # match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    
    mapper_options = None
    image_options = None
    if fix_intrinsics:
        intrinsic_path = ref_path / 'cameras.txt'
        with open(intrinsic_path, 'r') as f:
            lines = f.readlines()
            intrinsics = lines[-1].strip().split(' ')
        intrinsics = str.join(' ', intrinsics[4:])  # fx, fy, cx, cy
        # intrinsics = "640 640 640 480"
        image_options = pycolmap.ImageReaderOptions()
        image_options.camera_model = 'PINHOLE'
        image_options.camera_params = intrinsics
        image_options.todict()

        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.ba_refine_focal_length = False
        mapper_options.ba_refine_principal_point = False
        mapper_options.ba_refine_extra_params = False
        mapper_options = mapper_options.todict()
   
    if mode == "triangulate":
        triangulation.main(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)
    else:
        reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)


def dense_SfM(images_path: Path, output_path: Path, ref_path: Path, mode="triangulate", fix_intrinsics=True):
    sfm_pairs = output_path / 'pairs-sfm.txt'
    sfm_dir = output_path / 'sfm'
    # features = output_path / 'features.h5'
    # matches = output_path / 'matches.h5'
    if output_path is None:
        output_path = ref_path / "../hloc_sfm_loftr"
    if not output_path.exists():
        output_path.mkdir(parents=True)

    matcher_conf = match_dense.confs['loftr']

    # references = [p.relative_to(images_path).as_posix() for p in (images_path / 'images').iterdir()]
    with open(ref_path / 'images.txt', 'r') as f:
            lines = f.readlines()
            # references = ["images/"+line.strip().split(' ')[-1] for line in lines if line.strip()]
            references = [line.strip().split(' ')[-1] for line in lines if line.strip()]

    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    features, matches = match_dense.main(matcher_conf, sfm_pairs, images_path, output_path, max_kps=8192, overwrite=False) 

    image_options = None
    mapper_options = None
    if fix_intrinsics:
        # intrinsic_path = images_path / 'intrinsics.txt'
        # with open(intrinsic_path, 'r') as f:
        #     lines = f.readlines()
        #     intrinsics = lines[-1].strip().split(' ')
        # intrinsics = str.join(', ', intrinsics)  # fx, fy, cx, cy
        # intrinsics = "640 640 640 480"
        intrinsic_path = ref_path / 'cameras.txt'
        with open(intrinsic_path, 'r') as f:
            lines = f.readlines()
            intrinsics = lines[-1].strip().split(' ')
        intrinsics = str.join(' ', intrinsics[4:])  # fx, fy, cx, cy

        image_options = pycolmap.ImageReaderOptions()
        image_options.camera_model = 'PINHOLE'
        image_options.camera_params = intrinsics
        image_options.todict()

        # mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.ba_refine_focal_length = False
        mapper_options.ba_refine_principal_point = False
        mapper_options.ba_refine_extra_params = False
        mapper_options = mapper_options.todict()

    # model = reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
    # triangulation.main(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)

    if mode == "triangulate":
        triangulation.main(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)
    else:
        # model = reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
        reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)

if __name__ == "__main__":
    # dataset_list = ['apartment_1_cam_arr1', 'apartment_1_cam_arr2', 'apartment_1_circle1', 'apartment_1_slam1', 'apartment_1_slam2', 'office_3_cam_arr1', 'office_3_cam_arr2']
    # dataset_list = ['apartment_0_slam1']
    # for dataset in dataset_list:
    #     print(f"== Start Triangulate {dataset} ==")
    #     images_path = Path("/home/keunmo/workspace/Hierarchical-Localization/outputs/replica_capture/" + dataset + "/images")
    #     ref_path = Path("/home/keunmo/workspace/Hierarchical-Localization/outputs/replica_capture/" + dataset + "/sparse/model")
    #     output_path = ref_path / "../hloc_triangulate_sift_nn"
    #     triangulate(ref_path, images_path, output_path, refine=True)
    #     # dense_sfm_refine_known_cam_poses(ref_path, images_path, output_path)
    #     # sfm_refine_known_cam_poses(ref_path, images_path, output_path)
    ref_path = Path("/home/keunmo/workspace/Hierarchical-Localization/dataset2/replica_capture/room_0_traj1_plus2/sparse/0")
    images_path = Path("/home/keunmo/workspace/Hierarchical-Localization/dataset2/replica_capture/room_0_traj1_plus2/images")
    out_path = Path("/home/keunmo/workspace/Hierarchical-Localization/dataset2/replica_capture/room_0_traj1_plus2/sparse/disk_lightglue")
    dense_out_path = Path("/home/keunmo/workspace/Hierarchical-Localization/dataset2/replica_capture/room_0_traj1_plus2/sparse/loftr")
    # SfM(images_path, out_path, ref_path, mode="triangulate", use_retrieval=False, refine=False)
    dense_SfM(images_path, dense_out_path, ref_path, mode="triangulate", refine=False)
