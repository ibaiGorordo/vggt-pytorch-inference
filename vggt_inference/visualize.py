import rerun as rr
import numpy as np
import rerun.blueprint as rrb

from .vggt_inference import InferenceResult

def draw_pose(transform: np.ndarray, name: str, static: bool = False):
    rr.log(name,
        rr.Arrows3D(origins=[0,0,0], vectors= [[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    radii=[0.001, 0.001, 0.001]
        ),
        static=static,
    )

    rr.log(name,
        rr.Transform3D(
            translation=transform[:3, 3],
            mat3x3=transform[:3, :3],
        ),
        static=static,
    )

def visualize_results(results: list[InferenceResult], filter_percent: float = 50):
    rr.init("vggt_inference", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
            rrb.Spatial3DView(origin="body/pose", contents="body/**"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="body/cam/image"),
                    rrb.Spatial2DView(origin="body/cam/depth_map"),
                )
            )
        )
    )

    for i, result in enumerate(results):
        rr.set_time_sequence("frame", i)
        world_to_camera  = np.eye(4)
        world_to_camera[:3, :4] = result.extrinsic
        camera_to_world = np.linalg.inv(world_to_camera)

        rr.log(f"body/cam",
            rr.Pinhole(
                    image_from_camera=result.intrinsic,
                    width=result.width,
                    height=result.height,
                    image_plane_distance=0.02,
            )
        )

        rr.log(f"body/cam", rr.Transform3D(
            translation=camera_to_world[:3, 3],
            mat3x3=camera_to_world[:3, :3],
        ))

        draw_pose(camera_to_world, f"body/pose{i}", static=True)
        draw_pose(camera_to_world, f"body/pose")

        # Filter points based on confidence threshold
        conf = result.depth_conf
        conf_thres = np.percentile(conf, filter_percent)
        keep_mask = conf > conf_thres
        depth_maps = result.depth_map
        depth_maps[~keep_mask] = 0

        points = result.point_map_by_unprojection.reshape(-1, 3)
        keep_mask = keep_mask.reshape(-1)
        colors = result.image.reshape(-1, 3)

        points = points[keep_mask]
        colors = colors[keep_mask]

        rr.log(f"body/points{i}", rr.Points3D(points, colors=colors, radii=0.0003), static=True)
        rr.log(f"body/cam/image", rr.Image(result.image.astype(np.uint8)))
        rr.log(f"body/cam/depth_map", rr.DepthImage(depth_maps))
