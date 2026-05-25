import warp as wp

from .dataclasses import *
from .scene import *
from .presampler_functions import *


def terrain_presampler(const: RaytracerConstantsWP, scene:SceneWP):

    """
    Returns points, directions and weights for rays sampled on promising terrain cells and the receiver elements.
    """
    n_cells = (scene.Nx-1) * (scene.Ny-1)
    n_cell_rays = n_cells * const.n_rays_per_cell

    device = scene.normals.device
    dirs = wp.zeros(n_cell_rays, dtype=wp.vec3, device=device)

    # 1. Sample directions from TX to terrain. Every single cell gets const.n_rays_per_cell samples.
    wp.launch(
        kernel=uniform_k_cells_dir_sampling,
        dim=n_cell_rays,
        inputs= [const, scene, dirs],
        device=device,
    )

    scores = wp.zeros(n_cell_rays, dtype=wp.float32, device=device)

    # 2. Perform raytracing for every direction. Assign scores to every direction based on the clostest proximity to RX at any point of the path.
    wp.launch(
        kernel=terrain_presampler_kernel,
        dim=n_cell_rays,
        inputs= [const, scene, dirs, scores],
        device=device,
    )

    best = wp.zeros(n_cells)

    # 3. Find the best score for each cell.
    wp.launch(
        kernel=reduce_score_per_cell,
        dim=n_cells,
        inputs= [scores, best, const.n_rays_per_cell],
        device=device,
    )

    accepted_ids = wp.zeros(n_cells, dtype=wp.int32, device=device)
    accepted_count = wp.zeros(1, dtype=wp.int32, device=device)
    threshold = const.mp_threshold_dist # Tuneable proximity to the receiver at any point on the raypath.

    # 4. Choose cells with score below the threshold
    wp.launch(
        kernel=build_accepted_cells,
        dim=n_cells,
        inputs= [best, accepted_ids, accepted_count, threshold],
        device=device,
    )

    count = int(accepted_count.numpy()[0])
    accepted_ids = accepted_ids[:count]

    # 4.5 Find solid angles of the accepted cells and receivers for Monte-Carlo weights.
    solid_angles_mp = wp.zeros(count, dtype=float, device=device)
    solid_angles_rx = wp.zeros(scene.rx_n_rx, dtype=float, device=device)

    wp.launch(
        kernel=solid_angles_of_accepted_cells,
        dim=count + scene.rx_n_rx,
        inputs= [scene, accepted_ids, solid_angles_mp, solid_angles_rx],
        device=device,
    )

    n_rays = const.n_rays_mp + (const.n_rays_per_rx * scene.rx_n_rx)
    out_pts = wp.zeros(n_rays, dtype=wp.vec3, device=device)
    out_dirs = wp.zeros(n_rays, dtype=wp.vec3, device=device)
    out_weights = wp.zeros(n_rays, dtype=wp.float32, device=device)

    # 5. Uniformely sample the accepted cells const.n_rays times, and form directions from TX to sample and weight..
    wp.launch(
        kernel=sample_dirs_from_accepted_cells,
        dim=n_rays,
        inputs= [const, scene, accepted_ids, accepted_count, solid_angles_rx, solid_angles_mp, out_pts, out_dirs, out_weights],
        device=device,
    )


    if False:
        out_weightsn = out_weights.numpy()

        print("")
        print("--- Presampler stats ---")

        print("accepted cell count:", count)
        print("mean rx weight", out_weightsn[:const.n_rays_per_rx * scene.rx_n_rx].mean())
        print("mean rx solid angle",solid_angles_rx.numpy().mean())

        print("Number of mp rays", n_rays - (const.n_rays_per_rx * scene.rx_n_rx))
        print("Total accepted cell weight", out_weightsn[const.n_rays_per_rx * scene.rx_n_rx:].sum())
        print("mean accepted cell weight", out_weightsn[const.n_rays_per_rx * scene.rx_n_rx:].mean())
        print("mean accepted cell solid angle",solid_angles_mp.numpy().mean())

        print("weight ratio mp/rx", out_weightsn[const.n_rays_per_rx * scene.rx_n_rx:].sum()/ out_weightsn[:const.n_rays_per_rx * scene.rx_n_rx].sum())

        print("sum accepted solid angle =", solid_angles_mp.numpy().sum())
        print("sum rx solid angle =", solid_angles_rx.numpy().sum())
        print("solid angle ratio =", solid_angles_mp.numpy().sum() / solid_angles_rx.numpy().sum())


    wp.synchronize()
    return out_pts, out_dirs, out_weights
