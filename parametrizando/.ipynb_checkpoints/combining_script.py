import vuggyParamScript as vg


def combining_script(sims_dir,dir_sph,rock_type,sim_fraction,SMALL_DELTA, D0, GAMMA):
    
    sim_data = vg.sim_data(sims_dir, rock_type)

    sphere_data = vg.sphere_data(dir_sph, rock_type)

    combined_data = vg.data_combining(sim_data, sphere_data, sim_fraction)

    dt = vg.compute_dT(combined_data, SMALL_DELTA)
