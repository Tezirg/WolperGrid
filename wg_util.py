import warnings
import numpy as np
import math
import tensorflow as tf

def limit_gpu_usage():
    WARN_GPU_CONF = "Cannot configure tensorflow GPU memory growth property"
    try:
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) > 0:
            tf.config.experimental.set_memory_growth(devices[0], True)
    except AttributeError:
        try:
            devices = tf.config.experimental.list_physical_devices('GPU')
            if len(devices) > 0:
                tf.config.experimental.set_memory_growth(devices[0], True)
        except Exception:
            warnings.warn(WARN_GPU_CONF)
    except Exception:
        warnings.warn(WARN_GPU_CONF)

def wg_size_obs(observation_space):
    dims = np.array([
        # Time
        5, # Timestamp
        4 * observation_space.n_line, # lines cd
        observation_space.n_sub, # Sub cd
        # Gen
        observation_space.n_gen * 6,
        # Load
        observation_space.n_load * 4,
        # Line origins
        observation_space.n_line * 4,
        # Line extremities
        observation_space.n_line * 5
    ])
    return np.sum(dims)

def to_norm_vect(inputv, pad_v = 0.0, scale_v = 1.0):
    v = np.asarray(inputv)
    v = v / scale_v
    vsafe = np.nan_to_num(v, nan=pad_v, posinf=pad_v, neginf=pad_v)
    return vsafe.astype(np.float32)

def wg_convert_obs(obs, bias=0.0):
    # Store some shortcuts
    topo = obs.topo_vect
    g_pos = obs.gen_pos_topo_vect
    l_pos = obs.load_pos_topo_vect
    lor_pos = obs.line_or_pos_topo_vect
    lex_pos = obs.line_ex_pos_topo_vect

    # Get time data
    time_li = [obs.month / 12.0, obs.day / 31.0, obs.day_of_week / 7.0,
               obs.hour_of_day / 24.0, obs.minute_of_hour / 60.0]
    time_v = to_norm_vect(time_li)
    time_line_cd = to_norm_vect(obs.time_before_cooldown_line,
                                pad_v=-1.0, scale_v=20.0)
    time_line_nm = to_norm_vect(obs.time_next_maintenance, scale_v=12.0*31.0*24.0)
    time_line_dm = to_norm_vect(obs.duration_next_maintenance, scale_v=3*12.0*24.0)
    time_line_overflow = to_norm_vect(obs.timestep_overflow, scale_v=2.0)
    time_sub_cd = to_norm_vect(obs.time_before_cooldown_sub,
                               pad_v=-1.0, scale_v=20.0)
    
    # Get generators info
    g_p = to_norm_vect(obs.prod_p, scale_v=1000.0)
    g_q = to_norm_vect(obs.prod_q, scale_v=1000.0)
    g_v = to_norm_vect(obs.prod_v, scale_v=1000.0)
    g_tr = to_norm_vect(obs.target_dispatch, scale_v=1000.0)
    g_ar = to_norm_vect(obs.actual_dispatch, scale_v=1000.0)
    g_cost = to_norm_vect(obs.gen_cost_per_MW, pad_v=0.0, scale_v=1.0)
    g_buses = np.zeros(obs.n_gen)
    for gen_id in range(obs.n_gen):
        g_buses[gen_id] = topo[g_pos[gen_id]] * 1.0
        if g_buses[gen_id] <= 0.0:
            g_buses[gen_id] = 0.0
    g_bus = to_norm_vect(g_buses, pad_v=-1.0, scale_v=3.0)

    # Get loads info
    l_p = to_norm_vect(obs.load_p, scale_v=1000.0)
    l_q = to_norm_vect(obs.load_q, scale_v=1000.0)
    l_v = to_norm_vect(obs.load_v, scale_v=1000.0)
    l_buses = np.zeros(obs.n_load)
    for load_id in range(obs.n_load):
        l_buses[load_id] = topo[l_pos[load_id]] * 1.0
        if l_buses[load_id] <= 0.0:
            l_buses[load_id] = 0.0
    l_bus = to_norm_vect(l_buses, pad_v=-1.0, scale_v=3.0)

    # Get lines origin info
    or_p = to_norm_vect(obs.p_or, scale_v=1000.0)
    or_q = to_norm_vect(obs.q_or, scale_v=1000.0)
    or_v = to_norm_vect(obs.v_or, scale_v=1000.0)
    or_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        or_buses[line_id] = topo[lor_pos[line_id]] * 1.0
        if or_buses[line_id] <= 0.0:
            or_buses[line_id] = 0.0
    or_bus = to_norm_vect(or_buses, pad_v=-1.0, scale_v=3.0)
    or_rho = to_norm_vect(obs.rho, pad_v=-1.0)
    
    # Get extremities origin info
    ex_p = to_norm_vect(obs.p_ex, scale_v=1000.0)
    ex_q = to_norm_vect(obs.q_ex, scale_v=1000.0)
    ex_v = to_norm_vect(obs.v_ex, scale_v=1000.0)
    ex_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        ex_buses[line_id] = topo[lex_pos[line_id]] * 1.0
        if ex_buses[line_id] <= 0.0:
            ex_buses[line_id] = 0.0
    ex_bus = to_norm_vect(ex_buses, pad_v=-1.0, scale_v=3.0)
    ex_rho = to_norm_vect(obs.rho, pad_v=-1.0)

    res = np.concatenate([
        # Time
        time_v,
        time_line_cd,
        time_line_nm, time_line_dm,
        time_line_overflow,
        time_sub_cd,
        # Gens
        g_p, g_q, g_v, g_ar, g_tr, g_bus,
        # Loads
        l_p, l_q, l_v, l_bus,
        # Origins
        or_p, or_q, or_v, or_bus,
        # Extremities
        ex_p, ex_q, ex_v, ex_bus, ex_rho
    ])
    return res + bias

def disp_nn_to_act(obs, net_disp):
    # [-1.0;1.0] -> [-ramp_down;+ramp_up]
    act_redispatch = np.zeros(obs.n_gen)
    for i, d in enumerate(net_disp):
        if math.isclose(d, 0.0): # Skip if 0.0
            continue
        rmin = obs.gen_max_ramp_down[i]
        rmax = obs.gen_max_ramp_up[i]
        r = np.interp(d, [-1.0, 1.0], [-rmin, rmax])
        act_redispatch[i] = r
    return act_redispatch

def disp_act_to_nn(obs, act_redispatch):
    # [-ramp_down;+ramp_up] -> [-1.0;1.0]
    netdisp = np.zeros(obs.n_gen)
    for i, d in enumerate(act_redispatch):
        rmin = obs.gen_max_ramp_down[i]
        rmax = obs.gen_max_ramp_up[i]
        r = np.interp(d, [-rmin, rmax], [-1.0, 1.0])
        netdisp[i] = r

    return netdisp

def unitary_acts_to_impact_tree(action_space):
    tree = [
        [ [] for _ in range(action_space.n_line)],
        [ [] for _ in range(action_space.n_sub)],
        [ [] for _ in range(action_space.n_gen)]
    ]
    # Process impact for each actions
    # Skip do nothing at 0
    for act_id, a in enumerate(action_space.all_actions[1:]):
        impact = a.impact_on_objects()

        # Find category & index
        ## Lines set
        if np.any(a._set_line_status != 0):
            c = 0
            i = np.where(a._set_line_status != 0)[0][0]
        ## Lines change
        elif np.any(a._switch_line_status == True):
            c = 0
            i = np.where(a._switch_line_status == True)[0][0]
        ## Topo set
        elif np.any(a._set_topo_vect != 0):
            c = 1
            obj = np.where(a._set_topo_vect != 0)[0][0]
            _, _, i = a._obj_caract_from_topo_id(obj)
        ## Topo change
        elif np.any(a._change_bus_vect):
            c = 1
            obj = np.where(a._change_bus_vect == True)[0][0]
            _, _, i = a._obj_caract_from_topo_id(obj)
        ## Gen redispatch
        elif np.any(a._redispatch != 0.0):
            c = 2
            i = np.where(a._redispatch != 0.0)[0][0]

        # Register id into tree
        tree[c][i].append(act_id + 1)

    # Add do nothing in each category
    #for i in range(3):
    #    tree[i].append([0,0])
    return tree

def print_impact_tree(tree):
    print("Lines impact:")
    for l_id, l_acts in enumerate(tree[0]):
        print("Line {}: {} actions".format(l_id, len(l_acts)))
    print("Subs impact:")
    for s_id, s_acts in enumerate(tree[1]):
        print("Sub {}: {} actions".format(s_id, len(s_acts)))
    print("Gens impact:")
    for g_id, g_acts in enumerate(tree[2]):
        print("Gen {}: {} actions".format(g_id, len(g_acts)))

def draw_from_tree(tree, category_prob):
    c = np.random.choice(3, p=category_prob)
    i = np.random.choice(len(tree[c]))
    while len(tree[c][i]) == 0:
        i = np.random.choice(len(tree[c]))
    a = np.random.choice(len(tree[c][i]))
    return tree[c][i][a]
