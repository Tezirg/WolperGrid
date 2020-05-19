import numpy as np
import math

def size_obs(observation_space):
    dims = np.array([
        # Time
        5, # Timestamp
        2 * observation_space.n_line,
        observation_space.n_sub,
        # Gen
        observation_space.n_gen * 7,
        # Load
        observation_space.n_load * 4,
        # Line origins
        observation_space.n_line * 5,
        # Line extremities
        observation_space.n_line * 5
    ])
    return np.sum(dims)

def to_norm_vect(inputv, pad_v = 0.0, scale_v = 1.0):
    v = np.asarray(inputv)
    v = v / scale_v
    vsafe = np.nan_to_num(v, nan=pad_v, posinf=pad_v, neginf=pad_v)
    return vsafe.astype(np.float32)

def analog_convert_obs(obs, bias=0.0):
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
                                pad_v=-1.0, scale_v=10.0)
    time_line_nm = to_norm_vect(obs.time_next_maintenance, scale_v=10.0)
    time_sub_cd = to_norm_vect(obs.time_before_cooldown_sub,
                               pad_v=-1.0, scale_v=10.0)
    
    # Get generators info
    g_p = to_norm_vect(obs.prod_p, scale_v=1000.0)
    g_q = to_norm_vect(obs.prod_q, scale_v=1000.0)
    g_v = to_norm_vect(obs.prod_v, scale_v=1000.0)
    g_tr = to_norm_vect(obs.target_dispatch, scale_v=150.0)
    g_ar = to_norm_vect(obs.actual_dispatch, scale_v=150.0)
    g_cost = to_norm_vect(obs.gen_cost_per_MW, pad_v=0.0, scale_v=1.0)
    g_buses = np.zeros(obs.n_gen)
    for gen_id in range(obs.n_gen):
        g_buses[gen_id] = topo[g_pos[gen_id]]
        if g_buses[gen_id] <= 0.0:
            g_buses[gen_id] = 0.0
    g_bus = to_norm_vect(g_buses, pad_v=-1.0, scale_v=3.0)

    # Get loads info
    l_p = to_norm_vect(obs.load_p, scale_v=1000.0)
    l_q = to_norm_vect(obs.load_q, scale_v=1000.0)
    l_v = to_norm_vect(obs.load_v, scale_v=1000.0)
    l_buses = np.zeros(obs.n_load)
    for load_id in range(obs.n_load):
        l_buses[load_id] = topo[l_pos[load_id]]
        if l_buses[load_id] <= 0.0:
            l_buses[load_id] = 0.0
    l_bus = to_norm_vect(l_buses, pad_v=-1.0, scale_v=3.0)

    # Get lines origin info
    or_p = to_norm_vect(obs.p_or, scale_v=1000.0)
    or_q = to_norm_vect(obs.q_or, scale_v=1000.0)
    or_v = to_norm_vect(obs.v_or, scale_v=1000.0)
    or_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        or_buses[line_id] = topo[lor_pos[line_id]]
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
        ex_buses[line_id] = topo[lex_pos[line_id]]
        if ex_buses[line_id] <= 0.0:
            ex_buses[line_id] = 0.0
    ex_bus = to_norm_vect(ex_buses, pad_v=-1.0, scale_v=3.0)
    ex_rho = to_norm_vect(obs.rho, pad_v=-1.0)

    res = np.concatenate([
        # Time
        time_v, time_line_cd, time_sub_cd, time_line_nm,
        # Gens
        g_p, g_q, g_v, g_ar, g_tr, g_bus, g_cost,
        # Loads
        l_p, l_q, l_v, l_bus,
        # Origins
        or_p, or_q, or_v, or_bus, or_rho,
        # Extremities
        ex_p, ex_q, ex_v, ex_bus, ex_rho
    ])
    return res + bias

def netbus_rnd(obs, n_bus=2):
    # Copy obs state
    rnd_topo = np.zeros((n_bus, obs.dim_topo))
    rnd_topo[0][obs.topo_vect == 1] = 1.0
    rnd_topo[1][obs.topo_vect == 2] = 1.0
    # Pick a random substation
    rnd_sub = np.random.randint(obs.n_sub)
    n_elem = obs.sub_info[rnd_sub]
    # Pick a random number of elements to change
    rnd_n_changes = np.random.randint(n_elem+1)
    # Pick the elements to change at random
    rnd_sub_elems = np.random.randint(0, n_elem, rnd_n_changes)
    # Set the topo vect
    sub_topo_pos = np.sum(obs.sub_info[0:rnd_sub])
    for elem_pos in rnd_sub_elems:
        rnd_bus = np.random.randint(n_bus)
        rnd_topo[rnd_bus][sub_topo_pos + elem_pos] = 1.0
        # Set the other buses to 0.0
        for b in range(n_bus):
            if b == rnd_bus:
                continue;
            rnd_topo[b][sub_topo_pos + elem_pos] = 0.0

    return rnd_topo

def netbus_to_act_setbus(obs, net_bus):
    # n_bus x dim_topo x p([0.0; 1.0]) ->
    # -> dim_topo x [0 unchanged; 1: bus_1; 2 bus_2 ]
    # Pick the buses
    act_setbus = np.argmax(net_bus, axis=0) + 1
    # Don't set disconnected elements
    act_setbus[obs.topo_vect <= 0] = 0
    # Don't set elements already on the correct bus
    act_setbus[act_setbus == obs.topo_vect] = 0

    return act_setbus

def act_setbus_to_netbus(obs, act_setbus, n_bus=2):
    # [0 unchanged; 1: bus_1; 2 bus_2 ] ->
    # n_bus x dim_topo x p([0.0; 1.0])
    net_bus = np.zeros((n_bus, obs.dim_topo))
    # Get buses from obs
    for i in range(n_bus):
        net_bus[i][obs.topo_vect == (i + 1)] = 1.0
    # Override with buses from action
    for i in range(n_bus):
        net_bus[i][act_setbus == (i + 1)] = 1.0
    
    return net_bus

def netline_rnd(obs):
    rnd_lines = obs.line_status.astype(np.float32)
    rnd_lineid = np.random.randint(obs.n_line)
    rnd_linestatus = not obs.line_status[rnd_lineid]
    rnd_lines[rnd_lineid] = np.int32(rnd_linestatus)

    return rnd_lines
            
def netline_to_act_setstatus(obs, net_line):
    # [0.0 Disconnect; > 0.0 Connect] ->
    # -> [0.0 Unchanged; -1.0 Disconnect; 1.0 Connect]
    act_setstatus = np.copy(net_line)
    act_setstatus[net_line <= 0.0] = -1
    act_setstatus[net_line > 0.0] = 1
    # Do no 'set' already connected lines
    act_setstatus[obs.line_status == (act_setstatus == 1)] = 0
    # Do not 'set' already disconnected lines
    act_setstatus[(obs.line_status == False) == (act_setstatus == -1)] = 0
    return act_setstatus

def act_setstatus_to_netline(obs, setstatus):
    # [0.0 Unchanged; -1.0 Disconnect; 1.0 Connect] ->
    # -> [0.0 Disconnect; > 0.0 Connect]
    netline = np.zeros(obs.n_line)
    # Copy obs status
    netline[obs.line_status == True] = 1.0
    # Override with action changes
    netline[(obs.line_status == False) == (setstatus == 1)] = 1.0
    netline[(obs.line_status == True) == (setstatus == -1)] = 0.0
    return netline

def netdisp_rnd(obs):
    disp_rnd = np.zeros(obs.n_gen)
    # Take random gen to disp
    rnd_gen = np.random.randint(obs.n_gen)
    # Take a random disp 
    rnd_ramp = np.random.uniform(-1.0, 1.0)
    disp_rnd[rnd_gen] = rnd_ramp

    return disp_rnd

def netdisp_to_act_redispatch(obs, net_disp):
    # [-1.0;1.0] -> [-ramp_down;+ramp_up]
    act_redispatch = np.zeros(obs.n_gen)
    for i, d in enumerate(net_disp):
        if math.isclose(d, 0.0): # Skip if 0.0
            continue
        rmin = obs.gen_max_ramp_down[i]
        rmax = obs.gen_max_ramp_up[i]
        r = np.interp(d, [-1.0, 1.0], [-rmin, rmax])
        act_redispatch[i] = round(r) # Round at 1MW
    return act_redispatch

def act_redispatch_to_netdisp(obs, act_redispatch):
    # [-ramp_down;+ramp_up] -> [-1.0;1.0]
    netdisp = np.zeros(obs.n_gen)
    for i, d in enumerate(act_redispatch):
        rmin = obs.gen_max_ramp_down[i]
        rmax = obs.gen_max_ramp_up[i]
        r = np.interp(d, [-rmin, rmax], [-1.0, 1.0])
        netdisp[i] = r

    return netdisp
    
def analog_convert_act(action_space, obs, netbus, netline, netdisp):
    act_setbus = netbus_to_act_setbus(obs, netbus)
    act_setstatus = netline_to_act_setstatus(obs, netline)
    act_redispatch = netdisp_to_act_redispatch(obs, netdisp)
    act = action_space({
        'set_bus': act_setbus,
        'set_line_status': act_setstatus,
        'redispatch': act_redispatch
    })
    return act
