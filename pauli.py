def sigma_identity(state, site):
    return state**0, state

def sigma_x(state, site):
    return state**0, state^(1<<site)

def sigma_y(state, site):
    return ((-1.0)**((state >> site)&1))*1.0j, state^(1<<site)

def sigma_z(state, site):
    return (-1.0)**((state >> site)&1), state

def sigma_i(i, state, site):
    return (sigma_identity, sigma_x, sigma_y, sigma_z)[i](state, site)
