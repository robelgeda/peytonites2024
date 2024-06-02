from astropy import units as u

def kpc_to_cm(value):
    return (value * u.kpc).to(u.cm).value

def cm_to_kpc(value):
    return (value * u.cm).to(u.kpc).value

def lyr_to_cm(value):
    return (value * u.lyr).to(u.cm).value

def cm_to_lyr(value):
    return (value * u.cm).to(u.lyr).value

def au_to_cm(value):
    return (value * u.AU).to(u.cm).value

def cm_to_au(value):
    return (value * u.cm).to(u.AU).value