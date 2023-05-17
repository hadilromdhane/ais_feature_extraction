from __future__ import division
from utm.error import OutOfRangeError
import math
import numpy as np


# For most use cases in this module, numpy is indistinguishable
# from math, except it also works on numpy arrays
try:
    import numpy as mathlib
    use_numpy = True
except ImportError:
    import math as mathlib
    use_numpy = False

__all__ = ['to_latlon', 'from_latlon']

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1 - E)

SQRT_E = mathlib.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3 / 2 * _E - 27 / 32 * _E3 + 269 / 512 * _E5)
P3 = (21 / 16 * _E2 - 55 / 32 * _E4)
P4 = (151 / 96 * _E3 - 417 / 128 * _E5)
P5 = (1097 / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


######################################################################
def in_bounds(x, lower, upper, upper_strict=False):
    if upper_strict and use_numpy:
        return lower <= mathlib.min(x) and mathlib.max(x) < upper
    elif upper_strict and not use_numpy:
        return lower <= x < upper
    elif use_numpy:
        return lower <= mathlib.min(x) and mathlib.max(x) <= upper
    return lower <= x <= upper


######################################################################

def check_valid_zone(zone_number, zone_letter):
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError(
            'zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError(
                'zone letter out of range (must be between C and X)')

######################################################################


def mixed_signs(x):
    return use_numpy and mathlib.min(x) < 0 and mathlib.max(x) >= 0

######################################################################


def negative(x):
    if use_numpy:
        return mathlib.max(x) < 0
    return x < 0
######################################################################


def mod_angle(value):
    """Returns angle in radians to be between -pi and pi"""
    return (value + mathlib.pi) % (2 * mathlib.pi) - mathlib.pi
######################################################################


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, strict=True):
    """This function converts UTM coordinates to Latitude and Longitude

        Parameters
        ----------
        easting: int or NumPy array
            Easting value of UTM coordinates

        northing: int or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented with global map numbers of a UTM zone
            numbers map. For more information see utmzones [1]_

        zone_letter: str
            Zone letter can be represented as string values.  UTM zone
            designators can be seen in [1]_

        northern: bool
            You can set True or False to set this parameter. Default is None

        strict: bool
            Raise an OutOfRangeError if outside of bounds

        Returns
        -------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).


       .. _[1]: http://www.jaworski.ca/utmzones.htm

    """
    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if strict:
        if not in_bounds(easting, 100000, 1000000, upper_strict=True):
            raise OutOfRangeError(
                'easting out of range (must be between 100,000 m and 999,999 m)')
        if not in_bounds(northing, 0, 10000000):
            raise OutOfRangeError(
                'northing out of range (must be between 0 m and 10,000,000 m)')

    check_valid_zone(zone_number, zone_letter)

    if zone_letter:
        zone_letter = zone_letter.upper()
        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * mathlib.sin(2 * mu) +
             P3 * mathlib.sin(4 * mu) +
             P4 * mathlib.sin(6 * mu) +
             P5 * mathlib.sin(8 * mu))

    p_sin = mathlib.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = mathlib.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = mathlib.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = E_P2 * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    longitude = mod_angle(
        longitude + mathlib.radians(zone_number_to_central_longitude(zone_number)))

    return (mathlib.degrees(latitude),
            mathlib.degrees(longitude))

######################################################################


def dtr(angle):
    """Takes angle in degree and transforms it to radiant."""
    return angle * math.pi / 180

######################################################################


def head_inter(head_OS, head_TS, to_2pi=True):
    """Computes the intersection angle between headings in radiant (in [0, 2pi) if to_2pi, else [-pi, pi)).
    Corresponds to C_T in Xu et al. (2022, Neurocomputing)."""
    if to_2pi:
        return angle_to_2pi(head_TS - head_OS)
    else:
        return angle_to_pi(head_TS - head_OS)

######################################################################


def rtd(angle):
    """Takes angle in radiant and transforms it to degree."""
    return angle * 180 / math.pi

######################################################################


def from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None):
    """This function converts Latitude and Longitude to UTM coordinate

        Parameters
        ----------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).

        force_zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. You may force conversion to be included within one
            UTM zone number.  For more information see utmzones [1]_

        force_zone_letter: str
            You may force conversion to be included within one UTM zone
            letter.  For more information see utmzones [1]_

        Returns
        -------
        easting: float or NumPy array
            Easting value of UTM coordinates

        northing: float or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented by global map numbers of a UTM zone
            numbers map. More information see utmzones [1]_

        zone_letter: str
            Zone letter is represented by a string value. UTM zone designators
            can be accessed in [1]_


       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    if not in_bounds(latitude, -80, 84):
        raise OutOfRangeError(
            'latitude out of range (must be between 80 deg S and 84 deg N)')
    if not in_bounds(longitude, -180, 180):
        raise OutOfRangeError(
            'longitude out of range (must be between 180 deg W and 180 deg E)')
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = mathlib.radians(latitude)
    lat_sin = mathlib.sin(lat_rad)
    lat_cos = mathlib.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    if force_zone_letter is None:
        zone_letter = latitude_to_zone_letter(latitude)
    else:
        zone_letter = force_zone_letter

    lon_rad = mathlib.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = mathlib.radians(central_lon)

    n = R / mathlib.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * mathlib.sin(2 * lat_rad) +
             M3 * mathlib.sin(4 * lat_rad) -
             M4 * mathlib.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if mixed_signs(latitude):
        raise ValueError("latitudes must all have the same sign")
    elif negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter

######################################################################


def latitude_to_zone_letter(latitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if use_numpy and isinstance(latitude, mathlib.ndarray):
        latitude = latitude.flat[0]

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None

######################################################################


def latlon_to_zone_number(latitude, longitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if use_numpy:
        if isinstance(latitude, mathlib.ndarray):
            latitude = latitude.flat[0]
        if isinstance(longitude, mathlib.ndarray):
            longitude = longitude.flat[0]

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1

######################################################################


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3

######################################################################


def meter_to_NM(meter):
    """Convertes meter in nautical miles."""
    return meter / 1852
######################################################################


def NM_to_meter(NM):
    """Convertes nautical miles in meter."""
    return NM * 1852

######################################################################


def angle_to_2pi(angle):
    """Transforms an angle to [0, 2pi)."""
    if angle >= 0:
        return angle - math.floor(angle / (2*math.pi)) * 2*math.pi
    else:
        return angle + (math.floor(-angle / (2*math.pi)) + 1) * 2*math.pi


######################################################################


def angle_to_pi(angle):
    """Transforms an angle to [-pi, pi)."""
    if angle >= 0:
        return angle - math.floor((angle + math.pi) / (2*math.pi)) * 2*math.pi
    else:
        return angle + math.floor((-angle + math.pi) / (2*math.pi)) * 2*math.pi

######################################################################


def polar_from_xy(x, y, with_r=True, with_angle=True):
    """Get polar coordinates (r, angle in rad in [0, 2pi)) from x,y-coordinates. Angles are defined clockwise with zero at the y-axis.
    Args:
        with_r (bool):     Whether to compute the radius.
        with_angle (bool): Whether to compute the angle.
    Returns:
        r, angle as a tuple of floats."""

    r = math.sqrt(x**2 + y**2) if with_r else None
    angle = angle_to_2pi(math.atan2(x, y)) if with_angle else None
    return r, angle


######################################################################


def bng_abs(N0, E0, N1, E1):
    """Computes the absolute bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0)."""
    return polar_from_xy(x=E1-E0, y=N1-N0, with_r=False, with_angle=True)[1]

######################################################################


def bng_rel(N0, E0, N1, E1, head0, to_2pi=True):
    """Computes the relative bearing (in radiant in [0, 2pi) if to_2pi, else [-pi, pi)) of 
    (N1, E1) from perspective of (N0, E0) and heading head0."""
    if to_2pi:
        return angle_to_2pi(bng_abs(N0, E0, N1, E1) - head0)
    else:
        return angle_to_pi(bng_abs(N0, E0, N1, E1) - head0)

######################################################################


def ED(N0, E0, N1, E1, sqrt=True):
    """Computes the euclidean distance between two points."""
    d_sq = (N0 - N1)**2 + (E0 - E1)**2

    if sqrt:
        return np.sqrt(d_sq)
    return d_sq

######################################################################


def xy_from_polar(r, angle):
    """Get x,y-coordinates from polar system, where angle is defined clockwise with zero at the y-axis."""
    return r * np.sin(angle), r * np.cos(angle)
######################################################################


def tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes the time to closest point of approach (TCPA). Follows Lenart (1983)."""

    # easy access
    xOS = EOS
    yOS = NOS
    xTS = ETS
    yTS = NTS

    # compute velocities in x,y-coordinates
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)

    # relative velocity
    vrx = vxTS - vxOS
    vry = vyTS - vyOS

    # tcpa
    nom = (xTS - xOS)*vrx + (yTS - yOS)*vry
    den = vrx**2 + vry**2

    if den == 0:
        return 0.0
    return - nom / den

######################################################################


def cpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS, get_positions=False):
    """Returns DCPA and TCPA. Follows Chun et al. (2021, OE).
    If get_positions, returns DCPA, TCPA, and NOS, EOS, NTS, ETS when TCPA = 0."""

    # easy access
    xOS = EOS
    yOS = NOS
    xTS = ETS
    yTS = NTS

    # compute velocities in x,y-coordinates
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)

    # get TCPA
    TCPA = tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS)

    # forecast OS
    xOS_tcpa = xOS + TCPA * vxOS
    yOS_tcpa = yOS + TCPA * vyOS

    # forecast TS
    xTS_tcpa = xTS + TCPA * vxTS
    yTS_tcpa = yTS + TCPA * vyTS

    if get_positions:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA, yOS_tcpa, xOS_tcpa, yTS_tcpa, xTS_tcpa
    else:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA

    # final_df.loc[i,'DCPA_TS_OS'],final_df.loc[i,'TCPA_TS_OS']=  cpa(N0,E0,N1,E1,course0,course1,speed0,speed1)

######################################################################
