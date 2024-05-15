+++++++++++++++++++++++++++++++++++++++++++++
Distance Measuring & Geodetic Transformations
+++++++++++++++++++++++++++++++++++++++++++++

Distances on the Earth
======================

To a first approximation, the Earth can be modelled as a sphere with a radius of 6371 km. This
means that for points on the surface of the Earth at significant separation the
surface distance between them (the great-circle distance) is uniquely defined by the angle between them at the centre of the 
Earth, and can be calculated using the Haversine formula [1]_,

.. math::

    d = 2r\arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right)+\cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)

where :math:`\phi` and :math:`\lambda` are latitude and longitude respectively,
and :math:`r` is the radius of the Earth.

Where the points are very close together, the Haversine formula as given above is not
accurate for numerical calculations. In this case, the distance can be approximated by the Pythagorean
theorem, using the latitude and longitude as Cartesian coordinates. This is
only valid for small distances, but is much faster to calculate.

.. math:: d = r\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}

For intermediate distances, we can use
an iterative method such as the Vincenty formula, which takes into account the squashing of the Earth towards the poles
However, especially where more than two points are involved,
we may be better off using a projection method to transform the local band of latitude and longitude into a flat
2D surface.

Leveraging Projections
======================

Although the surface of the Earth is curved, for many purposes we can treat it
as locally flat. This is the basis of most map projections, which transform
latitude and longitude coordinates on the surface of the Earth to a flat
two-dimensional Cartesian coordinate system. This has many advantages for
concepts such as calculating distances, applying clustering algorithms and
visualising data, including in maps.

The Ordnance Survey National Grid
---------------------------------

For historical reasons, multiple coordinate systems exist in British mapping.
The Ordnance Survey has been mapping the British Isles since the 18th Century
and the last major retriangulation from 1936-1962 produced the Ordance Survey
National Grid (or **OSGB36**), which defined latitude and longitude across the
island of Great Britain [2]_. For convenience, a standard Transverse Mercator
projection [3]_ was also defined, producing a notionally flat gridded surface,
with gradations called eastings and westings. The scale for these gradations
was identified with metres. To a good approximation, the Pythagorean theorem
can be used to calculate distances on this grid.

.. math:: d = \sqrt{(E_1-E_2)^2+(N_1-N_2)^2}



The OSGB36 datum is based on the Airy Ellipsoid of 1830, which defines
semimajor axes for its model of the earth, :math:`a` and :math:`b`, a scaling
factor :math:`F_0` and ellipsoid height, :math:`H`.

.. math::
    a &= 6377563.396, \\
    b &= 6356256.910, \\
    F_0 &= 0.9996012717, \\
    H &= 24.7.

The point of origin for the transverse Mercator projection is defined in the
Ordnance Survey longitude-latitude and easting-northing coordinates as

.. math::
    \phi^{OS}_0 &= 49^\circ \mbox{ north}, \\
    \lambda^{OS}_0 &= 2^\circ \mbox{ west}, \\
    E^{OS}_0 &= 400000 m, \\
    N^{OS}_0 &= -100000 m.

GPS and the WGS84 Datum
-----------------------

More recently, the world has gravitated towards the use of the satellite based
Global Positioning Systems (GPS) for navigation and location. This
equipment, uses the (globally more appropriate) World Geodetic System
1984 (or **WGS84**). This datum uses a different ellipsoid, which offers a
better fit for a global coordinate system. Its key properties are:

.. math::
    a_{WGS} &= 6378137,, \\
    b_{WGS} &= 6356752.314, \\
    F_0 &= 0.9996.

For a given point on the WGS84 ellipsoid, an approximate mapping to the
OSGB36 datum can be found using a Helmert transformation [4]_,

.. math::
    \mathbf{x}^{OS} = \mathbf{t}+\mathbf{M}\mathbf{x}^{WGS}.


Here :math:`\mathbf{x}` denotes a coordinate in Cartesian space (i.e in 3D)
as given by the (invertible) transformation

.. math::
    \nu &= \frac{aF_0}{\sqrt{1-e^2\sin^2(\phi^{OS})}} \\
    x &= (\nu+H) \sin(\lambda)\cos(\phi) \\
    y &= (\nu+H) \cos(\lambda)\cos(\phi) \\
    z &= ((1-e^2)\nu+H)\sin(\phi)

and the transformation parameters are

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{t} &= \left(\begin{array}{c}
    -446.448\\ 125.157\\ -542.060
    \end{array}\right),\\
    \mathbf{M} &= \left[\begin{array}{ c c c }
    1+s& -r_3& r_2\\
    r_3 & 1+s & -r_1 \\
    -r_2 & r_1 & 1+s
    \end{array}\right], \\
    s &= 20.4894\times 10^{-6}, \\
    \mathbf{r} &= [0.1502'', 0.2470'', 0.8421''].
    \end{eqnarray*}

Given a latitude, :math:`\phi^{OS}` and longitude, :math:`\lambda^{OS}` in the
OSGB36 datum, easting and northing coordinates, :math:`E^{OS}` & :math:`N^{OS}`
can then be calculated using the following formulae:

.. math::
    \rho &= \frac{aF_0(1-e^2)}{\left(1-e^2\sin^2(\phi^{OS})\right)^{\frac{3}{2}}} \\
    \eta &= \sqrt{\frac{\nu}{\rho}-1} \\
    M &= bF_0\left[\left(1+n+\frac{5}{4}n^2+\frac{5}{4}n^3\right)(\phi^{OS}-\phi^{OS}_0)\right. \\
    &\quad-\left(3n+3n^2+\frac{21}{8}n^3\right)\sin(\phi-\phi_0)\cos(\phi^{OS}+\phi^{OS}_0) \\
    &\quad+\left(\frac{15}{8}n^2+\frac{15}{8}n^3\right)\sin(2(\phi^{OS}-\phi^{OS}_0))\cos(2(\phi^{OS}+\phi^{OS}_0)) \\
    &\left.\quad-\frac{35}{24}n^3\sin(3(\phi-\phi_0))\cos(3(\phi^{OS}+\phi^{OS}_0))\right] \\
    I &= M + N^{OS}_0 \\
    II &= \frac{\nu}{2}\sin(\phi^{OS})\cos(\phi^{OS}) \\
    III &= \frac{\nu}{24}\sin(\phi^{OS})cos^3(\phi^{OS})(5-\tan^2(phi^{OS})+9\eta^2) \\
    IIIA &= \frac{\nu}{720}\sin(\phi^{OS})cos^5(\phi^{OS})(61-58\tan^2(\phi^{OS})+\tan^4(\phi^{OS})) \\
    IV &= \nu\cos(\phi^{OS}) \\
    V &= \frac{\nu}{6}\cos^3(\phi^{OS})\left(\frac{\nu}{\rho}-\tan^2(\phi^{OS})\right) \\
    VI &= \frac{\nu}{120}\cos^5(\phi^{OS})(5-18\tan^2(\phi^{OS})+\tan^4(\phi^{OS}) \\
    &\quad+14\eta^2-58\tan^2(\phi^{OS})\eta^2) \\
    E^{OS} &= E^{OS}_0+IV(\lambda^{OS}-\lambda^{OS}_0)+V(\lambda-\lambda^{OS}_0)^3+VI(\lambda^{OS}-\lambda^{OS}_0)^5 \\
    N^{OS} &= I + II(\lambda^{OS}-\lambda^{OS}_0)^2+III(\lambda-\lambda^{OS}_0)^4+IIIA(\lambda^{OS}-\lambda^{OS}_0)^6



.. rubric:: References

.. [1] The Haversine formula https://en.wikipedia.org/wiki/Haversine_formula
.. [2] A guide to coordinate systems in Great Britain, Ordnance Survey
.. [3] Map projections - A Working Manual, John P. Snyder, https://doi.org/10.3133/pp1395
.. [4] Computing Helmert transformations, G Watson, http://www.maths.dundee.ac.uk/gawatson/helmertrev.pdf
