import embree
import numpy as np


from abc import ABC

use1M = True

def get_centroids(V, F):
    return V[F].mean(axis=1)


def get_cross_products(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C


def get_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    A = C_norms/2
    return A


def get_surface_normals(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    return N


def get_surface_normals_and_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A


class ShapeModel(ABC):
    pass


class TrimeshShapeModel(ShapeModel):
    """A shape model consisting of a single triangle mesh."""

    def __init__(self, V, F, N=None, P=None, A=None):
        """Initialize a triangle mesh shape model. No assumption is made about
        the way vertices or faces are stored when building the shape
        model except that V[F] yields the faces of the mesh. Vertices
        may be repeated or not.

        Parameters
        ----------
        V : array_like
            An array with shape (num_verts, 3) whose rows correspond to the
            vertices of the triangle mesh
        F : array_like
            An array with shape (num_faces, 3) whose rows index the faces
            of the triangle mesh (i.e., V[F] returns an array with shape
            (num_faces, 3, 3) such that V[F][i] is a 3x3 matrix whose rows
            are the vertices of the ith face.
        N : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            mesh face normals. Can be passed to specify the face normals.
            Otherwise, the face normals will be computed from the cross products
            of the face edges (i.e. np.cross(vi1 - vi0, vi2 - vi0) normalized).
        P : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            centroids. Can be optionally passed to avoid recomputing.
        A : array_like, optional
            An array of shape (num_faces,) containing the triangle areas. Can
            be optionally passed to avoid recomputing.

        """

        self.dtype = V.dtype

        self.V = V
        self.F = F

        if N is None and A is None:
            N, A = get_surface_normals_and_face_areas(V, F)
        elif A is None:
            if N.shape[0] != F.shape[0]:
                raise Exception(
                    'must pass same number of surface normals as faces (got ' +
                    '%d faces and %d normals' % (F.shape[0], N.shape[0])
                )
            A = get_face_areas(V, F)
        elif N is None:
            N = get_surface_normals(V, F)

        self.P = get_centroids(V, F)
        self.N = N
        self.A = A

        assert self.P.dtype == self.dtype
        assert self.N.dtype == self.dtype
        assert self.A.dtype == self.dtype

        self._make_scene()

    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.

        '''
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        scene = device.make_scene()
        scene.set_flags(embree.SceneFlags.ROBUST)
        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            self.V.shape[0], # item_count
        )
        vertex_buffer[:] = self.V[:]
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            self.F.shape[0]
        )
        index_buffer[:] = self.F[:]
        geometry.commit()
        scene.attach_geometry(geometry)
        geometry.release()
        scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)
        self.scene = scene

    def __reduce__(self):
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    def __repr__(self):
        return 'a TrimeshShapeModel with %d vertices and %d faces' % (
            self.num_verts, self.num_faces)

    @property
    def num_faces(self):
        return self.P.shape[0]

    @property
    def num_verts(self):
        return self.V.shape[0]

    def get_visibility(self, I, J, eps=None, oriented=False):
        '''Compute the visibility mask for pairs of indices (i, j) taken from
        index arrays I and J. If m = len(I) and N = len(J), the
        resulting array is an m x N binary matrix V, where V[i, j] ==
        1 if a ray traced from the centroid of facet i to the centroid
        of facet j is unoccluded.

        The parameter eps is used to perturb the start of each ray
        away from the facet centroid. This is because Embree (by
        default) doesn't know to check for self-intersection. A
        "filter function" should be set up to support this, but this
        hasn't been implemented. For now, we use the eps parameter,
        which is a bit of a hack.

        If oriented is True, then this will use the surface normal to
        check whether both triangles have the correct orientation (the
        normals and the vector pointing from the centroid of one
        triangle to the other have a positive dot product---i.e., the
        triangles are facing one another).

        '''
        # TODO: make it possible to have eps automatically set by
        # triangle area (but better to use a filter function, as
        # described above...)
        if eps is None:
            eps = 1e3*np.finfo(np.float32).resolution

        m, n = len(I), len(J)

        PJ = self.P[J]

        D = np.empty((m*n, 3), dtype=self.dtype)
        for q, i in enumerate(I):
            D[q*n:(q + 1)*n] = PJ - self.P[i]
        D_norm_sq = np.sqrt(np.sum(D**2, axis=1))
        mask = D_norm_sq > eps
        D = D[mask]/D_norm_sq[mask].reshape(-1, 1)

        P = np.empty((m*n, 3), dtype=self.dtype)
        for q, i in enumerate(I):
            P[q*n:(q + 1)*n] = self.P[i]

        J_extended = np.empty((m*n,), dtype=J.dtype)
        for q in range(m):
            J_extended[q*n:(q + 1)*n] = J
        J_extended = J_extended[mask]

        num_masked = mask.sum()

        rayhit = embree.RayHit1M(num_masked)

        context = embree.IntersectContext()
        context.flags = embree.IntersectContextFlags.COHERENT

        rayhit.org[:] = P[mask] + eps*D
        rayhit.dir[:] = D
        rayhit.tnear[:] = 0
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        if use1M:
            self.scene.intersect1M(context, rayhit)
        else:
            self.scene.intersectNp(context, rayhit)

        vis = np.ones((m*n,), dtype=bool) # vis by default
        vis[mask] = np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == J_extended
        )

        # set vis for any pairs of faces with unoccluded LOS which
        # aren't oriented towards each other to False
        if oriented:
            for q, i in enumerate(I):
                vis[q*n:(q + 1)*n] *= (PJ - self.P[i])@self.N[i] > 0

        # faces can't see themselves
        for q, i in enumerate(I):
            vis[q*n:(q + 1)*n][J == i] = False

        return vis.reshape(m, n)

    def get_visibility_1_to_N(self, i, J, eps=None, oriented=False):
        '''Convenience function for calling get_visibility with a single
        source triangle.

        '''
        return self.get_visibility([i], J, eps, oriented).ravel()

    def get_visibility_matrix(self, eps=None, oriented=False):
        '''Convenience function for computing the visibility matrix. This just
        calls get_visibility(I, I, eps, oriented), where I =
        np.arange(num_faces).

        '''
        I = np.arange(self.num_faces)
        return self.get_visibility(I, I, eps, oriented)

    def get_direct_irradiance(self, F0, Dsun, unit_Svec=False, basemesh=None, eps=None):
        '''Compute the insolation from the sun.

        Parameters
        ----------
        F0: float
            The solar constant. [W/m^2]

        Dsun: numpy.ndarray
            An length 3 vector or Mx3 array of sun directions: vectors
            indicating the direction of the sun in world coordinates.

        basemesh: same as self, optional
            mesh used to check (Sun, light source) visibility at "self.cells";
            it would usually cover a larger area than "self".

        eps: float
            How far to perturb the start of the ray away from each
            face. Default is 1e3*np.finfo(np.float32).resolution. This
            is to overcome precision issues with Embree.

        unit_Svec: bool
            defines if Dsun is a unit vector (Sun direction) or
            the actual Sun-origin vector (check AU units below)

        Returns
        -------
        E: numpy.ndarray
            A vector of length self.num_faces or an array of size
            M x self.num_faces, where M is the number of sun
            directions.

        '''
        if eps is None:
            eps = 1e3*np.finfo(np.float32).resolution

        if basemesh == None:
            basemesh = self

        # Here, we use Embree directly to find the indices of triangles
        # which are directly illuminated (I_sun) or not (I_shadow).

        n = self.num_faces

        if Dsun.ndim == 1:
            # Normalize Dsun
            distSunkm = np.sqrt(np.sum(Dsun ** 2))
            Dsun /= distSunkm

            ray = embree.Ray1M(n)
            if eps.ndim==0:
                ray.org[:] = self.P + eps*self.N
            else:
                ray.org[:] = self.P + eps[:,np.newaxis]*self.N
            ray.dir[:] = Dsun
            ray.tnear[:] = 0
            ray.tfar[:] = np.inf
            ray.flags[:] = 0
        elif Dsun.ndim == 2:
            # Normalize Dsun
            distSunkm = np.linalg.norm(Dsun,axis=1)[:,np.newaxis]
            Dsun /= distSunkm

            m = Dsun.size//3
            ray = embree.Ray1M(m*n)
            for i in range(m):
                ray.org[i*n:(i + 1)*n, :] = self.P + eps[:,np.newaxis]*self.N
            for i, d in enumerate(Dsun):
                ray.dir[i*n:(i + 1)*n, :] = d
            ray.tnear[:] = 0
            ray.tfar[:] = np.inf
            ray.flags[:] = 0

        context = embree.IntersectContext()
        basemesh.scene.occluded1M(context, ray)
        # Determine which rays escaped (i.e., can see the sun)
        I = np.isposinf(ray.tfar)

        # rescale solar flux depending on distance
        if not unit_Svec:
            AU_km = 149597900.
            F0 *= (AU_km / distSunkm) ** 2

        # Compute the direct irradiance
        if Dsun.ndim == 1:
            E = np.zeros(n, dtype=self.dtype)
            E[I] = F0*np.maximum(0, self.N[I]@Dsun)
        else:
            E = np.zeros((n, m), dtype=self.dtype)
            I = I.reshape(m, n)
            # TODO check if this can be vectorized
            for i, d in enumerate(Dsun):
                if unit_Svec:
                    E[I[i], i] = F0*np.maximum(0, self.N[I[i]]@d)
                else:
                    E[I[i], i] = F0[i]*np.maximum(0, self.N[I[i]]@d)
        return E

    def get_pyvista_unstructured_grid(self):
        try:
            import pyvista as pv
        except:
            raise ImportError('failed to import PyVista')

        try:
            import vtk as vtk
        except:
            raise ImportError('failed to import vtk')

        return pv.UnstructuredGrid({vtk.VTK_TRIANGLE: self.F}, self.V)
