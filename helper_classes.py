import numpy as np

# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    normal_axis = normalize(axis)
    vector = normalize(vector)
    # Calculate the dot product of the vector and axis
    dot_product = np.dot(vector, normal_axis)

    # Calculate the reflected vector using the formula
    reflected_vector = vector - 2 * dot_product * normal_axis

    return reflected_vector


## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        v = self.get_light_ray(intersection).direction
        d = self.get_distance_from_light(intersection)
        return (self.intensity * np.dot(normalize(self.direction), v)) / (self.kc + self.kl*d + self.kq * (d**2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        #TODO
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction = None):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction

class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        cos = np.dot(self.normal, ray.direction)
        if cos != 0:
            t = (np.dot(v, self.normal) / cos)
        else :
            t=-1
        if t > 0:
            return t, self
        else:
            return None


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()

    def compute_normal(self):
        v1 = self.abcd[1] - self.abcd[0]
        v2 = self.abcd[2] - self.abcd[0]
        n = normalize(np.cross(v1, v2))
        return n

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        plane = Plane(self.normal, self.abcd[0])
        intersection = plane.intersect(ray)
        if not intersection:
            return None
        p = ray.origin + intersection[0] * ray.direction
        for i in range(4):
            p1 = self.abcd[i] - p
            p2 = self.abcd[(i+1) % 4] - p
            if np.dot(self.normal,np.cross(p1,p2)) <= 0:
                return None
        return intersection[0], self




class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """

        a,b,c,d,e,f = map(np.array,(a,b,c,d,e,f))
        da = a - d
        df = f - d
        g = d + da + df
        cb = b - c
        ce = e - c
        h = c + cb + ce

        A = Rectangle(a,b,c,d)
        B = Rectangle(d,c,e,f)
        C = Rectangle(f,e,h,g)
        D = Rectangle(g,h,b,a)
        E = Rectangle(g,a,d,f)
        F = Rectangle(b,h,e,c)
        self.face_list = [A,B,C,D,E,F]


    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        min_dist = np.inf
        min_rec = None
        for f in self.face_list:
            intersect = f.intersect(ray)
            if intersect:
                if intersect[0] < min_dist:
                    min_dist = intersect[0]
                    min_rec = f
        if min_rec:
            return min_dist, min_rec
        return None



class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        else:
            t1 = (-b + np.square(discriminant)) / (2 * a)
            t2 = (-b - np.square(discriminant)) / (2 * a)
            if t1 < 0 and t2 < 0:
                return None
            else:
                t = min(t1, t2)
                return t, self 
    
    def compute_normal(self,p):
        return normalize((p - self.center))
