from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            intersection = find_intersection(ray, objects)
            if intersection[0]:
                color = get_color(ambient,lights,objects,ray,intersection[1],intersection[0],max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects


def find_intersection(ray, objects):
    min_dist = np.inf
    closest_object = None
    closest_intersection = None
    for obj in objects:
        intersection = obj.intersect(ray)
        if intersection:
            t = intersection[0]
            if t < min_dist:
                min_dist = t
                closest_object = intersection[1]
                closest_intersection = ray.origin + (t - 0.25) * ray.direction

    return closest_object, closest_intersection

def get_color(ambient,lights,objects,ray,intersection_point,intersection_object,max_depth):
    # ambient light
    color = ambient * intersection_object.ambient
    color = color.astype(np.float64)

    #diffuse and specular lights
    for light in lights:
        l_ray = light.get_light_ray(intersection_point)
        shadow = find_intersection(l_ray, objects)


        #with shadows:
        if not shadow[0]:
            color += diffuseColor(objects,light,ray,intersection_point,intersection_object)
            color += specularColor(objects, light, ray,intersection_point,intersection_object)
        else:
            intersections_distances = np.linalg.norm(intersection_point - shadow[1])
            if intersections_distances > light.get_distance_from_light(intersection_point):
                color += diffuseColor(objects, light, ray, intersection_point, intersection_object)
                color += specularColor(objects, light, ray, intersection_point, intersection_object)



    max_depth -= 1
    if max_depth == 0:
        return color

    #reflection and refraction light
    r_ray = reflectiveRay(ray,objects,intersection_point,intersection_object)
    r_hit = find_intersection(r_ray, objects)
    if r_hit[0]:
        color += intersection_object.reflection * get_color(ambient,lights,objects,r_ray,r_hit[1],r_hit[0],max_depth)

    if intersection_object.refraction:
        refRay = refractedRay(ray,intersection_object,intersection_point,True)
        refraction_hit = find_intersection(refRay, objects)
        if refraction_hit[0]:
            color +=   0.5 * get_color(ambient,lights,objects, refRay,refraction_hit[1],refraction_hit[0],max_depth)


    return color

def diffuseColor(objects,light,ray,intersection_point,intersection_object):
    intensity = light.get_intensity(intersection_point)
    kd = intersection_object.diffuse
    if isinstance(intersection_object, (Plane,Rectangle)):
        n = intersection_object.normal
    elif isinstance(intersection_object, Sphere):
        n = intersection_object.compute_normal(intersection_point)
    l = light.get_light_ray(intersection_point)

    diffuse = kd * intensity * np.dot(normalize(n), l.direction)

    return diffuse



def specularColor(objects,light,ray,intersection_point,intersection_object):
    intensity = light.get_intensity(intersection_point)
    ks = intersection_object.specular
    n = intersection_object.shininess
    
    l_ray = light.get_light_ray(intersection_point)

    if isinstance(intersection_object, (Plane,Rectangle)):
        normal = intersection_object.normal
    elif isinstance(intersection_object, Sphere):
        normal = intersection_object.compute_normal(intersection_point)

    r = normalize(reflected(-l_ray.direction, normal))
    normalized_v = normalize(-ray.direction)
    specular = ks * intensity * np.power(abs(np.dot(normalized_v, r)) , n/10)
    return specular

#calculate reflected ray
def reflectiveRay(ray,objects,intersection_point,intersection_object):
    if isinstance(intersection_object, (Plane, Rectangle)):
        normal = intersection_object.normal
    elif isinstance(intersection_object, Sphere):
        normal = intersection_object.compute_normal(intersection_point)
    return Ray(intersection_point,reflected(ray.direction, normal))



#create our scene
def your_own_scene():
    
    Pearth = Sphere([1.4, 0, -1],0.4)
    Pearth.set_material([0, 1, 1], [0.03, 0.01, 0.0001], [0.1, 0.3, 0.3], 100, 0.2)
    
#     Pmars = Sphere([1, -0.4, -1],0.23)
#     Pmars.set_material([1, 0.003, 0.004], [1, 0.003, 0.04], [1, 0.03, 0], 100, 1)
    
    Pmercury = Sphere([-0.8, 1.5, -1],0.2)
    Pmercury.set_material([1, 0.003, 0.004], [1, 0.03, 0.04], [1, 0.03, 0], 100, 0.5)
    
    Pvenus = Sphere([0.2, 1, -1],0.3)
    Pvenus.set_material([1, 0.003, 0.004], [1, 0.03, 0.04], [1, 0.03, 0], 100, 0.5)
    
#     plane = Plane([0,1,0], [0,-0.3,0])
#     plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 100, 0.5)

    background = Plane([0,0,1], [0,0,-3])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 100, 0.03)


    objects = [Pearth,Pvenus,Pmercury,background]

    light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([-1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)
    lightsun = PointLight(intensity= np.array([1, 1, 1]),position=np.array([-2,-1,-2]),kc=0.1,kl=0.1,kq=0.1)

    sun = SpotLight(intensity= np.array([255, 255, 0]),position=np.array([-2,-1,-2]), direction=([0,0,1]),
                    kc=0.1,kl=0.1,kq=0.1)
    lights = [light,sun,lightsun]

    ambient = np.array([0.6,0.4,0.5])

    camera = np.array([0,0,0.5])

    return camera, lights, objects, ambient

#calculate the refracted ray
def refractedRay(ray, intersection_object, intersection_point, in_object):
    if in_object:
        n1 = 1.0  
        n2 = intersection_object.refraction  
    else :
        n1 = intersection_object.refraction  
        n2 = 1.0


    incident_direction = ray.direction
    if isinstance(intersection_object, (Plane, Rectangle)):
        normal = intersection_object.normal
    elif isinstance(intersection_object, Sphere):
        normal = intersection_object.compute_normal(intersection_point)

    # Ensure that the normal is pointing outwards
    if np.dot(incident_direction, normal) > 0:
        normal = -normal
        n1, n2 = n2, n1

    incident_direction = normalize(incident_direction) 

    n_ratio = n1 / n2
    cos_i = -np.dot(incident_direction, normal)
    sin_t2 = n_ratio**2 * (1 - cos_i**2)

    # Check for total internal reflection
    if sin_t2 > 1.0:
        return None

    cos_t = np.sqrt(1.0 - sin_t2)

    refracted_direction = n_ratio * incident_direction + (n_ratio * cos_i - cos_t) * normal
    refracted_direction = refracted_direction / np.linalg.norm(refracted_direction)

    refracted_ray = Ray(intersection_point+ (0.3 *refracted_direction) , refracted_direction)
    return refracted_ray


## create the scene with refracted object
def refracted_scene():
    background = Plane([0,0,1], [0,0,-7])
    background.set_material([1,1,0], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)

    plane = Plane([0,1,0], [0,-0.3,0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)

    rectangle = Rectangle([-1,1,-2],[-1,0,-2],[1,0,-2],[1,1,-2])
    rectangle.set_material([1,0,0], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5, 1.1)


    sphere= Sphere(np.array([0,1.7,-5]), 0.7)
    sphere.set_material([0.5,0.6,0], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)

    objects = [rectangle,background, plane, sphere ]

    light = SpotLight(intensity= np.array([1,1,1]),position=np.array([0,0,1.5]), direction=([0,0,1]),
                        kc=0.1,kl=0.1,kq=0.1)
    lights= [light]
    camera = [0,0,1]
    ambient = np.array([1,1,1])
    return camera, lights, objects, ambient