#version 330

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float4x4 mat4
#define float3x3 mat3

in float2 fragmentTexCoord;

layout(location = 0) out vec4 fragColor;

uniform int g_screenWidth;
uniform int g_screenHeight;

uniform bool fog;
uniform bool shadows;
uniform bool ambient_acclusion;
uniform bool soft_shadows;

const float3 center = float3(0,0,0);
const float3 radius = float3(1,0,0);

const int MAX_MARCHING_STEPS = 1000;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;
const vec3  fogColor  = vec3(0.5,0.6,0.7);

uniform float4x4 g_rayMatrix;

uniform float4 g_bgColor = float4(0,0.3,0.5,0.5);

float3 EyeRayDir(float x, float y, float w, float h)
{
    float fov = 3.141592654f/(2.0f);
    float3 ray_dir;

	ray_dir.x = x+0.5f - (w/2.0f);
	ray_dir.y = y+0.5f - (h/2.0f);
	ray_dir.z = -(w)/tan(fov/2.0f);

    return normalize(ray_dir);
}

vec2 Sphere(vec3 p)
{
    return vec2(length(p - vec3(-0.75,0.5,5)) - 0.4, 1.0);
}

vec2 Box(vec3 p)
{
    return vec2(length(max(abs(p - vec3(-1.5,0.5,5)) - 0.3,0.0)), 3.0);
}

vec2 Torus(vec3 p)
{
    p = p - vec3(1, 0.5, 6);
    vec2 t = vec2(0.6,0.1);
    return vec2(length(vec2(length(p.xz)-t.x,p.y))-t.y, 4.0);
}

vec2 Surface( vec3 p)
{
    return vec2(p.y, 2.0);
}

vec2 Ellipsoid(vec3 p)
{
    p = p - vec3(1,0.9,6);
    vec3 r = vec3(0.2, 0.25, 0.05);
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    float dist = k0*(k0-1.0)/k1;
    return vec2(dist, 1.0);
}

vec2 Octahedron(vec3 p)
{
    float s = 0.35;
    p = abs(p- vec3(0, 0.6, 5));
    float m = p.x+p.y+p.z-s;
    vec3 q;
    if( 3.0*p.x < m )
        q = p.xyz;
    else if( 3.0*p.y < m )
        q = p.yzx;
    else if( 3.0*p.z < m )
        q = p.zxy;
    else
        return vec2(m*0.57735027, 5.0);
    float k = clamp(0.5*(q.z-q.y+s),0.0,s);
    return vec2(length(vec3(q.x,q.y-s+k,q.z-k)), 5.0);
}

vec2 Scene(vec3 p)
{
    float min1 = min(Sphere(p).x, Box(p).x);
    float min2 = min(Sphere(p).x, Torus(p).x);
    float min3 = min(min1, min2);
    float min4 = min(Surface(p).x, min3);
    float min5 = min(min4, Ellipsoid(p).x);
    float min = min(min5, Octahedron(p).x);
    if (min == Sphere(p).x) {
        return Sphere(p);
    } else if (min == Box(p).x) {
        return Box(p);
    } else if (min == Torus(p).x) {
        return Torus(p);
    } else if (min == Surface(p).x) {
        return Surface(p);
    } else if (min == Ellipsoid(p).x) {
        return Ellipsoid(p);
    } else {
        return Octahedron(p);
    }
}

struct Light
{
    vec3 pos;
    vec3 intensity;
};

Light lights[4];

void LightInit()
{
    lights[0].pos = vec3(0, 6, 10);
    lights[0].intensity = vec3(0.3, 0.3, 0.3);

    lights[1].pos = vec3(3, 6, 6);
    lights[1].intensity = vec3(0.3, 0.3, 0.3);

    lights[2].pos = vec3(-3, 6, 6);
    lights[2].intensity = vec3(0.3, 0.3, 0.3);

}

struct Material
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

Material emerald, bronze, pearl, cyan_plastic, ruby;

void MaterialInit()
{
    emerald.ambient = vec3(0.0215, 0.1745, 0.0215);
    emerald.diffuse = vec3(0.07568, 0.61424, 0.07568);
    emerald.specular = vec3(0.633, 0.727811, 0.633);
    emerald.shininess = 0.6;

    bronze.ambient = vec3(0.2125, 0.1275, 0.054);
    bronze.diffuse =vec3(0.714, 0.4284, 0.18144);
    bronze.specular = vec3(0.393548, 0.271906, 0.166721);
    bronze.shininess = 0.2;

    pearl.ambient = vec3(0.25, 0.20725, 0.20725);
    pearl.diffuse = vec3(1, 0.829, 0.829);
    pearl.specular = vec3(0.296648, 0.296648, 0.296648);
    pearl.shininess =0.088;

    cyan_plastic.ambient = vec3(0.0, 0.1, 0.06);
    cyan_plastic.diffuse = vec3(0.0, 0.50980392, 0.50980392);
    cyan_plastic.specular = vec3(0.50196078, 0.50196078, 0.50196078);
    cyan_plastic.shininess = 0.25;

    ruby.ambient = vec3(0.1745, 0.01175, 0.01175);
    ruby.diffuse = vec3(0.61424, 0.04136, 0.04136);
    ruby.specular = vec3(0.727811, 0.626959, 0.626959);
    ruby.shininess = 0.6;

}

vec3 EstimateNormal(float3 z, float eps)
{
    float3 z1 = z + float3(eps, 0, 0);
    float3 z2 = z - float3(eps, 0, 0);
    float3 z3 = z + float3(0, eps, 0);
    float3 z4 = z - float3(0, eps, 0);
    float3 z5 = z + float3(0, 0, eps);
    float3 z6 = z - float3(0, 0, eps);
    float dx = Scene(z1).x - Scene(z2).x;
    float dy = Scene(z3).x - Scene(z4).x;
    float dz = Scene(z5).x - Scene(z6).x;
    return normalize(float3(dx, dy, dz) / (2.0*eps));
}

float Shadow(vec3 ray_pos, vec3 light_pos, float k)
{
    float len = length(light_pos - ray_pos);
    light_pos = normalize(light_pos - ray_pos);
    float res = 1.0;
    for(float i = EPSILON; i <= len; i += dist)
    {
        float dist = Scene(ray_pos + light_pos*i).x;
        if(dist < EPSILON/100)
            return 0.0;
        if (soft_shadows)
        {
            res = min(res, k*dist/i);
        }
    }
    return res;
}
vec2 IsLightVisible(vec3 ray_pos, vec3 light_pos, vec3 ray)
{
    vec3 N = EstimateNormal(ray_pos, EPSILON/100);
    vec3 L = normalize(light_pos - ray_pos);
    vec3 V = normalize(ray - ray_pos);
    vec3 R = normalize(reflect(-L, N));

    return vec2(dot(L, N), dot(R, V));
}

vec3 PhongDiffuseSpecular(vec3 diffuse, vec3 specular, float shininess, vec3 ray_pos, vec3 ray, vec3 light_pos, vec3 lightIntensity)
{
    vec2 light = IsLightVisible(ray_pos, light_pos, ray);
    if (light[0] < 0.0)
    {
        return vec3(0.0, 0.0, 0.0);
    } else if (light[1] < 0.0) {
        return lightIntensity * (diffuse * light[0]) + 0.01;
    } else {
        return lightIntensity * diffuse * light[0] + lightIntensity * specular * pow(light[1], shininess) +
        lightIntensity * specular * pow(light[1], 5);
    }
}

vec3 Phong(vec3 ambient, vec3 diffuse, vec3 specular, float shininess, vec3 ray, vec3 ray_pos) {

    vec3 colour = 0.3 * vec3(1.0, 1.0, 1.0) * ambient;

    if (ambient_acclusion)
    {
        vec3 norm = EstimateNormal(ray_pos, EPSILON/100);
        float step = 0.0001;
        float oc = 0.0f;
      	for(int i = 0; i < 15; ++i, step += 0.0001, oc += step - dist)
        {
       	    float dist = Scene(ray_pos + norm * step).x;
        }
        colour = colour * (clamp(oc, 0, 1));
    }

    if (shadows || soft_shadows)
    {
        float k = 1.0;
        if (soft_shadows)
        {
            k = 8.0;
        }
        for (int i = 0; i < 3; i++)
        {
            colour += PhongDiffuseSpecular(diffuse, specular, shininess, ray, ray_pos, lights[i].pos, lights[i].intensity)
            *Shadow(ray, lights[i].pos, k)+0.01;
        }
    } else {
        for (int i = 0; i < 3; i++)
        {
            colour += PhongDiffuseSpecular(diffuse, specular, shininess, ray, ray_pos, lights[i].pos, lights[i].intensity);

        }
    }

    return colour;
}

vec3 ApplyFog(vec3  rgb, float dist)
{
    return mix(rgb, fogColor, 1.0 - exp(-dist*0.03));
}

vec4 RayMarching(vec3 ray_pos, vec3 ray_dir) {
    float depth = MIN_DIST;
    vec3 ray = ray_pos;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        vec2 dist = Scene(ray);
        if (dist.x < EPSILON) {
            Material current;
            if (dist.y == 1) {
                current = emerald;
            } else if(dist.y == 2) {
                current = bronze;
            } else if (dist.y == 3) {
                current = pearl;
            } else if (dist.y == 4) {
                current = cyan_plastic;
            } else if (dist.y == 5) {
                current = ruby;
            }
            vec3 colour = vec3(Phong(current.ambient, current.diffuse, current.specular, current.shininess, ray, ray_pos));
            if (fog)
            {
                colour = ApplyFog(colour, length(ray_pos - ray));
            }
            return vec4(colour, 1.0);
        }
        if (depth >= MAX_DIST) {
             if (fog)
             {
                vec3 colour = ApplyFog(vec3(0,0,1), length(ray_pos - ray));
                return vec4(colour,0);
             }
             return g_bgColor;
        }
        depth += dist.x;
        ray += dist.x * ray_dir;
    }
    if (fog)
    {
        vec3 colour = ApplyFog(vec3(0,0,1), length(ray_pos - ray));
        return vec4(colour,0);
    }
    return g_bgColor;
}

void main(void)
{
    float w = float(g_screenWidth);
    float h = float(g_screenHeight);

    // get curr pixelcoordinates
    float x = fragmentTexCoord.x*w;
    float y = fragmentTexCoord.y*h;

    // generate initial ray
    float3 ray_pos = float3(0,0,0);
    float3 ray_dir = EyeRayDir(x,y,w,h);

    // transorm ray with matrix
    ray_pos = (g_rayMatrix*float4(ray_pos,1)).xyz;
    ray_dir = float3x3(g_rayMatrix)*ray_dir;

    MaterialInit();
    LightInit();
    fragColor = RayMarching(ray_pos, ray_dir);
    return;
}