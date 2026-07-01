// Reference path tracer. Inline ray tracing (RayQuery) over the scene TLAS,
// a metallic-roughness BRDF (Lambert diffuse + GGX specular), next-event
// estimation for the punctual lights, and the environment map as the miss
// radiance. Results accumulate progressively into an HDR buffer.

static const float PI = 3.14159265358979;

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct PTMaterial {
    float4 base_color;
    float metallic;
    float roughness;
    uint has_base_tex;
    uint has_mr_tex;
    float3 emissive;
    uint has_emissive_tex;
    Texture2D<float4>.Handle base_tex;
    Texture2D<float4>.Handle mr_tex;
    Texture2D<float4>.Handle emissive_tex;
    SamplerState.Handle samp;
};

struct DirLight {
    float3 direction;
    float intensity;
    float3 color;
    float pad;
};

struct PointLight {
    float3 position;
    float range;
    float3 color;
    float intensity;
};

RaytracingAccelerationStructure g_tlas;
StructuredBuffer<Vertex> g_vertices;
StructuredBuffer<uint> g_indices;
StructuredBuffer<uint> g_tri_material;
StructuredBuffer<PTMaterial> g_materials;
StructuredBuffer<DirLight> g_dir_lights;
StructuredBuffer<PointLight> g_point_lights;

Texture2D<float4> g_env;
SamplerState g_env_sampler;

RWTexture2D<float4> g_accum;
RWTexture2D<float4> g_output;

uniform float4x4 g_inv_view_proj;
uniform float3 g_eye;
uniform uint g_frame;
uniform uint g_reset;
uniform uint g_bounces;
uniform uint g_num_dir;
uniform uint g_num_point;
uniform float g_exposure;
uniform uint g_use_env;
uniform uint g_alpha_cutout;

// ---- rng ------------------------------------------------------------------

uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randf(inout uint s)
{
    return (pcg(s) & 0xffffffu) / float(0x1000000);
}

// ---- sampling helpers -----------------------------------------------------

void onb(float3 n, out float3 t, out float3 b)
{
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float d = n.x * n.y * a;
    t = float3(1.0 + sign * n.x * n.x * a, sign * d, -sign * n.x);
    b = float3(d, sign + n.y * n.y * a, -n.y);
}

float3 to_world(float3 v, float3 n)
{
    float3 t, b;
    onb(n, t, b);
    return v.x * t + v.y * b + v.z * n;
}

float3 cosine_sample(float3 n, float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float3 h = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    return normalize(to_world(h, n));
}

float3 ggx_sample_h(float3 n, float alpha, float2 u)
{
    float phi = 2.0 * PI * u.x;
    float ct = sqrt((1.0 - u.y) / (1.0 + (alpha * alpha - 1.0) * u.y));
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float3 h = float3(st * cos(phi), st * sin(phi), ct);
    return normalize(to_world(h, n));
}

// ---- microfacet terms -----------------------------------------------------

float d_ggx(float ndh, float alpha)
{
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-8);
}

float g1(float ndx, float alpha)
{
    float a2 = alpha * alpha;
    return 2.0 * ndx / max(ndx + sqrt(a2 + (1.0 - a2) * ndx * ndx), 1e-8);
}

float g_smith(float ndv, float ndl, float alpha)
{
    return g1(ndv, alpha) * g1(ndl, alpha);
}

float3 fresnel(float vdh, float3 f0)
{
    float m = clamp(1.0 - vdh, 0.0, 1.0);
    float m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// ---- environment ----------------------------------------------------------

float3 env_radiance(float3 d)
{
    if (g_use_env == 0)
        return float3(0.0);
    float u = atan2(d.x, d.z) / (2.0 * PI) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return g_env.SampleLevel(g_env_sampler, float2(u, v), 0).rgb;
}

// ---- ray tracing pipeline -------------------------------------------------

// Ray payload: a surface hit for the primary ray, or a visibility flag for
// shadow rays (hit = 1 means occluded; the miss shader clears it to 0).
struct HitInfo {
    float3 pos;
    float3 normal;
    float2 uv;
    uint material;
    uint hit;
};

// Alpha cutout: geometry is built non-opaque so the any-hit shader runs; a
// candidate hit is only accepted when the base-colour alpha clears the mask.
bool alpha_passes(uint prim, float2 bc)
{
    if (g_alpha_cutout == 0)
        return true;
    PTMaterial m = g_materials[g_tri_material[prim]];
    float a = m.base_color.a;
    if (m.has_base_tex != 0) {
        uint i0 = g_indices[prim * 3 + 0];
        uint i1 = g_indices[prim * 3 + 1];
        uint i2 = g_indices[prim * 3 + 2];
        float2 uv = (1.0 - bc.x - bc.y) * g_vertices[i0].uv
                    + bc.x * g_vertices[i1].uv + bc.y * g_vertices[i2].uv;
        a *= m.base_tex.SampleLevel(m.samp, uv, 0).a;
    }
    return a >= 0.5;
}

[shader("anyhit")]
void rt_anyhit(inout HitInfo pl, BuiltInTriangleIntersectionAttributes attr)
{
    if (!alpha_passes(PrimitiveIndex(), attr.barycentrics))
        IgnoreHit();
}

[shader("closesthit")]
void rt_closesthit(inout HitInfo pl, BuiltInTriangleIntersectionAttributes attr)
{
    uint prim = PrimitiveIndex();
    float2 bc = attr.barycentrics;
    uint i0 = g_indices[prim * 3 + 0];
    uint i1 = g_indices[prim * 3 + 1];
    uint i2 = g_indices[prim * 3 + 2];
    float w = 1.0 - bc.x - bc.y;
    pl.normal = normalize(w * g_vertices[i0].normal + bc.x * g_vertices[i1].normal
                          + bc.y * g_vertices[i2].normal);
    pl.uv = w * g_vertices[i0].uv + bc.x * g_vertices[i1].uv + bc.y * g_vertices[i2].uv;
    pl.pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    pl.material = g_tri_material[prim];
    pl.hit = 1;
}

[shader("miss")]
void rt_miss(inout HitInfo pl)
{
    pl.hit = 0;
}

bool occluded(float3 o, float3 d, float tmax)
{
    HitInfo pl;
    pl.hit = 1;
    RayDesc r;
    r.Origin = o;
    r.Direction = d;
    r.TMin = 1e-3;
    r.TMax = tmax;
    TraceRay(g_tlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xff, 0, 0, 0, r, pl);
    return pl.hit != 0;
}

// GGX + Lambert BRDF evaluated for a given light direction.
float3 brdf_eval(float3 n, float3 v, float3 l, float3 diff, float3 f0, float alpha)
{
    float ndl = dot(n, l);
    float ndv = dot(n, v);
    if (ndl <= 0.0 || ndv <= 0.0)
        return float3(0.0);
    float3 h = normalize(v + l);
    float ndh = max(dot(n, h), 0.0);
    float vdh = max(dot(v, h), 0.0);
    float3 f = fresnel(vdh, f0);
    float d = d_ggx(ndh, alpha);
    float g = g_smith(ndv, ndl, alpha);
    float3 spec = f * d * g / max(4.0 * ndv * ndl, 1e-6);
    float3 kd = (1.0 - f) * diff / PI;
    return kd + spec;
}

float3 direct_light(float3 p, float3 n, float3 v, float3 diff, float3 f0, float alpha)
{
    float3 sum = float3(0.0);
    for (uint i = 0; i < g_num_dir; i++) {
        DirLight li = g_dir_lights[i];
        float3 l = normalize(-li.direction);
        float ndl = dot(n, l);
        if (ndl <= 0.0)
            continue;
        if (occluded(p + n * 1e-3, l, 1e30))
            continue;
        sum += brdf_eval(n, v, l, diff, f0, alpha) * li.color * li.intensity * ndl;
    }
    for (uint i = 0; i < g_num_point; i++) {
        PointLight li = g_point_lights[i];
        float3 d = li.position - p;
        float dist = length(d);
        float3 l = d / max(dist, 1e-4);
        float ndl = dot(n, l);
        if (ndl <= 0.0)
            continue;
        if (occluded(p + n * 1e-3, l, dist - 1e-3))
            continue;
        float att = 1.0 / max(dist * dist, 1e-4);
        sum += brdf_eval(n, v, l, diff, f0, alpha) * li.color * li.intensity * att * ndl;
    }
    return sum;
}

[shader("raygeneration")]
void rt_raygen()
{
    uint2 tid = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    uint seed = (tid.x * 1973u + tid.y * 9277u + g_frame * 26699u) | 1u;
    pcg(seed);

    float2 jitter = float2(randf(seed), randf(seed));
    float2 uv = (float2(tid) + jitter) / float2(dim);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 wp = mul(g_inv_view_proj, float4(ndc, 1.0, 1.0));

    float3 l = float3(0.0);
    float3 thr = float3(1.0);
    float3 o = g_eye;
    float3 d = normalize(wp.xyz / wp.w - g_eye);

    for (uint bounce = 0; bounce < g_bounces; bounce++) {
        HitInfo pl;
        pl.hit = 0;
        RayDesc r;
        r.Origin = o;
        r.Direction = d;
        r.TMin = bounce == 0 ? 0.0 : 1e-3;
        r.TMax = 1e30;
        TraceRay(g_tlas, RAY_FLAG_NONE, 0xff, 0, 0, 0, r, pl);

        if (pl.hit == 0) {
            l += thr * env_radiance(d);
            break;
        }

        float3 n = pl.normal;
        float3 p = pl.pos;
        float2 huv = pl.uv;
        float3 v = -d;
        if (dot(n, v) < 0.0)
            n = -n;

        PTMaterial m = g_materials[pl.material];
        float4 base = m.base_color;
        if (m.has_base_tex != 0)
            base *= m.base_tex.SampleLevel(m.samp, huv, 0);
        float metallic = m.metallic;
        float rough = m.roughness;
        if (m.has_mr_tex != 0) {
            float4 mr = m.mr_tex.SampleLevel(m.samp, huv, 0);
            rough *= mr.g;
            metallic *= mr.b;
        }
        rough = clamp(rough, 0.03, 1.0);
        float alpha = rough * rough;

        float3 albedo = base.rgb;
        float3 diff = albedo * (1.0 - metallic);
        float3 f0 = lerp(float3(0.04), albedo, metallic);

        float3 emissive = m.emissive;
        if (m.has_emissive_tex != 0)
            emissive *= m.emissive_tex.SampleLevel(m.samp, huv, 0).rgb;
        l += thr * emissive;

        l += thr * direct_light(p, n, v, diff, f0, alpha);

        // Pick a diffuse or specular lobe, then importance sample it.
        float p_spec = clamp(luminance(f0) / max(luminance(f0) + luminance(diff), 1e-4),
                             0.1, 0.9);
        float2 u = float2(randf(seed), randf(seed));
        float3 new_dir;
        float3 weight;
        if (randf(seed) < p_spec) {
            float3 h = ggx_sample_h(n, alpha, u);
            new_dir = reflect(d, h);
            float ndl = dot(n, new_dir);
            if (ndl <= 0.0)
                break;
            float ndv = max(dot(n, v), 1e-4);
            float ndh = max(dot(n, h), 1e-4);
            float vdh = max(dot(v, h), 1e-4);
            float3 f = fresnel(vdh, f0);
            float g = g_smith(ndv, ndl, alpha);
            weight = f * g * vdh / (ndv * ndh) / p_spec;
        } else {
            new_dir = cosine_sample(n, u);
            if (dot(n, new_dir) <= 0.0)
                break;
            weight = diff / (1.0 - p_spec);
        }

        thr *= weight;
        if (bounce >= 3) {
            float pr = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 1.0);
            if (randf(seed) > pr)
                break;
            thr /= pr;
        }

        o = p + n * 1e-3;
        d = new_dir;
    }

    if (any(isnan(l)) || any(isinf(l)))
        l = float3(0.0);

    float4 prev = g_reset != 0 ? float4(0.0) : g_accum[tid];
    float3 sum = prev.rgb + l;
    float count = prev.a + 1.0;
    g_accum[tid] = float4(sum, count);

    float3 c = sum / count * g_exposure;
    c = c / (1.0 + c);
    g_output[tid] = float4(pow(clamp(c, 0.0, 1.0), 1.0 / 2.2), 1.0);
}
