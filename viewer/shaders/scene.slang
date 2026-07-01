struct VSIn {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct VSOut {
    float4 clip : SV_Position;
    float3 wnormal : NORMAL;
    float3 wpos : POSITION0;
    float2 uv : TEXCOORD0;
};

struct Camera {
    float4x4 view_proj;
    float3 eye;
    float pad;
};

struct Draw {
    float4 base_color;
    uint has_texture;
    uint has_mr_texture;
    uint has_occ_texture;
    uint has_normal_texture;
    float metallic;
    float roughness;
    float occlusion_strength;
    float pad1;
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

ParameterBlock<Camera> g_camera;
ParameterBlock<Draw> g_draw;

Texture2D g_base_color;
Texture2D g_mr_texture;
Texture2D g_occ_texture;
Texture2D g_normal_texture;
SamplerState g_sampler;

Texture2D g_shadow_map;
SamplerState g_shadow_sampler;

StructuredBuffer<DirLight> g_dir_lights;
StructuredBuffer<PointLight> g_point_lights;
StructuredBuffer<float4> g_sh;

uniform uint g_num_dir;
uniform uint g_num_point;
uniform uint g_shading_mode;
uniform uint g_alpha_cutout;
uniform float3 g_headlight_dir;
uniform float3 g_cam_right;
uniform float3 g_cam_up;
uniform float g_headlight_intensity;
uniform float g_ambient;
uniform float4x4 g_light_vp;
uniform uint g_has_shadow;

[shader("vertex")]
VSOut vs_main(VSIn in)
{
    VSOut o;
    float3 wp = in.position;
    o.wpos = wp;
    o.clip = mul(g_camera.view_proj, float4(wp, 1.0));
    o.wnormal = in.normal;
    o.uv = in.uv;
    return o;
}

float3 encode_srgb(float3 c)
{
    c = clamp(c, 0.0, 1.0);
    return pow(c, 1.0 / 2.2);
}

// Perturb a geometric normal by a tangent-space normal map, deriving the
// tangent frame from screen-space derivatives (no precomputed tangents needed).
float3 perturb_normal(float3 N, float3 wpos, float2 uv, float3 sample_rgb)
{
    float3 map = sample_rgb * 2.0 - 1.0;
    float3 dp1 = ddx(wpos);
    float3 dp2 = ddy(wpos);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);
    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float inv_max = rsqrt(max(dot(T, T), dot(B, B)));
    return normalize(map.x * T * inv_max + map.y * B * inv_max + map.z * N);
}

// Magma colormap (polynomial fit by Matt Zucker), input in [0, 1].
float3 magma(float t)
{
    t = saturate(t);
    const float3 c0 = float3(-0.002136485053939, -0.000749655052795, -0.005386127855323);
    const float3 c1 = float3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
    const float3 c2 = float3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
    const float3 c3 = float3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
    const float3 c4 = float3(52.17613981234068, -27.94360607168351, 12.94416944238394);
    const float3 c5 = float3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
    const float3 c6 = float3(18.65570506591883, -11.48977351997711, -5.601961508734096);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Diffuse irradiance from the 9 SH coefficients (already cosine-convolved).
float3 sh_irradiance(float3 n)
{
    float x = n.x, y = n.y, z = n.z;
    float3 r = g_sh[0].rgb * 0.282095;
    r += g_sh[1].rgb * (0.488603 * y);
    r += g_sh[2].rgb * (0.488603 * z);
    r += g_sh[3].rgb * (0.488603 * x);
    r += g_sh[4].rgb * (1.092548 * x * y);
    r += g_sh[5].rgb * (1.092548 * y * z);
    r += g_sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0));
    r += g_sh[7].rgb * (1.092548 * x * z);
    r += g_sh[8].rgb * (0.546274 * (x * x - y * y));
    return max(r, 0.0);
}

float3 tonemap(float3 c)
{
    c = c / (1.0 + c);
    return pow(clamp(c, 0.0, 1.0), 1.0 / 2.2);
}

// Shadow factor (1 = lit, 0 = shadowed) for the primary directional light,
// sampled from its shadow map with 3x3 PCF.
float dir_shadow(float3 wpos)
{
    if (g_has_shadow == 0)
        return 1.0;
    float4 lc = mul(g_light_vp, float4(wpos, 1.0));
    float3 ndc = lc.xyz / lc.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0)
        return 1.0;

    const float bias = 0.0025;
    const float texel = 1.0 / 2048.0;
    float sum = 0.0;
    for (int oy = -1; oy <= 1; oy++)
        for (int ox = -1; ox <= 1; ox++) {
            float sd = g_shadow_map.SampleLevel(g_shadow_sampler, uv + float2(ox, oy) * texel, 0).r;
            sum += (ndc.z - bias > sd) ? 0.0 : 1.0;
        }
    return sum / 9.0;
}

[shader("fragment")]
float4 fs_main(VSOut in)
    : SV_Target
{
    float3 N = normalize(in.wnormal);

    float4 albedo = g_draw.base_color;
    if (g_draw.has_texture != 0)
        albedo *= g_base_color.Sample(g_sampler, in.uv);

    // alpha cutout: drop fragments below the mask threshold (foliage, grates)
    if (g_alpha_cutout != 0 && albedo.a < 0.5)
        discard;

    // mode 2: unlit albedo
    if (g_shading_mode == 2)
        return float4(encode_srgb(albedo.rgb), 1.0);

    // mode 3: geometric (interpolated vertex) normals
    if (g_shading_mode == 3)
        return float4(N * 0.5 + 0.5, 1.0);

    // mode 4: shading normals (geometric normal perturbed by the normal map)
    if (g_shading_mode == 4) {
        float3 sn = N;
        if (g_draw.has_normal_texture != 0)
            sn = perturb_normal(N, in.wpos, in.uv, g_normal_texture.Sample(g_sampler, in.uv).rgb);
        return float4(sn * 0.5 + 0.5, 1.0);
    }

    // glTF packs metalness in B, roughness in G; occlusion in R of its texture.
    // modes 5/6/7: metallic / roughness / occlusion as a magma heatmap.
    if (g_shading_mode == 5) {
        float m = g_draw.metallic;
        if (g_draw.has_mr_texture != 0)
            m *= g_mr_texture.Sample(g_sampler, in.uv).b;
        return float4(magma(m), 1.0);
    }
    if (g_shading_mode == 6) {
        float r = g_draw.roughness;
        if (g_draw.has_mr_texture != 0)
            r *= g_mr_texture.Sample(g_sampler, in.uv).g;
        return float4(magma(r), 1.0);
    }
    if (g_shading_mode == 7) {
        float o = 1.0;
        if (g_draw.has_occ_texture != 0)
            o = 1.0 + g_draw.occlusion_strength * (g_occ_texture.Sample(g_sampler, in.uv).r - 1.0);
        return float4(magma(o), 1.0);
    }

    float3 V = normalize(g_camera.eye - in.wpos);
    if (dot(N, V) < 0.0)
        N = -N;

    // mode 1: Solid -- Blender-style viewport shading. A view-space studio
    // light rig (key + fill) over a hemispheric ambient gradient, plus a soft
    // specular sheen. Lighting follows the camera, ignoring scene lights.
    if (g_shading_mode == 1) {
        float3 base = float3(0.40);
        float3 to_viewer = -g_headlight_dir;
        float3 key_L = normalize(g_cam_right * 0.45 + g_cam_up * 0.65 + to_viewer * 0.55);
        float3 fill_L = normalize(-g_cam_right * 0.55 + g_cam_up * 0.10 + to_viewer * 0.50);
        float up = saturate(N.y * 0.5 + 0.5);
        float3 hemi = lerp(float3(0.18, 0.18, 0.21), float3(0.40, 0.41, 0.44), up);
        float key = max(dot(N, key_L), 0.0) * 0.50 * g_headlight_intensity;
        float fill = max(dot(N, fill_L), 0.0) * 0.12;
        float3 lit = base * (hemi * (0.5 + g_ambient) + key + fill);
        float3 H = normalize(key_L + V);
        lit += pow(max(dot(N, H), 0.0), 32.0) * 0.06 * g_headlight_intensity;
        return float4(encode_srgb(lit), 1.0);
    }

    // mode 0: Lit -- environment SH irradiance plus the punctual lights.
    float3 lit = albedo.rgb * sh_irradiance(N);
    for (uint i = 0; i < g_num_dir; i++) {
        DirLight l = g_dir_lights[i];
        float3 L = normalize(-l.direction);
        float shadow = (i == 0) ? dir_shadow(in.wpos) : 1.0;
        lit += albedo.rgb * l.color * l.intensity * max(dot(N, L), 0.0) * shadow;
    }
    for (uint i = 0; i < g_num_point; i++) {
        PointLight l = g_point_lights[i];
        float3 d = l.position - in.wpos;
        float dist = length(d);
        float3 L = d / max(dist, 1e-4);
        float att = 1.0 / (1.0 + dist * dist);
        lit += albedo.rgb * l.color * l.intensity * att * max(dot(N, L), 0.0);
    }
    return float4(tonemap(lit), 1.0);
}
