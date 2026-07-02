// Screen-space light markers rendered in the viewport: each light is a
// fixed-pixel-size billboard sampling a Font Awesome glyph from a 3-tile atlas
// (0 = directional/sun, 1 = point/bulb, 2 = spot/bullseye).

struct LightIcon {
    float3 pos;
    float type;
    float3 color;
    float pad;
};

struct VOut {
    float4 clip : SV_Position;
    float2 uv : TEXCOORD0;          // local quad uv [0,1]
    float3 color : COLOR0;
    nointerpolation float type : TEXCOORD1;
};

StructuredBuffer<LightIcon> g_lights;
Texture2D g_atlas;
SamplerState g_atlas_sampler;
uniform float4x4 g_view_proj;
uniform float3 g_cam_right;  // world-space camera right
uniform float3 g_cam_up;     // world-space camera up
uniform float g_size;        // icon half-size in world units
uniform float2 g_viewport;   // pixels
uniform float g_max_px;      // clamp on-screen half-size (pixels)

static const float2 CORNERS[6] = {
    float2(0, 1), float2(1, 1), float2(1, 0),
    float2(0, 1), float2(1, 0), float2(0, 0),
};

[shader("vertex")]
VOut icon_vs(uint vid: SV_VertexID)
{
    uint i = vid / 6;
    uint c = vid % 6;
    LightIcon L = g_lights[i];
    float2 quad = CORNERS[c];

    VOut o;
    o.uv = quad;
    o.color = L.color;
    o.type = L.type;

    // World-space camera-facing billboard: fixed world size, so it scales
    // with distance under perspective projection. Clamp the projected size
    // so it never exceeds g_max_px pixels on screen.
    float size = g_size;
    float4 c0 = mul(g_view_proj, float4(L.pos, 1.0));
    float4 c1 = mul(g_view_proj, float4(L.pos + g_cam_right * g_size, 1.0));
    if (c0.w > 0.0 && c1.w > 0.0) {
        float2 d = (c1.xy / c1.w - c0.xy / c0.w) * 0.5 * g_viewport;
        float px = length(d);
        if (px > g_max_px)
            size *= g_max_px / px;
    }

    float2 offset = (quad - 0.5) * 2.0 * size;
    float3 world = L.pos + g_cam_right * offset.x + g_cam_up * offset.y;
    o.clip = mul(g_view_proj, float4(world, 1.0));
    return o;
}

[shader("fragment")]
float4 icon_fs(VOut i)
    : SV_Target
{
    float tile = 1.0 / 4.0;
    float2 auv = float2((i.type + i.uv.x) * tile, 1.0 - i.uv.y);
    float a = g_atlas.SampleLevel(g_atlas_sampler, auv, 0).a;
    if (a <= 0.01)
        discard;
    return float4(i.color, a);
}
