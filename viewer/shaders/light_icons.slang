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
uniform float2 g_viewport;   // pixels
uniform float g_size;        // icon size in pixels

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

    float4 clip = mul(g_view_proj, float4(L.pos, 1.0));
    if (clip.w <= 0.0) {
        o.clip = float4(0.0, 0.0, -10.0, 1.0);
        return o;
    }
    float2 offset = (quad - 0.5) * g_size * 2.0 / g_viewport;  // centre the quad
    float2 ndc = clip.xy / clip.w + float2(offset.x, -offset.y);
    o.clip = float4(ndc * clip.w, clip.z, clip.w);
    return o;
}

[shader("fragment")]
float4 icon_fs(VOut i)
    : SV_Target
{
    float tile = 1.0 / 3.0;
    float2 auv = float2((i.type + i.uv.x) * tile, i.uv.y);
    float a = g_atlas.SampleLevel(g_atlas_sampler, auv, 0).a;
    if (a <= 0.01)
        discard;
    return float4(i.color, a);
}
