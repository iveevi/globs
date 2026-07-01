// Screen-space thick lines: each line-list segment (a pair of vertices in
// g_verts) is expanded into a camera-facing quad of fixed pixel width, so the
// light gizmos render as smooth, thick strokes instead of 1px lines.

struct GVert {
    float3 pos;
    float3 color;
};

struct VOut {
    float4 clip : SV_Position;
    float3 color : COLOR0;
};

StructuredBuffer<GVert> g_verts;
uniform float4x4 g_view_proj;
uniform float2 g_viewport;
uniform float g_width;   // line width in pixels

// (endpoint, side) for the two triangles of the quad. endpoint 0 = a, 1 = b.
static const float2 QUAD[6] = {
    float2(0, -1), float2(1, -1), float2(1, 1),
    float2(0, -1), float2(1, 1), float2(0, 1),
};

[shader("vertex")]
VOut line_vs(uint vid: SV_VertexID)
{
    uint seg = vid / 6;
    float2 q = QUAD[vid % 6];

    GVert va = g_verts[seg * 2 + 0];
    GVert vb = g_verts[seg * 2 + 1];
    float4 ca = mul(g_view_proj, float4(va.pos, 1.0));
    float4 cb = mul(g_view_proj, float4(vb.pos, 1.0));

    VOut o;
    o.color = (q.x < 0.5) ? va.color : vb.color;

    // Clip segments crossing the near plane to keep the projection stable.
    if (ca.w <= 1e-4 || cb.w <= 1e-4) {
        o.clip = float4(0.0, 0.0, -10.0, 1.0);
        return o;
    }

    float2 sa = ca.xy / ca.w;   // NDC
    float2 sb = cb.xy / cb.w;
    float2 dir = normalize((sb - sa) * g_viewport);
    if (all(dir == 0.0))
        dir = float2(1.0, 0.0);
    float2 nrm = float2(-dir.y, dir.x) * g_width / g_viewport;  // NDC offset

    float4 c = (q.x < 0.5) ? ca : cb;
    float2 ndc = ((q.x < 0.5) ? sa : sb) + nrm * q.y;
    o.clip = float4(ndc * c.w, c.z, c.w);
    return o;
}

[shader("fragment")]
float4 line_fs(VOut in)
    : SV_Target
{
    return float4(pow(clamp(in.color, 0.0, 1.0), 1.0 / 2.2), 1.0);
}
