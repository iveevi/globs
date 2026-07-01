struct GridVSIn {
    float3 position : POSITION;
};

struct GridVSOut {
    float4 clip : SV_Position;
    float3 wpos : POSITION0;
};

struct Camera {
    float4x4 view_proj;
};

ParameterBlock<Camera> g_camera;

uniform float g_cell;
uniform float3 g_center;
uniform float g_fade_radius;
uniform float3 g_color;
uniform float g_line_width;

[shader("vertex")]
GridVSOut grid_vs(GridVSIn in)
{
    GridVSOut o;
    o.clip = mul(g_camera.view_proj, float4(in.position, 1.0));
    o.wpos = in.position;
    return o;
}

[shader("fragment")]
float4 grid_fs(GridVSOut in)
    : SV_Target
{
    float2 coord = in.wpos.xz / g_cell;
    float2 deriv = fwidth(coord);
    // distance to nearest grid line, measured in pixels
    float2 g = abs(frac(coord - 0.5) - 0.5) / max(deriv, float2(1e-6));
    float dist_px = min(g.x, g.y);
    // line of width g_line_width px with a 1px antialiased falloff
    float coverage = saturate(g_line_width - dist_px);

    float d = distance(in.wpos.xz, g_center.xz);
    float fade = saturate(1.0 - d / max(g_fade_radius, 1e-3));
    fade *= fade;

    float a = coverage * fade;
    if (a <= 0.001)
        discard;
    return float4(pow(clamp(g_color, 0.0, 1.0), 1.0 / 2.2), a);
}
