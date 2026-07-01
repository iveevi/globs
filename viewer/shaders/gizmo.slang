struct GVSIn {
    float3 position : POSITION;
    float3 color : COLOR;
};

struct GVSOut {
    float4 clip : SV_Position;
    float3 color : COLOR;
    float3 wpos : POSITION0;
};

struct Camera {
    float4x4 view_proj;
};

ParameterBlock<Camera> g_camera;

uniform uint g_fade_enable;
uniform float3 g_fade_center;
uniform float g_fade_radius;

[shader("vertex")]
GVSOut gizmo_vs(GVSIn in)
{
    GVSOut o;
    o.clip = mul(g_camera.view_proj, float4(in.position, 1.0));
    o.color = in.color;
    o.wpos = in.position;
    return o;
}

[shader("fragment")]
float4 gizmo_fs(GVSOut in)
    : SV_Target
{
    float a = 1.0;
    if (g_fade_enable != 0) {
        float d = distance(in.wpos.xz, g_fade_center.xz);
        a = saturate(1.0 - d / max(g_fade_radius, 1e-3));
        a = a * a;
    }
    float3 c = clamp(in.color, 0.0, 1.0);
    return float4(pow(c, 1.0 / 2.2), a);
}
