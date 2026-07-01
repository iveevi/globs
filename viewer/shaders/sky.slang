struct SkyOut {
    float4 clip : SV_Position;
    float2 ndc : TEXCOORD0;
};

Texture2D g_env;
SamplerState g_env_sampler;

uniform float4x4 g_inv_view_proj;
uniform float3 g_eye;
uniform float g_exposure;

static const float PI = 3.14159265358979;

[shader("vertex")]
SkyOut sky_vs(uint vid: SV_VertexID)
{
    float2 c = float2((vid << 1) & 2, vid & 2);
    float2 ndc = c * 2.0 - 1.0;
    SkyOut o;
    o.clip = float4(ndc, 1.0, 1.0);
    o.ndc = ndc;
    return o;
}

[shader("fragment")]
float4 sky_fs(SkyOut in)
    : SV_Target
{
    float4 wp = mul(g_inv_view_proj, float4(in.ndc, 1.0, 1.0));
    float3 dir = normalize(wp.xyz / wp.w - g_eye);

    float u = atan2(dir.x, dir.z) / (2.0 * PI) + 0.5;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    float3 c = g_env.SampleLevel(g_env_sampler, float2(u, v), 0).rgb * g_exposure;
    c = c / (1.0 + c);
    return float4(pow(clamp(c, 0.0, 1.0), 1.0 / 2.2), 1.0);
}
