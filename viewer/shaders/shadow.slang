struct SVIn {
    float3 position : POSITION;
};

uniform float4x4 g_light_vp;

[shader("vertex")]
float4 shadow_vs(SVIn in)
    : SV_Position
{
    return mul(g_light_vp, float4(in.position, 1.0));
}

[shader("fragment")]
void shadow_fs()
{
}
